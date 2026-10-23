"""
smoke.py + expert-cache logging.

Same model / engine config / generation settings as smoke.py. Runs the prompt
N_REPEATS times in one process so you can see the cold-cache request (all misses)
vs. the warm-cache requests (mostly hits).

Outputs (in $LOG_DIR, default ./logs):
  <tag>_layers.csv    one row per (request, forward step, MoE layer)  [ExpertCacheLogger]
  <tag>_requests.csv  one row per request: latency, token counts, cache counters
  stdout              C++ per-expert HIT/MISS/evict lines if MUDUO_LOG_DEBUG=1
                      -> parse with parse_debug_log.py
"""
import csv
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from moe_infinity import MoE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_cache_logger import ExpertCacheLogger  # noqa: E402

N_REPEATS = int(os.environ.get("N_REPEATS", "3"))
SYNC_LAYERS = os.environ.get("SYNC_LAYERS", "0") == "1"
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
TAG = os.environ.get("RUN_TAG", time.strftime("%Y%m%d-%H%M%S"))
PROMPT = os.environ.get("PROMPT", "Explain what a GPU is in one sentence.")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, "Expected one visible GPU"
gpu = torch.cuda.get_device_name(0)
assert "RTX 6000 Ada" in gpu, gpu
print("GPU:", gpu, "| PyTorch:", torch.__version__, flush=True)

root = Path(os.environ["MOE_WORK_ROOT"])
checkpoint = root / "models" / "Qwen3-30B-A3B-ad44e777"
offload = root / "state" / "offload-qwen3-bf16-moe022"
offload.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
model = MoE(
    str(checkpoint),
    {
        "offload_path": str(offload),
        "device_memory_ratio": 0.5,
        "num_threads": 4,
        "use_native_engine": False,
        "adaptive_memory_enabled": False,
        "adaptive_expert_precision": False,
    },
)
assert getattr(model.engine, "dtype_cls", None) == torch.bfloat16

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = ExpertCacheLogger(model, str(LOG_DIR / f"{TAG}_layers.csv"),
                           sync_layers=SYNC_LAYERS)

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
n_in = int(inputs["input_ids"].shape[1])

req_rows = []
for i in range(N_REPEATS):
    before = logger.snapshot()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with logger.request(f"r{i}", run=TAG), torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    after = logger.snapshot()

    gen = output[0, n_in:]
    req_rows.append(dict(
        run=TAG, request=f"r{i}", latency_s=f"{latency:.4f}",
        input_tokens=n_in, output_tokens=int(gen.shape[0]),
        h2d_bytes=after["h2d_bytes_total"] - before["h2d_bytes_total"],
        accesses=after["cum_accesses"] - before["cum_accesses"],
        resident_before=before["resident_experts"],
        resident_after=after["resident_experts"],
        text=tokenizer.decode(gen, skip_special_tokens=True),
    ))
    r = req_rows[-1]
    print(f"[{r['request']}] {r['latency_s']}s in={n_in} out={r['output_tokens']} "
          f"h2d={r['h2d_bytes'] / 2**20:.1f}MiB resident {r['resident_before']}->"
          f"{r['resident_after']} | {r['text']!r}", flush=True)

logger.close()
with open(LOG_DIR / f"{TAG}_requests.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(req_rows[0].keys()))
    w.writeheader()
    w.writerows(req_rows)
print("LOGS:", LOG_DIR / f"{TAG}_layers.csv", LOG_DIR / f"{TAG}_requests.csv", flush=True)
print("SMOKE_LOGGED_OK", flush=True)
