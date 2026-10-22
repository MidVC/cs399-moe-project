import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer
from moe_infinity import MoE

DEFAULT_PROMPT = "Explain what a GPU is in one sentence."

# Greedy decoding with everything pinned so repeated runs are comparable.
# temperature/top_p/top_k are set to their "no-op" values so Qwen's
# generation_config.json sampling defaults are overridden. They can't be None:
# MoE.generate() calls float() on top_p.
GEN_SETTINGS = {
    "max_new_tokens": 16,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "num_beams": 1,
    "use_cache": True,
}


def load_model():
    assert torch.cuda.is_available(), "CUDA unavailable"

    root = Path(os.environ["MOE_WORK_ROOT"])
    checkpoint = root / "models" / "Qwen3-30B-A3B-ad44e777"
    offload = root / "state" / "offload-qwen3-bf16-moe022"
    offload.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        local_files_only=True,
    )

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
    return model, tokenizer


def run_prompt(model, tokenizer, prompt, gen_settings=GEN_SETTINGS):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda:0")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            **gen_settings,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    generated = output[0, input_len:]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "chat_text": text,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "input_tokens": int(input_len),
        "output_tokens": int(generated.shape[0]),
        "hit_eos": bool(generated.numel() and generated[-1].item() == tokenizer.eos_token_id),
        "latency_s": latency,
        "gen_settings": gen_settings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="results/fixed_prompt.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model()
    record = run_prompt(model, tokenizer, args.prompt)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(json.dumps(record, indent=2), flush=True)
    print(f"Saved to {out}", flush=True)


if __name__ == "__main__":
    main()
