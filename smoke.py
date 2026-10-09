import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from moe_infinity import MoE

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, "Expected one visible GPU"

gpu = torch.cuda.get_device_name(0)
assert "RTX 6000 Ada" in gpu, gpu
assert torch.cuda.is_bf16_supported()

print("GPU:", gpu, flush=True)
print("PyTorch:", torch.__version__, flush=True)

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

dtype = getattr(model.engine, "dtype_cls", None)
assert dtype == torch.bfloat16, f"Unexpected dtype: {dtype}"
print("Engine dtype:", dtype, flush=True)

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain what a GPU is in one sentence."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False,
).to("cuda:0")

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()

generated = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True), flush=True)
print("SMOKE_OK", flush=True)
