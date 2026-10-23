# MoE-Infinity expert-cache experiments (Qwen3-30B-A3B)

Group repo for the CS399 MoE expert-cache project. It combines:

- [YuhanWu2004/Moe-cache-logging](https://github.com/YuhanWu2004/Moe-cache-logging) (steps 1–2), in `moe-cache-logging/`
- [brKhoo/moe-experiments](https://github.com/brKhoo/moe-experiments) (step 3), at the repo root

Both build on the shared setup [w00606/moe-setup](https://github.com/w00606/moe-setup). The git history of both source repos is kept.

## Layout

| Path | Step | What it is |
|---|---|---|
| `setup.sh`, `run.sh`, `smoke.py`, `*.lock` | — | Shared environment setup from `w00606/moe-setup` |
| `moe-cache-logging/` | 1–2 | Code map of expert residency, loading and eviction in MoE-Infinity, plus the cache loggers. See [its README](moe-cache-logging/README.md). |
| `moe-cache-logging/logs/` | 1–2 | Sample logging run `20261018-011323` |
| `fixed_prompt.py` | 3 | Fixed-prompt script (below) |
| `results/` | 3 | Step 3 results |

Steps 4 and 5 will be added here.

`run.sh` and `moe-cache-logging/run_logged.sh` both read the work root from `.work-root` in the repo root, which `setup.sh` writes. `run_logged.sh` expects to sit one folder below `run.sh`, as it does here.

## Step 3: fixed-prompt script

`fixed_prompt.py` sends one prompt to the model with sampling disabled and fixed generation settings. Each run appends one JSON line to `results/fixed_prompt.jsonl` with:

| Field | Meaning |
|---|---|
| `prompt` | The user prompt |
| `chat_text` | The prompt after Qwen's chat template is applied |
| `generated_text` | The model's output |
| `input_tokens` | Prompt length in tokens, chat template included |
| `output_tokens` | Number of newly generated tokens |
| `hit_eos` | Whether generation stopped on the end-of-sequence token rather than the length limit |
| `latency_s` | Wall time of `model.generate()` |
| `gen_settings` | The exact generation settings used |

### Generation settings

```python
max_new_tokens = 16     
do_sample      = False  # greedy decoding
temperature    = 1.0    # no-op values that override Qwen's sampling defaults
top_p          = 1.0    # (generation_config.json); ignored when do_sample=False
top_k          = 0
num_beams      = 1
use_cache      = True   # KV cache on
```

Engine config is the same as in `smoke.py`: `device_memory_ratio=0.5`, `num_threads=4`, `use_native_engine=False`, adaptive memory and precision off.

With `use_native_engine=False`, `MoE.generate()` passes these arguments unchanged to Hugging Face `generate()`. The sampling values can't be `None`, because `MoE.generate()` calls `float()` on `top_p`.

### How to run

From inside a GPU session on WATGPU, after `bash setup.sh` has finished:

```bash
salloc --gres=gpu:1 --cpus-per-task=2 --mem=192G --time=04:00:00 --constraint=ADA6000 -w watgpu208
cd ~/moe-setup
bash run.sh fixed_prompt.py                                   # default prompt
bash run.sh fixed_prompt.py --prompt "..." --out results/x.jsonl
```

### Results

`results/fixed_prompt.jsonl`, two separate runs on watgpu208 (RTX 6000 Ada), 2026-10-22:

| Run | Input tokens | Output tokens | Latency | Generated text |
|---|---|---|---|---|
| 1 | 22 | 16 | 6.14 s | "A GPU, or Graphics Processing Unit, is a specialized electronic circuit designed to rapidly" |
| 2 | 22 | 16 | 6.58 s | identical |

- Both runs produced identical text, which confirms decoding is deterministic.
- The text is cut off because generation stops at `max_new_tokens=16` (`hit_eos: false`). Every request therefore has the same output length, so differences in latency don't come from output length.
- Each run is a new process, so the GPU expert cache starts empty both times. These are cold-cache latencies, which agree with the 5.97 s first request in the step 1–2 logs.

## Environment

- Qwen/Qwen3-30B-A3B (bf16, revision `ad44e777`)
- MoE-Infinity 0.2.2, moe-store 0.2.2, PyTorch 2.11.0+cu128, transformers 5.3.0
- WATGPU node watgpu208: one NVIDIA RTX 6000 Ada (48 GB), 2 CPUs, 192 GB RAM
