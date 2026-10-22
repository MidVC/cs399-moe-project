# MoE-Infinity fixed-prompt runs (Qwen3-30B-A3B)

Built on the shared setup [w00606/moe-setup](https://github.com/w00606/moe-setup)

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

Loading the model takes about 2 minutes. Don't press Ctrl-C during loading. If you do, the process can keep running in the background and hold the lock in `run.sh`; `run.sh` now prints a message when that happens.

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

### WATGPU notes

- Home (`/u801/...`) is mounted `noexec` on the login node but not on the GPU nodes. Run `setup.sh` and everything else inside `salloc`, not on the login node.
- `watgpu108` has a broken GPU; exclude it with `-x watgpu108`.

## Changes from moe-setup

- `fixed_prompt.py`, `results/`, and `README.md` are new.
- `run.sh` takes the script to run as an argument (`bash run.sh fixed_prompt.py`; the default is still `smoke.py`), and prints a message instead of exiting silently when another run holds the lock.
