# Step 5: Does prompt order change latency? (Qwen3-30B-A3B on MoE-Infinity)

Step 5 builds on step 3 (`fixed_prompt.py`) for model loading, prompts, generation settings and timing. It uses steps 1–2 (`moe-cache-logging/expert_cache_logger.py`) for the expert-cache logs. Terms follow the steps 1–2 write-up (`moe-cache-logging/README.md` and the group's Google Doc).

## Summary

Each prompt is about **32% faster when the other prompt ran just before it** in the same process:

- A: 9.04 s → 6.06 s
- B: 8.41 s → 5.73 s

These are medians of 10 trials. The standard deviation in every condition is at most 0.075 s.

The cache logs explain most of the difference. The second prompt finds many of its experts already on the GPU, so it copies about 13–14 GiB fewer experts from CPU to GPU. The whole gain is in prefill; decode is unaffected.

## Key terms

| Term | Meaning |
|---|---|
| Expert cache | The ~2,200 experts (of 6,144) that MoE-Infinity keeps in GPU memory. The rest stay in CPU memory. |
| Hit / miss | Hit: an expert the router picked is already on the GPU. Miss: it has to be copied in first. |
| H2D copy | Copying one expert (about 9 MiB) from CPU memory to the GPU. Every miss is one copy, and copies are the slow part. |
| Prefill | The model reads the whole prompt in one pass. Each of the 22 prompt tokens picks 8 experts per layer, so prefill touches many experts at once. |
| Decode | The model then writes the answer one token at a time, touching at most 8 experts per layer per step. Here, 1 prefill pass and 15 decode steps produce the 16 tokens. |
| First (cold) | The prompt ran first in a new process, on an empty expert cache. |
| After | The other prompt ran just before it in the same process. |

## Setup

| | |
|---|---|
| Prompt A | "Explain what a GPU is in one sentence." (step 3's default; 22 input tokens) |
| Prompt B | "Write a haiku about autumn leaves." (20 input tokens) |
| Generation | Step 3's `GEN_SETTINGS`: greedy, `max_new_tokens=16`. Every request produced exactly 16 tokens. |
| Engine | Same as `smoke.py`: `device_memory_ratio=0.5`, `num_threads=4`, `use_native_engine=False`, adaptive memory and precision off |
| Trials | 10 per order, alternating AB, BA, AB, … Each trial is a new process, so every pair starts from an empty GPU expert cache. |
| Logging | `ExpertCacheLogger` with `sync_layers=False`, so logging doesn't perturb latency. No C++ debug log. |
| Hardware | watgpu802: one RTX 6000 Ada (48 GB), 4 CPUs, 192 GB RAM. SLURM job 1641492, 2026-10-25. |

## Code

All new code is in the folder `order-effect/`:

| File | What it does |
|---|---|
| `order_effect.py` | Loads the model once (`fixed_prompt.load_model`) and attaches `ExpertCacheLogger`. Runs the prompts in the given order with `fixed_prompt.run_prompt`. Appends one record per request to `requests.jsonl`, and writes `trial<NN>_<order>_layers.csv` (one row per request × forward step × MoE layer). |
| `run_order_effect.sh` | Runs `order_effect.py` through `run.sh` for each trial and order. Refreshes `summary.txt` after every round, so a run cut off by the time limit is still summarized. |
| `summarize.py` | Builds `summary.txt`: sanity checks, latency and cache counters per condition, the prefill/decode split, and the expert overlap between A and B. |

Each line of `requests.jsonl` has the step 3 record plus these fields:

- `trial`, `order`, `position`, `prompt_id`, `prev_prompt_id`
- `h2d_bytes`, `accesses`, `hits`, `misses`
- `resident_before`, `resident_after`

How to run, inside a `salloc` session on an RTX 6000 Ada node:

```bash
cd ~/cs399-moe-project
bash run.sh                                       # smoke test first
bash order-effect/run_order_effect.sh             # defaults: N_TRIALS=10, ORDERS="AB BA"
ORDERS="AB BA AA BB" bash order-effect/run_order_effect.sh   # with same-prompt controls
bash order-effect/run_order_effect.sh --prompt-b "..."       # other prompts (--prompt-a too)
python3 order-effect/summarize.py order-effect/results/<tag>
```

## Results

Run `job1641492`: 20 processes, 40 requests.

### Latency

| Condition | n | Median | Mean | SD | Min | Max |
|---|---|---|---|---|---|---|
| A first (cold) | 10 | 9.038 s | 9.025 s | 0.053 s | 8.891 s | 9.083 s |
| A after B | 10 | 6.062 s | 6.034 s | 0.075 s | 5.889 s | 6.093 s |
| B first (cold) | 10 | 8.406 s | 8.410 s | 0.049 s | 8.308 s | 8.473 s |
| B after A | 10 | 5.728 s | 5.712 s | 0.059 s | 5.590 s | 5.778 s |

Running second saves 2.98 s for A (−32.9%) and 2.68 s for B (−31.9%). That is about 40–55 times the standard deviation.

### Cache counters (mean per request)

| Condition | Copied CPU→GPU | Hits | Misses | Hit rate |
|---|---|---|---|---|
| A first | 39.23 GiB | 4417 | 4137 | 51.6% |
| A after B | 22.09 GiB | 6041 | 2513 | 70.6% |
| B first | 35.89 GiB | 4677 | 3757 | 55.5% |
| B after A | 20.40 GiB | 6112 | 2322 | 72.5% |

### Prefill vs. decode (mean per request)

| Condition | Prefill misses | Prefill copied | Prefill MoE time | Decode misses | Decode copied | Decode MoE time |
|---|---|---|---|---|---|---|
| A first | 2794 | 24.56 GiB | 4542 ms | 1343 | 11.81 GiB | 526 ms |
| A after B | 1114 | 9.79 GiB | 2479 ms | 1399 | 12.29 GiB | 520 ms |
| B first | 2674 | 23.50 GiB | 4320 ms | 1083 | 9.52 GiB | 519 ms |
| B after A | 1186 | 10.43 GiB | 2556 ms | 1135 | 9.98 GiB | 514 ms |

MoE time is the summed wall time of the MoE layers. The layers aren't CUDA-synchronized, so it is approximate.

In the "first" rows, the per-layer copies add up to about 2.9 GiB less than the per-request totals in the cache-counter table (A: 36.36 vs. 39.23 GiB). In the "after" rows they match. The steps 1–2 write-up found the same extra ~2.9 GiB in the first request of a process, probably the non-expert weights reaching the GPU for the first time (not yet confirmed). So of the drop in data copied, about 14.3 GiB (A) and 12.6 GiB (B) is fewer expert copies, and the rest is this one-time copy.

### Expert overlap between A and B

These are the distinct (layer, expert) pairs each prompt was routed to, taken from trial 1:

| Phase | A | B | Shared | Jaccard | Shared / A | Shared / B |
|---|---|---|---|---|---|---|
| Prefill | 2794 | 2674 | 2227 | 0.687 | 79.7% | 83.3% |
| Decode | 1814 | 1596 | 774 | 0.294 | 42.7% | 48.5% |
| All | 3363 | 3127 | 2572 | 0.656 | 76.5% | 82.3% |

The GPU cache holds about 2,200 of the 6,144 experts (48 layers × 128). That is fewer than one request uses.

## Findings

1. **Order changes latency by about 3 s (~32%).** The effect is the same size in both directions and very stable across trials.
2. **The gain comes from prefill.** When the other prompt ran first:
   - prefill misses drop by more than half (A: 2,794 → 1,114; B: 2,674 → 1,186);
   - prefill MoE time drops by about 2 s;
   - decode misses and decode time stay the same, because by the time decoding starts, the request's own prefill has already loaded what decode needs.
3. **Shared experts explain why.** About 80% of the experts each prompt uses in prefill are also used by the other prompt, so whichever runs first leaves many of them on the GPU. The carry-over can only come from the expert cache: the KV cache is cleared at the start of every request (steps 1–2 write-up).
4. **Cache size matters.** The cache held 2,198 experts in 17 of the 20 processes and 2,264 in the other 3 (trial 1 BA, and trial 9 AB and BA). In each of the four conditions, the fastest run came from one of those 3 processes.
5. **Greedy output is not fully deterministic.**
   - B produced 7 distinct texts in 20 runs, including runs that started from the same empty cache.
   - A produced 2. In one run (trial 9 BA) it ended "…specialized processor designed to rapidly manipulate".
   - This contradicts step 3's conclusion, which was based on two runs.
   - A likely cause, not yet verified: experts finish in a timing-dependent order, so BF16 results are summed in different orders and the rounding flips near-tied tokens.
   - The latency comparison is unaffected, because every request still produced exactly 16 tokens. Routing during decode can differ slightly between runs, though, and the overlap table uses trial 1 only.

## Limitations and next steps

- **About 0.9 s of each speedup doesn't appear in the MoE-layer timings.** For A, the total drop is 2.98 s but the summed MoE time drops by only 2.07 s; for B, 2.68 s vs. 1.77 s. Part of it is probably one-time work in the first request of each process, such as the extra ~2.9 GiB copy described under the prefill/decode table, rather than an expert-cache effect. Same-prompt controls (`ORDERS="AB BA AA BB"`) would separate the two.
- **Only one pair of prompts was tested.** Both are short English instructions with high overlap. A dissimilar prompt, such as code or another language, would show whether the speedup tracks expert overlap.
- **Absolute latency depends on the node and settings.** Cold A took 9.0 s on watgpu802 with 4 CPUs, but about 6.0 s in steps 1–3 on watgpu208 with 2 CPUs, with the same ~39 GiB copied. The steps 1–2 write-up estimates about 0.4 ms per expert copy on watgpu208. Step 5 didn't measure copy times, so it can't say whether slower copies explain the gap. Compare numbers within one run only.
- **The logs don't say which individual experts hit or missed.** The C++ debug log (`MUDUO_LOG_DEBUG=1`) was off to keep timings clean, so the logs give counts per layer.

## Run notes

- On 2026-10-25, watgpu208 couldn't run CUDA. Opening `/dev/nvidia-uvm` failed with an I/O error, so PyTorch reported "CUDA unknown error" even though `nvidia-smi` worked. The job was resubmitted with `--exclude=watgpu208`. Steps 1–2 hit the same error on watgpu108, so the fault has now been seen on two nodes.
- The first run on a new work root builds MoE-Infinity's offload store. The smoke test plus that build took about 6 minutes. Each trial process took about 80 s, mostly model loading. The whole run took 34 minutes.
- Results are in [`results/job1641492/`](results/job1641492/): `summary.txt`, `requests.jsonl`, and one layer CSV per trial. The raw stdout log stays on the cluster, because `*.log` files are git-ignored.
