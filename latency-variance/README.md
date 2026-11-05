# Step 4: Latency variance for a repeated prompt (Qwen3-30B-A3B on MoE-Infinity)

Step 4 sends the same prompt with the same greedy generation settings many times, records each request's latency and output-token count, and summarizes the latency distribution. It builds on step 3 (`fixed_prompt.py`) for the model, prompt, settings and timing, and on steps 1–2 (`moe-cache-logging/expert_cache_logger.py`) for cache counters that can explain the variance. Terms follow the steps 1–2 write-up (`moe-cache-logging/README.md`) and step 5 (`order-effect/README.md`).

This is the noise baseline for the project: step 5 found that running after another prompt saves about 3 s, and that signal is only useful to an attacker if it is large compared to the natural spread measured here.

## Summary

**Once the first request of a process is excluded, latency is very stable.** Warm requests have a median of 4.39 s and a standard deviation of 0.05–0.08 s (CV under 2%).

- The first request in each process is about **2× slower** (8.96 s), because it starts from an empty GPU expert cache and includes one-time work. This is the only large source of variance.
- The small warm-request spread has three visible sources:
  - cache capacity, which differs between processes;
  - a slow drift of about +0.1 s over the first ~20 requests, as the cache settles;
  - about ±0.05 s of noise that the cache counters don't explain.
- The order effects found so far are **20–60× larger than this noise**. These numbers are a best case: one user, no concurrent load, no network.

## Key terms

| Term | Meaning |
|---|---|
| Cold | The first request in a new process. The GPU expert cache starts empty. |
| Warm | Every later request in the same process. |
| Within-process spread | How much warm latency varies inside one process. |
| Between-process spread | How much the warm *mean* differs from one process to the next. |
| CV | Coefficient of variation: SD / mean. |

## Setup

| | |
|---|---|
| Prompt | "Explain what a GPU is in one sentence." (step 3's default, prompt A in step 5; 22 input tokens) |
| Generation | Step 3's `GEN_SETTINGS`: greedy, `max_new_tokens=16`. Every request produced exactly 16 tokens. |
| Engine | Same as `smoke.py`: `device_memory_ratio=0.5`, `num_threads=4`, `use_native_engine=False`, adaptive memory and precision off |
| Logging | `ExpertCacheLogger` with `sync_layers=False`, so logging doesn't perturb latency. No C++ debug log. |
| Runs | `within`: 1 process × 50 requests. `across`: 5 processes × 20 requests. Each process starts from an empty GPU expert cache. Requests run back to back. |
| Hardware | watgpu802: one RTX 6000 Ada (48 GB), 4 CPUs, 192 GB RAM. SLURM job 1654577, 2026-10-29. This is the same node and CPU count as step 5, so the numbers are directly comparable. |

## Code

All new code is in the folder `latency-variance/`:

| File | What it does |
|---|---|
| `repeat_latency.py` | Loads the model once (`fixed_prompt.load_model`), attaches `ExpertCacheLogger`, and calls `fixed_prompt.run_prompt` N times. Appends one record per request to `requests.jsonl` and writes `proc<NN>_layers.csv`. |
| `run_repeat.sh` | Runs `repeat_latency.py` through `run.sh` once per process. Records the node, `nvidia-smi` state and other GPU processes in `stdout.log`, and refreshes `summary.txt` after every process. |
| `summarize.py` | Builds `summary.txt` and `latency.csv` (one row per request, for plotting). The summary covers: sanity checks; the latency distribution for cold, warm and each process; within- vs between-process spread; a per-request-index table; and how warm latency correlates with misses, H2D bytes and prefill/decode time. |
| `step4.sbatch` | Batch job that runs the smoke test and both runs. |

Each line of `requests.jsonl` has the step 3 record plus these fields:

- `proc`, `req`, `node`, `slurm_job`, `cpus`, `gpu`
- `h2d_bytes`, `accesses`, `hits`, `misses`
- `resident_before`, `resident_after`

How to run, from the repo root on the login node (`setup.sh` must have been run once, on a compute node; home is mounted `noexec` on the login node):

```bash
sbatch latency-variance/step4.sbatch                                    # everything, ~40 min
# or, inside an salloc session on an RTX 6000 Ada node:
N_PROCS=1 N_REQUESTS=50 RUN_TAG=within bash latency-variance/run_repeat.sh
N_PROCS=5 N_REQUESTS=20 RUN_TAG=across bash latency-variance/run_repeat.sh
python3 latency-variance/summarize.py latency-variance/results/<tag>    # re-summarize
```

## Results

Run `job1654577`: 6 processes, 150 requests.

### Latency distribution

| Group | n | Median | Mean | SD | CV | p5 | p95 | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| Cold, `within` | 1 | 8.944 s | 8.944 s | – | – | – | – | 8.944 s | 8.944 s |
| Cold, `across` | 5 | 8.965 s | 8.960 s | 0.040 s | 0.4% | 8.907 s | 8.992 s | 8.894 s | 8.994 s |
| Warm, `within` | 49 | 4.400 s | 4.393 s | 0.052 s | 1.2% | 4.310 s | 4.459 s | 4.229 s | 4.492 s |
| Warm, `across` | 95 | 4.385 s | 4.374 s | 0.078 s | 1.8% | 4.229 s | 4.494 s | 4.164 s | 4.533 s |
| All, `across` | 100 | 4.388 s | 4.603 s | 1.007 s | 21.9% | 4.229 s | 4.751 s | 4.164 s | 8.994 s |

Including the cold request raises the SD from 0.08 s to 1.0 s. So the first request alone accounts for almost all of the variance.

### Warm latency per process (`across`)

| Process | Cache capacity (experts) | Warm mean | Warm SD |
|---|---|---|---|
| 1 | 2264 | 4.282 s | 0.060 s |
| 2 | 2198 | 4.414 s | 0.069 s |
| 3 | 2198 | 4.384 s | 0.067 s |
| 4 | 2198 | 4.425 s | 0.045 s |
| 5 | 2198 | 4.366 s | 0.057 s |

- Mean within-process SD: 0.060 s.
- SD of the process means: 0.057 s.

So the within-process spread and the between-process spread are about the same size. The single `within` process also had a capacity of 2,264 experts.

### Cache counters over a process (`within`, selected requests)

| Request | Latency | Misses | H2D | Prefill misses | Prefill MoE ms | Decode misses | Decode MoE ms |
|---|---|---|---|---|---|---|---|
| 0 (cold) | 8.944 s | 4102 | 38.92 GiB | 2794 | 4556 | 1308 | 521 |
| 1 | 4.337 s | 1405 | 12.35 GiB | 885 | 2052 | 520 | 486 |
| 2 | 4.365 s | 1254 | 11.02 GiB | 1070 | 2417 | 184 | 470 |
| 5 | 4.229 s | 1163 | 10.22 GiB | 1127 | 2417 | 36 | 461 |
| 10 | 4.318 s | 1205 | 10.59 GiB | 1147 | 2496 | 58 | 463 |
| 20 | 4.444 s | 1215 | 10.68 GiB | 1213 | 2662 | 2 | 459 |
| 30 | 4.403 s | 1231 | 10.82 GiB | 1221 | 2613 | 10 | 460 |
| 49 | 4.445 s | 1231 | 10.82 GiB | 1230 | 2667 | 1 | 461 |

The full per-request table is in `results/within-1654577/summary.txt`. MoE ms is the summed wall time of the MoE layers. The layers aren't CUDA-synchronized, so it is approximate.

### What tracks warm latency

Correlation with warm latency:

| Variable | `within` (n=49) | `across` (n=95) |
|---|---|---|
| Total misses | 0.25 | 0.66 |
| Prefill misses | 0.49 | 0.32 |
| Decode misses | −0.25 | 0.18 |
| Prefill MoE time | 0.69 | 0.42 |
| Decode MoE time | −0.09 | 0.23 |
| Cache capacity (`resident_before`) | – (constant) | −0.60 |
| Output tokens | – (always 16) | – (always 16) |

## Findings

1. **The first request is the only large source of variance.** Cold requests take 8.96 s and warm ones 4.39 s, so the first request is about 2× slower. Cold requests are themselves very consistent: SD 0.04 s over 5 processes. Cold latency matches step 5's "A first" (9.04 s) on the same node.
2. **Warm latency is tight.** The SD is 0.05 s within one process and 0.08 s across five (CV 1.2–1.8%). The p5–p95 range is about 0.15–0.27 s.
3. **Cache capacity changes warm latency by about 0.1 s.** Process 1 got a cache of 2,264 experts and the others got 2,198. Process 1 was the fastest (4.28 s vs 4.37–4.43 s), and capacity correlates with latency at −0.60. Step 5 saw the same pattern (its finding 4). Here it rests on one process with the larger cache.
4. **The cache settles over about 20 requests, and latency drifts up slightly.**
   - Decode misses fall from 520 to almost 0.
   - Prefill misses rise from 885 to about 1,230.
   - The total stays about 1,220.
   - Latency rises by about 0.1 s and then levels off. Prefill MoE time tracks latency best (correlation 0.69 within one process).
   - A likely cause: the eviction rule keeps the experts that have been loaded most often (Part 1, `FindExpertEvict`), and those counts are never reset. So the cache gradually keeps the experts decode needs and churns the ones prefill needs. This is not directly verified.
5. **Most of the remaining noise is unexplained.** Within one process, total misses correlate with latency at only 0.25. The leftover ±0.05 s probably comes from sources the cache logs don't see, such as CPU scheduling or the timing of the PCIe copies.
6. **Greedy output is not fully deterministic** (this confirms step 5's finding 5).
   - The prompt produced 3 distinct texts within one process and 4 across 100 requests.
   - Two cold requests (process 1 and process 4, request 0) produced different texts, even though both started from an empty cache.
   - All requests still produced 16 tokens, so latency is unaffected.

### Comparison with step 5

| Condition | Source | Median | Difference from "A after A" |
|---|---|---|---|
| A after A (2nd request in process) | step 4, request 1 mean | 4.41 s | – |
| A after B | step 5 | 6.06 s | +1.65 s (~20× warm SD) |
| A first (cold) | step 5 | 9.04 s | +4.63 s (~60× warm SD) |

Every row is the second request in its process, except the cold one. So the 1.65 s gap reflects only which prompt ran before. It is the comparison the attack depends on. The two halves come from different jobs, but on the same node and with the same settings. Processes within this job differed by at most 0.14 s.

### An unexplained gap

Latency minus summed MoE-layer time:

| Condition | Latency | MoE time (prefill + decode) | Remainder |
|---|---|---|---|
| A cold (step 4) | 8.96 s | 5.07 s | ~3.9 s |
| A after B (step 5) | 6.06 s | 3.00 s | ~3.1 s |
| A after A (step 4) | 4.39 s | 3.04 s | ~1.35 s |

The time outside the MoE layers drops by about 1.7 s when the same prompt repeats. A after B has about 1,050 more expert misses than A after A, but only about 0.4 s more MoE time. Possible causes, not yet tested:

- One-time work the first time each input shape is seen. A has 22 input tokens and B has 20, so A after B sees a new shape.
- GPU work being counted outside the MoE layers, because the per-layer timings aren't synchronized.

This is related to step 5's limitation that about 0.9 s of its speedup doesn't appear in the MoE timings. Until it is explained, part of the order effect may not come from the expert cache.

## Limitations and next steps

- **This is a best-case noise floor.** There was one user, no concurrent requests, no network, and latency was timed on the server. A remote attacker would see more noise.
- **One job on one node.** Variance between jobs and between nodes was not measured. Step 5 saw cold A at 6.0 s on watgpu208 and 9.0 s on watgpu802.
- **One short prompt, 16 output tokens.** Longer outputs may be noisier.
- **The A-after-A vs A-after-B comparison crosses jobs.** Running step 5 with `ORDERS="AA AB BA BB"` in a single job would confirm it, and would help separate the cache effect from the gap above.
- **Same-prompt control.** Step 5 asked for an "AA" control. Request 1 here is such a control (A after A), though from a different job.

## Run notes

- Job 1654577 ran on watgpu802 (4 CPUs) and finished on 2026-10-29 at 13:05 UTC.
- The job script excludes watgpu108 and watgpu208, which have both had CUDA faults (steps 1–2 and 5).
- On watgpu.cs, home (`/u801`) is mounted `noexec` on the login node, so the Miniforge install in `setup.sh` fails there with "Permission denied". Run `setup.sh` from an `salloc` session on a compute node.
- Results are in [`results/within-1654577/`](results/within-1654577/) and [`results/across-1654577/`](results/across-1654577/): `summary.txt`, `latency.csv`, `requests.jsonl`, and one layer CSV per process. `stdout.log` and the `slurm-*.out` file are git-ignored.
