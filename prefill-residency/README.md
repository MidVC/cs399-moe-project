# Prefill residency: do the victim's experts survive the attacker's prefill?

Follow-on to the step 5 order-effect work. Question: when prompt A (victim) has run
and prompt B (attacker) then runs its **prefill**, are A's experts still resident in
the GPU expert cache? Reuses `fixed_prompt.load_model/run_prompt` (step 3) and the
`ExpertCacheLogger` (steps 1–2).

## Method

Everything runs in one process; the GPU expert cache starts empty.

1. **Victim** runs a full generation (16 tokens) — loads its experts.
2. **Attacker** (optional) runs its **prefill only** (`max_new_tokens=1`), or full (16).
3. **Probe**: the victim's prefill is re-run (`max_new_tokens=1`). Routing is greedy
   and deterministic, so the probe re-touches exactly the victim's prefill experts.
   Each is a cache **hit** if still resident, a **miss** if evicted.

A no-attacker control (victim → probe) runs alongside each attacker case, so the
*extra* probe misses isolate the attacker. Two ways to read survival:

- **Attacker segment (clean, direct):** in the attacker's prefill log, each access is
  a HIT (resident from the victim) or MISS. Attacker HITs = victim experts still
  resident when the attacker's prefill reaches them. No probe needed.
- **Probe (confounded, see caveats):** the probe re-runs the victim's prefill and
  counts hits. With the per-expert debug log on, survivors split into experts the
  attacker also used (shared) vs victim-unique.

### Code

| File | What |
|---|---|
| `residency.py` | Runs a sequence (`A:16,B:1,A:1`), tags each request, logs per layer. Also probes the engine for a resident-set listing method (none exists in 0.2.2) and, with `MUDUO_LOG_DEBUG=1`, records per-expert identities. |
| `run_residency.sh` | Runs each sequence `N_TRIALS` times, each a fresh process. `CPP_DEBUG=1` captures and parses the per-expert debug log. |
| `summarize_residency.py` | Builds `summary.txt`: probe survival, attacker's marginal effect, and the shared-vs-unique split from the debug log. |

Run inside a GPU `salloc` on an RTX 6000 Ada node, from the repo root:

```bash
bash run.sh                                           # smoke test
RUN_TAG=run1 N_TRIALS=5 bash prefill-residency/run_residency.sh
RUN_TAG=run1-dbg N_TRIALS=2 CPP_DEBUG=1 \
  SEQUENCES="AvA=A:16,A:1 AvBp=A:16,B:1,A:1 BvB=B:16,B:1 BvAp=B:16,A:1,B:1" \
  bash prefill-residency/run_residency.sh
python3 prefill-residency/summarize_residency.py prefill-residency/results/run1-dbg
```

## Results

Run `job1659263-dbg`, watgpu108, 2026-10-30, 2 trials each. Sequences: `AvA`
(A victim, no attacker), `AvBp` (A victim, B prefill attacker), and the B-victim
mirrors `BvB`, `BvAp`. The victim is A = "Explain what a GPU is in one sentence.",
B = "Write a haiku about autumn leaves." The full results are in
[`results/job1659263-dbg/summary.txt`](results/job1659263-dbg/summary.txt).

### 1. Yes — a large fraction of the victim's experts survive into the attacker's prefill

From the attacker's own prefill log (the clean measurement):

| Attacker prefill | Accesses | Hit (already resident) | Of those, victim's |
|---|---|---|---|
| B, after victim A | 2,674 | ~1,509 (56%) | all ~1,509 |
| A, after victim B | 2,794 | ~1,674 (60%) | all ~1,674 |

So ~1,500 of the victim's experts are still in GPU memory when the attacker's prefill
reaches them. (This matches the ~1,488 estimated from the step 5 logs.) Because the
attacker only ran after the victim, every attacker-prefill hit is a victim expert — so
**the attacker's hit/miss pattern is directly shaped by which experts the victim
loaded.** That is the premise of the MoEcho cache channel, reproduced here.

### 2. The survivors are mostly the experts the two prompts share

Of the victim's experts, re-probed after the attacker's prefill:

| Victim expert type | Survived (probe hit) |
|---|---|
| Shared with the attacker | ~63% (1477/2324, 1423/2324 for A) |
| Victim-unique | ~42% (432/1038, 422/1038 for A) |

The cache keeps experts that get re-touched; shared experts are re-touched by the
attacker, victim-unique ones are not.

### 3. But the attacker's prefill does not evict much *more* than the victim already does

Probe survival with vs without the attacker is within noise:

| Victim | Control (no attacker) | After attacker prefill | Marginal |
|---|---|---|---|
| A | 1844 / 2794 (66.0%) | 1877 / 2794 (67.2%) | ~0 (−1.2%) |
| B | 1954 / 2674 (73.1%) | 1924 / 2674 (72.0%) | ~0 (+1.1%) |

Two reasons: the prompts share ~80% of their prefill experts, so the attacker's
prefill mostly *reloads* shared experts rather than displacing new victim ones; and
the victim's prefill working set (2,794 experts) already exceeds the cache (~2,200),
so its experts are churned by capacity regardless of the attacker.

## Caveats

- **The probe is confounded.** Because the victim's prefill set (2,794) is larger than
  the cache (~2,200), re-running it churns the cache on its own, which swamps the
  attacker's marginal effect. Lean on the **attacker-segment hit count (~1,500)**,
  which needs no probe, for the headline answer; treat the probe survival % and the
  marginal-effect number as indicative only.
- **No resident-set snapshot.** MoE-Infinity 0.2.2 exposes a resident-expert *count*
  but no way to *list* the resident set (`archer`: `get_resident_expert_count` only;
  `take_last_active_experts` / `replace_cache_candidates` / `get_expert_policy_stats`
  exist but were not used). A listing method would give a clean, probe-free snapshot.
- **Two trials, not more.** The planned count run (more trials + a full-attacker
  variant) died on a flaky CUDA segfault mid-run; the driver now continues past a
  crashed process. Routing is deterministic and the two trials agree to within ~50
  experts, so the numbers are stable, but the full-attacker comparison is still open.
- Absolute latencies and cache budget vary by node; see the step 5 notes.

## What's committed

The per-layer CSVs, per-request `requests.jsonl`, and `summary.txt` for the debug run.
The raw per-expert debug logs (`*_stdout.log`, ~3 MB; `*_expert_events.csv`) and the
partial, segfaulted count run stay on the cluster under
`~/cs399-moe-project/prefill-residency/results/`.
