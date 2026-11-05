"""
Prefill-residency probe: after an attacker prompt's prefill runs, how many of a
victim prompt's experts are still resident in the GPU expert cache?

All steps run in one process; the GPU expert cache starts empty.
  1. Victim V runs a full generation (loads V's experts).
  2. (optional) Attacker T runs -- prefill only (1 new token) or full (16).
  3. V' re-runs V's PREFILL (1 new token). Each expert V' re-touches is a cache
     HIT if still resident, a MISS if it was evicted. V and V' route identically
     in prefill (greedy, deterministic), so V' probes exactly V's prefill experts.

Comparing a control (V, V') with a target (V, T, V'): the extra V' prefill misses
are the victim experts the attacker evicted. If the engine exposes a way to list
the resident experts, we also snapshot the set after each step (no probe needed).

Reuses fixed_prompt.load_model/run_prompt (step 3) and ExpertCacheLogger (steps 1-2).
Outputs into --out-dir:
  requests.jsonl                     one line per request
  <label>_trial<NN>_layers.csv       per (request, fwd_step, layer) [ExpertCacheLogger]
  <label>_trial<NN>_resident.json    resident-set snapshots, if the engine exposes them
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "moe-cache-logging"))
from fixed_prompt import DEFAULT_PROMPT, GEN_SETTINGS, load_model, run_prompt  # noqa: E402
from expert_cache_logger import ExpertCacheLogger  # noqa: E402

DEFAULT_PROMPT_B = "Write a haiku about autumn leaves."

# Methods that might enumerate the resident experts (names are guesses; we probe at
# runtime and fall back to the reload probe if none work).
RESIDENT_SET_CANDIDATES = [
    "get_resident_experts", "get_resident_expert_keys", "get_resident_expert_ids",
    "resident_experts", "get_cached_experts", "cached_experts",
    "get_cached_expert_keys", "list_resident_experts", "get_resident_expert_list",
]


def find_resident_set_fn(engine):
    """Return a no-arg callable listing resident experts, or None. Prints the API."""
    objs = {"engine": engine, "archer": getattr(engine, "archer_engine", None),
            "dispatcher": getattr(engine, "expert_dispatcher", None)}
    for name, obj in objs.items():
        if obj is None:
            continue
        meths = [m for m in dir(obj) if not m.startswith("__")
                 and any(k in m.lower() for k in ("resid", "cache", "expert", "evict", "occup"))]
        print(f"[api] {name}: {meths}", flush=True)
    for name, obj in objs.items():
        if obj is None:
            continue
        for cand in RESIDENT_SET_CANDIDATES:
            fn = getattr(obj, cand, None)
            if not callable(fn):
                continue
            try:
                keys = list(fn())
            except Exception as e:
                print(f"[api] {name}.{cand}() failed: {e}", flush=True)
                continue
            print(f"[api] {name}.{cand}() -> {len(keys)} items, sample {keys[:3]}", flush=True)
            if 100 < len(keys) < 7000:
                return fn
    return None


def snapshot_resident(fn):
    """Normalise a resident-set listing to a sorted list of ints or [layer, expert] lists."""
    out = []
    for k in fn():
        if isinstance(k, (list, tuple)):
            out.append([int(x) for x in k])
        else:
            out.append(int(k))
    out.sort(key=lambda x: (x if isinstance(x, int) else tuple(x)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True, help='steps as pid:ntokens, e.g. "A:16,B:1,A:1"')
    ap.add_argument("--label", required=True, help="name for this sequence (folder-safe)")
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-a", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-b", default=DEFAULT_PROMPT_B)
    args = ap.parse_args()

    steps = []
    for tok in args.sequence.split(","):
        pid, n = tok.split(":")
        steps.append((pid, int(n)))
    prompts = {"A": args.prompt_a, "B": args.prompt_b}
    assert all(pid in prompts for pid, _ in steps), args.sequence

    model, tokenizer = load_model()
    assert torch.cuda.device_count() == 1, "Expected one visible GPU"
    gpu = torch.cuda.get_device_name(0)
    assert "RTX 6000 Ada" in gpu, gpu
    assert getattr(model.engine, "dtype_cls", None) == torch.bfloat16

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = f"{args.label}_trial{args.trial:02d}"
    logger = ExpertCacheLogger(model, str(out_dir / f"{run}_layers.csv"), sync_layers=False)
    resident_fn = find_resident_set_fn(model.engine)

    resident_snaps = {}
    for pos, (pid, ntok) in enumerate(steps, start=1):
        gen = dict(GEN_SETTINGS)
        gen["max_new_tokens"] = ntok
        tag = f"{pos}{pid}n{ntok}"
        role = "victim" if pos == 1 else ("probe" if (pos == len(steps) and ntok == 1) else "attacker")
        # Marker so a per-expert debug log (MUDUO_LOG_DEBUG=1) can be split by request.
        print(f"###REQ run={run} pos={pos} pid={pid} ntok={ntok} role={role}###", flush=True)
        before = logger.snapshot()
        with logger.request(tag, run=run):
            record = run_prompt(model, tokenizer, prompts[pid], gen_settings=gen)
        after = logger.snapshot()
        if resident_fn is not None:
            try:
                resident_snaps[tag] = snapshot_resident(resident_fn)
            except Exception as e:
                print(f"[api] resident snapshot failed: {e}", flush=True)
        acc = after["cum_accesses"] - before["cum_accesses"]
        hits = (round(after["cum_hit_rate"] * after["cum_accesses"])
                - round(before["cum_hit_rate"] * before["cum_accesses"]))
        record.update(
            label=args.label, trial=args.trial, sequence=args.sequence, pos=pos,
            pid=pid, ntok=ntok, role=role, request_tag=tag, gpu=gpu,
            accesses=acc, hits=hits, misses=acc - hits,
            resident_before=before["resident_experts"], resident_after=after["resident_experts"],
        )
        with (out_dir / "requests.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{run} {tag} {role}] {record['latency_s']:.3f}s out={record['output_tokens']} "
              f"hits={hits}/{acc} resident {record['resident_before']}->{record['resident_after']}",
              flush=True)

    logger.close()
    if resident_snaps:
        (out_dir / f"{run}_resident.json").write_text(json.dumps(resident_snaps))
    print("RESIDENCY_OK", run, flush=True)


if __name__ == "__main__":
    main()
