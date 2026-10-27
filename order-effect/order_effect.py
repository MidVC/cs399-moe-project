"""
Step 5: run two fixed prompts in a given order (e.g. AB or BA) in one process.

Reuses step 3 (fixed_prompt.py: load_model, run_prompt, GEN_SETTINGS) for the
model, prompts and timing, and steps 1-2 (ExpertCacheLogger) for the cache logs.
The GPU expert cache starts empty in every process, so the first prompt is a
cold-cache request and the second one runs on the cache the first one left.

Outputs (in --out-dir):
  requests.jsonl                   one line per request, appended across trials:
                                   the step 3 record + trial, order, position,
                                   cache counters for that request
  trial<NN>_<order>_layers.csv     one row per (request, forward step, MoE layer)
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "moe-cache-logging"))
from fixed_prompt import DEFAULT_PROMPT, load_model, run_prompt  # noqa: E402
from expert_cache_logger import ExpertCacheLogger  # noqa: E402

DEFAULT_PROMPT_B = "Write a haiku about autumn leaves."


def cum_hits(snap):
    return round(snap["cum_hit_rate"] * snap["cum_accesses"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", required=True, help="e.g. AB or BA")
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt-a", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-b", default=DEFAULT_PROMPT_B)
    args = parser.parse_args()

    prompts = {"A": args.prompt_a, "B": args.prompt_b}
    assert args.order and set(args.order) <= set(prompts), args.order

    model, tokenizer = load_model()
    assert torch.cuda.device_count() == 1, "Expected one visible GPU"
    gpu = torch.cuda.get_device_name(0)
    assert "RTX 6000 Ada" in gpu, gpu
    assert getattr(model.engine, "dtype_cls", None) == torch.bfloat16

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = f"trial{args.trial:02d}_{args.order}"
    logger = ExpertCacheLogger(model, str(out_dir / f"{run}_layers.csv"),
                               sync_layers=False)

    prev = None
    for position, pid in enumerate(args.order, start=1):
        before = logger.snapshot()
        with logger.request(f"{position}{pid}", run=run):
            record = run_prompt(model, tokenizer, prompts[pid])
        after = logger.snapshot()

        accesses = after["cum_accesses"] - before["cum_accesses"]
        hits = cum_hits(after) - cum_hits(before)
        record.update(
            trial=args.trial, order=args.order, position=position,
            prompt_id=pid, prev_prompt_id=prev, gpu=gpu,
            h2d_bytes=after["h2d_bytes_total"] - before["h2d_bytes_total"],
            accesses=accesses, hits=hits, misses=accesses - hits,
            resident_before=before["resident_experts"],
            resident_after=after["resident_experts"],
        )
        with (out_dir / "requests.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{run} pos{position} {pid}] {record['latency_s']:.3f}s "
              f"in={record['input_tokens']} out={record['output_tokens']} "
              f"h2d={record['h2d_bytes'] / 2**30:.2f}GiB hits={hits}/{accesses} "
              f"resident {record['resident_before']}->{record['resident_after']} "
              f"| {record['generated_text']!r}", flush=True)
        prev = pid

    logger.close()
    print("ORDER_EFFECT_OK", run, flush=True)


if __name__ == "__main__":
    main()
