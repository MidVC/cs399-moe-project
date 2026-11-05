"""
Step 4: send the same prompt N times in one process and record each request's
latency and output-token count.

Reuses step 3 (fixed_prompt.py: load_model, run_prompt, GEN_SETTINGS) for the
model, prompt, generation settings and timing, and steps 1-2 (ExpertCacheLogger)
for the cache counters used to explain any variance. The GPU expert cache starts
empty in every process, so request 0 is cold and requests 1..N-1 are warm.

Outputs (in --out-dir):
  requests.jsonl          one line per request, appended across processes:
                          the step 3 record + proc, req, node, cache counters
  proc<NN>_layers.csv     one row per (request, forward step, MoE layer)
"""
import argparse
import json
import os
import socket
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "moe-cache-logging"))
from fixed_prompt import DEFAULT_PROMPT, load_model, run_prompt  # noqa: E402
from expert_cache_logger import ExpertCacheLogger  # noqa: E402


def cum_hits(snap):
    return round(snap["cum_hit_rate"] * snap["cum_accesses"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-requests", type=int, required=True)
    parser.add_argument("--proc", type=int, required=True, help="process index (1, 2, ...)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    model, tokenizer = load_model()
    assert torch.cuda.device_count() == 1, "Expected one visible GPU"
    gpu = torch.cuda.get_device_name(0)
    assert "RTX 6000 Ada" in gpu, gpu
    assert getattr(model.engine, "dtype_cls", None) == torch.bfloat16

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = f"proc{args.proc:02d}"
    logger = ExpertCacheLogger(model, str(out_dir / f"{run}_layers.csv"),
                               sync_layers=False)
    env = dict(
        node=socket.gethostname(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
        gpu=gpu,
    )

    for req in range(args.n_requests):
        before = logger.snapshot()
        with logger.request(f"r{req:03d}", run=run):
            record = run_prompt(model, tokenizer, args.prompt)
        after = logger.snapshot()

        accesses = after["cum_accesses"] - before["cum_accesses"]
        hits = cum_hits(after) - cum_hits(before)
        record.update(
            proc=args.proc, req=req, **env,
            h2d_bytes=after["h2d_bytes_total"] - before["h2d_bytes_total"],
            accesses=accesses, hits=hits, misses=accesses - hits,
            resident_before=before["resident_experts"],
            resident_after=after["resident_experts"],
        )
        with (out_dir / "requests.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{run} r{req:03d}] {record['latency_s']:.3f}s "
              f"in={record['input_tokens']} out={record['output_tokens']} "
              f"h2d={record['h2d_bytes'] / 2**30:.2f}GiB hits={hits}/{accesses} "
              f"resident {record['resident_before']}->{record['resident_after']} "
              f"| {record['generated_text']!r}", flush=True)

    logger.close()
    print("REPEAT_LATENCY_OK", run, flush=True)


if __name__ == "__main__":
    main()
