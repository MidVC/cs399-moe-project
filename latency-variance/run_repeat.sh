#!/bin/bash
# Step 4 driver: runs repeat_latency.py N_PROCS times, each a new process that
# sends the same prompt N_REQUESTS times. Every process starts from an empty GPU
# expert cache. Run it inside your salloc session:
#   N_PROCS=1 N_REQUESTS=50 RUN_TAG=within  bash latency-variance/run_repeat.sh
#   N_PROCS=5 N_REQUESTS=20 RUN_TAG=across  bash latency-variance/run_repeat.sh
#
# Knobs (env vars):
#   N_PROCS=5        number of processes
#   N_REQUESTS=20    requests per process
#   RUN_TAG=...      name of the results folder (default: timestamp)
#   OUT_DIR=...      default latency-variance/results/$RUN_TAG
# Extra arguments go to repeat_latency.py, e.g. --prompt "...".
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"

N_PROCS="${N_PROCS:-5}"
N_REQUESTS="${N_REQUESTS:-20}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HERE/results/$RUN_TAG}"
mkdir -p "$OUT_DIR"

{
    echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} cpus ${SLURM_CPUS_PER_TASK:-?}"
    # Anything else on the GPU would add noise; record it.
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
    for proc in $(seq 1 "$N_PROCS"); do
        echo "=== proc $proc $(date +%H:%M:%S)"
        bash "$REPO_DIR/run.sh" latency-variance/repeat_latency.py \
            --n-requests "$N_REQUESTS" --proc "$proc" --out-dir "$OUT_DIR" "$@"
        # Refresh summary.txt after every process, so a run cut off by the time limit is still summarized.
        python3 "$HERE/summarize.py" "$OUT_DIR" > /dev/null
    done
} 2>&1 | tee -a "$OUT_DIR/stdout.log"

cat "$OUT_DIR/summary.txt"
echo "RESULTS: $OUT_DIR"
