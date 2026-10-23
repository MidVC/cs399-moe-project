#!/bin/bash
# Same environment as moe-setup/run.sh, but runs smoke_logged.py and saves logs.
# Put this folder inside your moe-setup checkout (next to run.sh) and run it
# inside your salloc session:  bash moe-cache-logging/run_logged.sh
#
# Knobs (env vars):
#   CPP_DEBUG=1      turn on MoE-Infinity's C++ DLOG_DEBUG lines (per-expert HIT/MISS/evict)
#   N_REPEATS=3      how many times to send the same prompt in one process
#   SYNC_LAYERS=0    1 = cuda-synchronize around each MoE layer (accurate per-layer ms,
#                    but slows the whole request; keep 0 for latency measurements)
#   RUN_TAG=...      name prefix for the log files
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"
export MOE_WORK_ROOT="$(cat "$REPO_DIR/.work-root")"

source "$MOE_WORK_ROOT/miniforge/etc/profile.d/conda.sh"
conda activate "$MOE_WORK_ROOT/env"

mkdir -p "$MOE_WORK_ROOT"/{hf,cache,tmp,state/torch_extensions,state/triton}
export PYTHONNOUSERSITE=1
export HF_HOME="$MOE_WORK_ROOT/hf"
export XDG_CACHE_HOME="$MOE_WORK_ROOT/cache"
export TMPDIR="$MOE_WORK_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$MOE_WORK_ROOT/state/torch_extensions"
export TRITON_CACHE_DIR="$MOE_WORK_ROOT/state/triton"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
export LOG_DIR="${LOG_DIR:-$HERE/logs}"
mkdir -p "$LOG_DIR"

if [ "${CPP_DEBUG:-1}" = "1" ]; then
    export MUDUO_LOG_DEBUG=1          # read once by core/base/logging.cc:initLogLevel()
fi

cd "$REPO_DIR"
# stdbuf so C++ printf/fwrite(stdout) lines are not held in a 4 KB buffer
flock -n "$MOE_WORK_ROOT/state/run.lock" \
    stdbuf -oL -eL python -u "$HERE/smoke_logged.py" 2>&1 | tee "$LOG_DIR/${RUN_TAG}_stdout.log"

if [ "${CPP_DEBUG:-1}" = "1" ]; then
    python "$HERE/parse_debug_log.py" \
        "$LOG_DIR/${RUN_TAG}_stdout.log" "$LOG_DIR/${RUN_TAG}_expert_events.csv"
fi
