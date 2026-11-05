#!/bin/bash
# Prefill-residency driver. Runs residency.py for each sequence, N_TRIALS times,
# each in a fresh process (cold cache). Run inside a GPU salloc session:
#   bash prefill-residency/run_residency.sh
#
# Knobs (env vars):
#   N_TRIALS=5     trials per sequence
#   CPP_DEBUG=0    1 = also capture MoE-Infinity's per-expert HIT/MISS/evict log
#                  (MUDUO_LOG_DEBUG) and parse it to <run>_expert_events.csv. Big + slow.
#   RUN_TAG=...    results-folder name (default: timestamp)
#   OUT_DIR=...    default prefill-residency/results/$RUN_TAG
#   SEQUENCES="label=spec ..."  spec is pid:ntokens,...; first step = victim,
#                  middle = attacker, last (ntok 1) = probe.
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

N_TRIALS="${N_TRIALS:-5}"
CPP_DEBUG="${CPP_DEBUG:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HERE/results/$RUN_TAG}"
mkdir -p "$OUT_DIR"

# label=spec ; victim=first step, attacker=middle, probe=last (ntok 1)
SEQUENCES="${SEQUENCES:-AvA=A:16,A:1 AvBp=A:16,B:1,A:1 AvBf=A:16,B:16,A:1 BvB=B:16,B:1 BvAp=B:16,A:1,B:1 BvAf=B:16,A:16,B:1}"

[ "$CPP_DEBUG" = "1" ] && export MUDUO_LOG_DEBUG=1

cd "$REPO_DIR"
set +e   # a single process crashing (e.g. a flaky CUDA segfault) must not kill the run
run_one() {  # label spec trial
    local label="$1" spec="$2" trial="$3"
    local stem="$OUT_DIR/${label}_trial$(printf '%02d' "$trial")"
    if [ "$CPP_DEBUG" = "1" ]; then
        stdbuf -oL -eL python -u "$HERE/residency.py" \
            --sequence "$spec" --label "$label" --trial "$trial" --out-dir "$OUT_DIR" \
            > "${stem}_stdout.log" 2>&1
        python "$REPO_DIR/moe-cache-logging/parse_debug_log.py" \
            "${stem}_stdout.log" "${stem}_expert_events.csv" || true
    else
        python -u "$HERE/residency.py" \
            --sequence "$spec" --label "$label" --trial "$trial" --out-dir "$OUT_DIR"
    fi
}

(
flock -n 9 || { echo "another residency run holds the lock"; exit 75; }
for trial in $(seq 1 "$N_TRIALS"); do
    for pair in $SEQUENCES; do
        label="${pair%%=*}"; spec="${pair#*=}"
        echo "=== trial $trial $label ($spec) $(date +%H:%M:%S)"
        run_one "$label" "$spec" "$trial" || echo "=== FAILED $label trial $trial (continuing)"
    done
    python3 "$HERE/summarize_residency.py" "$OUT_DIR" > /dev/null || true
done
) 9> "$MOE_WORK_ROOT/state/run.lock" 2>&1 | tee -a "$OUT_DIR/driver.log"

python3 "$HERE/summarize_residency.py" "$OUT_DIR" || true
echo "RESULTS: $OUT_DIR"
