#!/bin/bash
# SSE decode-timing driver. Starts the SSE server, waits until it is ready, then
# runs the client for prompt A (cold cache) and prompt B (warm -- sees what A
# left resident), analyzing each. Run inside a GPU salloc session:
#   bash sse-timing/run_sse_timing.sh
#
# Knobs (env vars):
#   MAX_NEW=96            tokens generated per prompt
#   PROMPT_A / PROMPT_B   the victim / attacker prompts
#   PORT=8077             server port
#   RUN_TAG=...           results-folder name (default: timestamp)
#   OUT_DIR=...           default sse-timing/results/$RUN_TAG
set -eo pipefail

# ---- locate the repo root and load the shared environment (as the siblings do) ----
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

# ---- knobs ----
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HERE/results/$RUN_TAG}"
export LOG_DIR="$OUT_DIR"            # sse_server.py writes its CSVs here
mkdir -p "$OUT_DIR"
export PORT="${PORT:-8077}"
MAX_NEW="${MAX_NEW:-96}"
PROMPT_A="${PROMPT_A:-Explain how a nuclear reactor produces electricity, step by step.}"
PROMPT_B="${PROMPT_B:-Write a short Python function that reverses a linked list.}"

# ---- start the server; model load takes a few minutes ----
echo "[sse-timing] starting server (tag $RUN_TAG) ..."
stdbuf -oL -eL python -u "$HERE/sse_server.py" > "$OUT_DIR/${RUN_TAG}_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT          # always stop the server on exit

# ---- wait (up to ~6 min) for the server to answer /health ----
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 2
    if ! kill -0 $SRV 2>/dev/null; then echo "server died:"; cat "$OUT_DIR/${RUN_TAG}_server.log"; exit 1; fi
done
echo "[sse-timing] server up."

# ---- one request: client fetches + streams, then analyze ----
run_one () {  # args: tag  prompt  cache
    echo "[sse-timing] client $1 (cache=$3)"
    python "$HERE/sse_client.py" --host 127.0.0.1 --port "$PORT" \
        --tag "$1" --prompt "$2" --max-new-tokens "$MAX_NEW" --cache "$3" --logdir "$OUT_DIR"
    python "$HERE/analyze.py" --tag "$1" --layers-tag "$RUN_TAG" --logdir "$OUT_DIR"
}

run_one "A" "$PROMPT_A" "cold"
run_one "B" "$PROMPT_B" "warm"   # B sees whatever experts A left on the GPU
echo "[sse-timing] done. results in $OUT_DIR (tag $RUN_TAG; per-request A_*, B_*)"
