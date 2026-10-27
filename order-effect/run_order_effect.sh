#!/bin/bash
# Step 5 driver: runs order_effect.py N_TRIALS times per order, alternating the
# orders (AB, BA, AB, BA, ...). Every run is a new process, so every pair starts
# from an empty GPU expert cache. Run it inside your salloc session:
#   bash order-effect/run_order_effect.sh
#
# Knobs (env vars):
#   N_TRIALS=10      trials per order
#   ORDERS="AB BA"   orders to alternate; add "AA BB" for same-prompt controls
#   RUN_TAG=...      name of the results folder (default: timestamp)
#   OUT_DIR=...      default order-effect/results/$RUN_TAG
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"

N_TRIALS="${N_TRIALS:-10}"
ORDERS="${ORDERS:-AB BA}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$HERE/results/$RUN_TAG}"
mkdir -p "$OUT_DIR"

for trial in $(seq 1 "$N_TRIALS"); do
    for order in $ORDERS; do
        echo "=== trial $trial order $order $(date +%H:%M:%S)"
        bash "$REPO_DIR/run.sh" order-effect/order_effect.py \
            --order "$order" --trial "$trial" --out-dir "$OUT_DIR" "$@"
    done
    # Refresh summary.txt every round, so a run cut off by the time limit is still summarized.
    python3 "$HERE/summarize.py" "$OUT_DIR" > /dev/null
done 2>&1 | tee -a "$OUT_DIR/stdout.log"

cat "$OUT_DIR/summary.txt"
echo "RESULTS: $OUT_DIR"
