#!/bin/bash
# OPTIONAL. Rebuilds MoE-Infinity v0.2.2 with expert_cache_events.patch and swaps
# ONLY the patched _store*.so into your env (original is backed up).
# Run on a GPU node inside salloc (needs nvcc 12.8 to match torch 2.11+cu128).
#
#   bash moe-cache-logging/cpp_patch/build_patched_store.sh            # build + install
#   bash moe-cache-logging/cpp_patch/build_patched_store.sh --restore  # put original back
#
# After installing, enable the log per run with:
#   MOE_EXPERT_EVENT_LOG=$PWD/logs/events.csv bash moe-cache-logging/run_logged.sh
set -eo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
export MOE_WORK_ROOT="$(cat "$REPO_DIR/.work-root")"
source "$MOE_WORK_ROOT/miniforge/etc/profile.d/conda.sh"
conda activate "$MOE_WORK_ROOT/env"

SITE_PKG="$(python -c 'import moe_infinity,os;print(os.path.dirname(moe_infinity.__file__))')"
SO_NAME="$(cd "$SITE_PKG" && ls _store*.so)"
BACKUP="$SITE_PKG/$SO_NAME.orig"

if [ "$1" = "--restore" ]; then
    cp -v "$BACKUP" "$SITE_PKG/$SO_NAME"; exit 0
fi

# nvcc: load your cluster's CUDA 12.8 module if nvcc is not already on PATH,
# e.g.  module load cuda/12.8   (name differs per cluster)
command -v nvcc >/dev/null || { echo "nvcc not found: module load cuda/12.8 first"; exit 1; }
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
nvcc --version | tail -2

SRC="$MOE_WORK_ROOT/src"
mkdir -p "$SRC"
[ -d "$SRC/MoE-Infinity" ] || git clone https://github.com/EfficientMoE/MoE-Infinity.git "$SRC/MoE-Infinity"
[ -d "$SRC/cutlass" ] || git clone --depth 1 https://github.com/NVIDIA/cutlass.git "$SRC/cutlass"

cd "$SRC/MoE-Infinity"
git checkout -q v0.2.2
git checkout -q -- core/parallel/expert_dispatcher.cpp
git apply "$HERE/expert_cache_events.patch"
echo "patch applied"

python -m pip install -q "setuptools>=78.1.1,<82" "setuptools-scm>=8" wheel ninja py-cpuinfo

# RTX 6000 Ada = sm_89; the default sm_80 cubin runs on it. Skip sm_90 to save time.
export CUTLASS_DIR="$SRC/cutlass"
export MOE_ENABLE_SM90=0
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$MOE_WORK_ROOT/tmp"
python setup.py build_ext --inplace 2>&1 | tail -20

NEW_SO="$(ls moe_infinity/_store*.so)"
[ -f "$BACKUP" ] || cp -v "$SITE_PKG/$SO_NAME" "$BACKUP"
cp -v "$NEW_SO" "$SITE_PKG/$SO_NAME"

python - <<'PY'
import moe_infinity._store, moe_infinity._engine
print("PATCHED_STORE_IMPORT_OK")
PY
strings "$SITE_PKG/$SO_NAME" | grep -q MOE_EXPERT_EVENT_LOG && echo "PATCH_PRESENT"
