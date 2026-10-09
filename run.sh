#!/bin/bash
set -eo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

cd "$REPO_DIR"

flock -n "$MOE_WORK_ROOT/state/run.lock" \
    python -u "$REPO_DIR/smoke.py"
