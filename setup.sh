#!/bin/bash
set -eo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOE_WORK_ROOT="${MOE_WORK_ROOT:-$HOME/moe-work}"

mkdir -p "$MOE_WORK_ROOT"/{models,state,tmp,hf,cache}
printf '%s\n' "$MOE_WORK_ROOT" > "$REPO_DIR/.work-root"

if [ ! -x "$MOE_WORK_ROOT/miniforge/bin/conda" ]; then
    curl -fL --retry 3 \
        -o "$MOE_WORK_ROOT/miniforge.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

    bash "$MOE_WORK_ROOT/miniforge.sh" \
        -b \
        -p "$MOE_WORK_ROOT/miniforge"
fi

source "$MOE_WORK_ROOT/miniforge/etc/profile.d/conda.sh"

ENV_DIR="$MOE_WORK_ROOT/env"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    if [ -f "$REPO_DIR/conda-linux-64.lock" ]; then
        conda create -y --copy \
            --prefix "$ENV_DIR" \
            --file "$REPO_DIR/conda-linux-64.lock"
    else
        conda create -y --copy \
            --prefix "$ENV_DIR" \
            --override-channels \
            -c conda-forge \
            python=3.12 pip "libstdcxx-ng>=12" libuuid
    fi
fi

conda activate "$ENV_DIR"

export PYTHONNOUSERSITE=1
export HF_HOME="$MOE_WORK_ROOT/hf"
export XDG_CACHE_HOME="$MOE_WORK_ROOT/cache"
export TMPDIR="$MOE_WORK_ROOT/tmp"

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade
fi

if [ -f "$REPO_DIR/requirements.lock" ]; then
    python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --requirement "$REPO_DIR/requirements.lock"
else
    python -m pip install --upgrade pip

    python -m pip install \
        "torch==2.11.0" \
        --index-url https://download.pytorch.org/whl/cu128

    cat > "$MOE_WORK_ROOT/constraints.txt" <<'TXT'
torch==2.11.0+cu128
transformers==5.3.0
sglang-kernel==0.4.6
TXT

    python -m pip install \
        --constraint "$MOE_WORK_ROOT/constraints.txt" \
        "https://github.com/EfficientMoE/moe-store/releases/download/v0.2.2/moe_store-0.2.2-py3-none-any.whl" \
        "https://github.com/EfficientMoE/MoE-Infinity/releases/download/v0.2.2/moe_infinity-0.2.2-cp312-cp312-manylinux1_x86_64.whl"
fi

python -m pip check

python - <<'PY'
import torch
from moe_infinity import MoE
import moe_infinity._store
import moe_infinity._engine

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("MOE_IMPORT_OK")
PY

python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = (
    Path(os.environ["MOE_WORK_ROOT"])
    / "models"
    / "Qwen3-30B-A3B-ad44e777"
)

snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B",
    revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    local_dir=str(model_dir),
    allow_patterns=[
        "*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"
    ],
    max_workers=4,
)

config = json.loads((model_dir / "config.json").read_text())
dtype = config.get("dtype", config.get("torch_dtype"))
assert dtype == "bfloat16", dtype

print("MODEL_DOWNLOAD_OK")
PY

echo "SETUP_OK"
