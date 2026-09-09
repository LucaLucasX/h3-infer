#!/usr/bin/env bash
# Create / refresh .venv_sage_bench with PyTorch cu130 + SageAttention 2.2 + SageAttn3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv_sage_bench"
SAGE_SRC="${SAGEATTN_SRC:-/tmp/SageAttention}"

if [[ ! -d "$SAGE_SRC" ]]; then
  git clone --depth 1 https://github.com/thu-ml/SageAttention.git "$SAGE_SRC"
fi

python3.11 -m venv "$VENV"
"$VENV/bin/pip" install -U pip setuptools wheel
"$VENV/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install -r "$ROOT/ComfyUI-master_cp/requirements.txt"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

"$VENV/bin/pip" install "$SAGE_SRC" --no-build-isolation
"$VENV/bin/pip" install "$SAGE_SRC/sageattention3_blackwell" --no-build-isolation

"$VENV/bin/python" -c "
import torch
from sageattention.core import get_cuda_arch_versions
from sageattn3 import sageattn3_blackwell
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('sage archs', get_cuda_arch_versions())
print('sageattn3 ok')
"

echo "Done: $VENV"
