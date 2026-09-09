#!/usr/bin/env bash
# Build SpargeAttn in-place under vendor/ (no pip install into .venv_sage_bench).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/sparse_attn"
VENDOR="$EXP/vendor/SpargeAttn"
VENV="${VENV:-$ROOT/.venv_sage_bench}"

if [[ ! -d "$VENDOR" ]]; then
  echo "missing $VENDOR — clone first" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$VENDOR"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"$VENV/bin/pip" install -q ninja packaging wheel setuptools
export EXT_PARALLEL="${EXT_PARALLEL:-4}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 8}"
export MAX_JOBS="${MAX_JOBS:-4}"
# 5090 only — avoid compiling unused sm80/89/90 matrices for every arch.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
# Clean previous failed objects
rm -rf build
# In-place build so .so lives under vendor/; PYTHONPATH injects it for :8190 only.
"$VENV/bin/python" setup.py build_ext --inplace
"$VENV/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import spas_sage_attn
print("spas_sage_attn OK from", spas_sage_attn.__file__)
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
print("API spas_sage2_attn_meansim_topk_cuda OK")
PY
echo "build done. Use run_comfy_exp.sh (sets PYTHONPATH)."
