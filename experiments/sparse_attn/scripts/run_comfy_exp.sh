#!/usr/bin/env bash
# Experimental ComfyUI on :8190. Does not touch :8188.
# PYTHONPATH injects vendor SpargeAttn; symlink registers EXP nodes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/sparse_attn"
CP="$ROOT/ComfyUI-master_cp"
PORT="${H3_SPARGE_PORT:-8190}"
GPU="${H3_SPARGE_GPU:-1}"
VENV="${VENV:-$ROOT/.venv_sage_bench}"
VENDOR="$EXP/vendor/SpargeAttn"
NODE_LINK="$CP/custom_nodes/H3-SpargeAttn-Exp"
NODE_SRC="$EXP/custom_nodes/H3-SpargeAttn-Exp"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped ComfyUI :${PORT}"
  exit 0
fi

if [[ ! -e "$NODE_LINK" ]]; then
  ln -s "$NODE_SRC" "$NODE_LINK"
  echo "symlinked EXP node -> custom_nodes/H3-SpargeAttn-Exp"
fi

mkdir -p "$CP/logs" "$EXP/output"
cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${VENDOR}${PYTHONPATH:+:$PYTHONPATH}"
# cache-classic: keep LoadUNet/CLIP/VAE/LoRA node outputs across prompts
# Do NOT use --gpu-only/--highvram here: H3 CLIP ~15GB + UNet ~20GB + activations
# exceeds a 32GB 5090 and OOMs on the first sampler step.
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --database-url "sqlite:///${CP}/user/comfyui_${PORT}.db" \
  > "$EXP/output/comfyui_exp_${PORT}.log" 2>&1 &
echo $! | tee "$EXP/output/comfyui_exp_${PORT}.pid"
echo "EXP ComfyUI GPU${GPU}: http://127.0.0.1:${PORT}  (PYTHONPATH=$VENDOR)"
