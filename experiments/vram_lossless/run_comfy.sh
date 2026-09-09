#!/usr/bin/env bash
# GPU1 experiment ComfyUI for lossless H3 mem patch (shared ComfyUI-master_cp).
set -euo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
PDD="$REPO/experiments/pdd/run_comfy.sh"
CP="$REPO/ComfyUI-master_cp"
NODE_SRC="$EXP/custom_nodes/ComfyUI-H3-LosslessMem"
NODE_LINK="$CP/custom_nodes/ComfyUI-H3-LosslessMem"

if [[ "${1:-}" == "stop" ]]; then
  "$PDD" stop
  exit 0
fi

if [[ ! -d "$NODE_SRC" ]]; then
  echo "missing lossless mem node at $NODE_SRC" >&2
  exit 1
fi

ln -sfn "$NODE_SRC" "$NODE_LINK"

# Recycle GPU1 :8193 so the new node is imported. Production :8190 is left alone.
"$PDD" stop || true
sleep 2
PORT="${H3_PDD_PORT:-8193}"
GPU="${H3_PDD_GPU:-1}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
EXTRA_YAML="$REPO/experiments/pdd/extra_model_paths.yaml"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CP"
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_pdd_${PORT}.db" \
  > "$REPO/experiments/pdd/logs/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$REPO/experiments/pdd/logs/comfyui_${PORT}.pid"
echo "lossless-mem GPU${GPU} http://127.0.0.1:${PORT} (no reserve-vram, expandable_segments)"
echo "lossless-mem node: $NODE_LINK"
