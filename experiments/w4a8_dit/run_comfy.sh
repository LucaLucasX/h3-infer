#!/usr/bin/env bash
# w4a8 DiT benchmark on shared ComfyUI-master_cp (GPU1 :8194, cache-none).
set -euo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_W4A8_PORT:-8194}"
GPU="${H3_W4A8_GPU:-1}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
SPARGE_VENDOR="${H3_SPARGE_VENDOR:-$REPO/experiments/sparse_attn/vendor/SpargeAttn}"
NODE_SRC="$REPO/dev/sage_lora_sparge/custom_nodes/H3-SpargeAttn-Exp"
NODE_LINK="$CP/custom_nodes/H3-SpargeAttn-Exp"
EXTRA_YAML="$CP/extra_model_paths.yaml"
LOG_DIR="$EXP/logs"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped w4a8 ComfyUI :${PORT}"
  exit 0
fi

mkdir -p "$LOG_DIR" "$CP/user"
ln -sfn "$NODE_SRC" "$NODE_LINK"
cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SPARGE_VENDOR}${PYTHONPATH:+:$PYTHONPATH}"

nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-none \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_w4a8_${PORT}.db" \
  > "$LOG_DIR/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$LOG_DIR/comfyui_${PORT}.pid"
echo "w4a8_dit L0 GPU${GPU} http://127.0.0.1:${PORT} (cache-none)"
echo "log: $LOG_DIR/comfyui_${PORT}.log"
