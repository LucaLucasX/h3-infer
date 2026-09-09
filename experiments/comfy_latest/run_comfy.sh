#!/usr/bin/env bash
# Legacy alias: Kitchen/Sparge A/B on shared ComfyUI-master_cp (:8192 GPU1).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
VENV="${VENV:-$REPO/.venv_sage_bench}"
PORT="${H3_LATEST_PORT:-8192}"
GPU="${H3_LATEST_GPU:-1}"
SPARGE_VENDOR="${H3_SPARGE_VENDOR:-$REPO/experiments/sparse_attn/vendor/SpargeAttn}"
EXTRA_YAML="$REPO/experiments/comfy_latest/extra_model_paths.yaml"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped ComfyUI :${PORT}"
  exit 0
fi

mkdir -p "$REPO/experiments/comfy_latest/logs" "$CP/user"
cd "$CP"
export PYTHONPATH="${SPARGE_VENDOR}${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  --listen 0.0.0.0
  --port "$PORT"
  --cuda-device "$GPU"
  --preview-method none
  --cache-classic
  --use-ck-attention
  --extra-model-paths-config "$EXTRA_YAML"
  --database-url "sqlite:///${CP}/user/comfyui_${PORT}.db"
)
nohup "$VENV/bin/python" main.py "${ARGS[@]}" \
  > "$REPO/experiments/comfy_latest/logs/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$REPO/experiments/comfy_latest/logs/comfyui_${PORT}.pid"
echo "ComfyUI-master_cp GPU${GPU} http://127.0.0.1:${PORT} (--use-ck-attention)"
echo "log: $REPO/experiments/comfy_latest/logs/comfyui_${PORT}.log"
