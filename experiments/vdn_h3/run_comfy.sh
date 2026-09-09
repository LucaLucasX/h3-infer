#!/usr/bin/env bash
# VDN experiment ComfyUI: master_cp + .venv_sage_bench, no Sage / Sparge / turbo LoRA.
# Those are graph options; this launcher also does not link Sparge nodes or put
# SpargeAttn on PYTHONPATH.
set -euo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_VDN_PORT:-8190}"
GPU="${H3_VDN_GPU:-0}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
LOG_DIR="$EXP/logs"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped VDN ComfyUI :${PORT}"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$CP/user"
cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset PYTHONPATH
EXTRA_YAML="$CP/extra_model_paths.yaml"
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_${PORT}.db" \
  > "$LOG_DIR/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$LOG_DIR/comfyui_${PORT}.pid"
echo "vdn  GPU${GPU}  http://127.0.0.1:${PORT}"
echo "log: $LOG_DIR/comfyui_${PORT}.log"
