#!/usr/bin/env bash
# PDD Acc experiment ComfyUI on shared ComfyUI-master_cp.
# Port default 8193; GPU default 1.
set -euo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_PDD_PORT:-8193}"
GPU="${H3_PDD_GPU:-1}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
NODE_SRC="$EXP/custom_nodes/ComfyUI-MiniMax-H3-PDD-Acc"
NODE_LINK="$CP/custom_nodes/ComfyUI-MiniMax-H3-PDD-Acc"
PDD_MODELS="$EXP/models/pdd_acc"
PDD_LINK="$CP/models/pdd_acc"
EXTRA_YAML="$EXP/extra_model_paths.yaml"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped PDD-exp ComfyUI :${PORT}"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV" >&2
  exit 1
fi
if [[ ! -d "$CP" ]]; then
  echo "missing $CP" >&2
  exit 1
fi
if [[ ! -d "$NODE_SRC" ]]; then
  echo "missing PDD node pack at $NODE_SRC" >&2
  exit 1
fi

mkdir -p "$EXP/logs" "$CP/user" "$CP/models" "$PDD_MODELS"
ln -sfn "$NODE_SRC" "$NODE_LINK"
ln -sfn "$PDD_MODELS" "$PDD_LINK"

cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_pdd_${PORT}.db" \
  > "$EXP/logs/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$EXP/logs/comfyui_${PORT}.pid"
echo "pdd-exp  ComfyUI-master_cp GPU${GPU}  http://127.0.0.1:${PORT}"
echo "log: $EXP/logs/comfyui_${PORT}.log"
echo "pdd files: $PDD_MODELS"
