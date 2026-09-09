#!/usr/bin/env bash
# Production stack: Sage2 MemEff + 4-step Turbo LoRA + SpargeAttn
# Does not copy weights. Uses ComfyUI-master_cp + .venv_sage_bench.
set -euo pipefail
DEV="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$DEV/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_SPARGE_PORT:-8190}"
GPU="${H3_SPARGE_GPU:-0}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
VENDOR="${H3_SPARGE_VENDOR:-$REPO/experiments/sparse_attn/vendor/SpargeAttn}"
NODE_LINK="$CP/custom_nodes/H3-SpargeAttn-Exp"
NODE_SRC="$DEV/custom_nodes/H3-SpargeAttn-Exp"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped ComfyUI :${PORT}"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV" >&2
  exit 1
fi
if [[ ! -e "$VENDOR/spas_sage_attn" ]]; then
  echo "missing SpargeAttn vendor at $VENDOR" >&2
  exit 1
fi

ln -sfn "$NODE_SRC" "$NODE_LINK"
mkdir -p "$DEV/logs" "$CP/user"
cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${VENDOR}${PYTHONPATH:+:$PYTHONPATH}"
EXTRA_YAML="$CP/extra_model_paths.yaml"
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_${PORT}.db" \
  > "$DEV/logs/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$DEV/logs/comfyui_${PORT}.pid"
echo "sage+lora+sparge  GPU${GPU}  http://127.0.0.1:${PORT}"
echo "log: $DEV/logs/comfyui_${PORT}.log"
