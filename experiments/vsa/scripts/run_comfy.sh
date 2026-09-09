#!/usr/bin/env bash
# VSA experiment ComfyUI: registers H3-VSAAttn-Exp (+ Sparge already present).
# Port default 8197 so it does not collide with production or Agnes.
set -euo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_VSA_PORT:-8197}"
GPU="${H3_VSA_GPU:-0}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
SPARGE_VENDOR="${H3_SPARGE_VENDOR:-$REPO/experiments/sparse_attn/vendor/SpargeAttn}"
NODE_VSA_LINK="$CP/custom_nodes/H3-VSAAttn-Exp"
NODE_VSA_SRC="$EXP/custom_nodes/H3-VSAAttn-Exp"
NODE_SPARGE_LINK="$CP/custom_nodes/H3-SpargeAttn-Exp"
NODE_SPARGE_SRC="$REPO/dev/sage_lora_sparge/custom_nodes/H3-SpargeAttn-Exp"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped ComfyUI :${PORT}"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV" >&2
  exit 1
fi
if [[ ! -e "$SPARGE_VENDOR/spas_sage_attn" ]]; then
  echo "missing SpargeAttn vendor at $SPARGE_VENDOR" >&2
  exit 1
fi
if [[ "$CP" != *ComfyUI-master_cp ]]; then
  echo "expected ComfyUI-master_cp, got $CP" >&2
  exit 1
fi

ln -sfn "$NODE_VSA_SRC" "$NODE_VSA_LINK"
ln -sfn "$NODE_SPARGE_SRC" "$NODE_SPARGE_LINK"
mkdir -p "$EXP/logs" "$CP/user"
cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SPARGE_VENDOR}${PYTHONPATH:+:$PYTHONPATH}"
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --extra-model-paths-config "$CP/extra_model_paths.yaml" \
  --database-url "sqlite:///${CP}/user/comfyui_vsa_${PORT}.db" \
  > "$EXP/logs/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$EXP/logs/comfyui_${PORT}.pid"
echo "vsa-exp  GPU${GPU}  http://127.0.0.1:${PORT}"
echo "log: $EXP/logs/comfyui_${PORT}.log"
