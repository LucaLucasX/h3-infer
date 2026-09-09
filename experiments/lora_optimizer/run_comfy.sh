#!/usr/bin/env bash
# LoRA Optimizer + SLA experiment on ComfyUI-master_cp.
set -euo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_LORA_OPT_PORT:-8195}"
GPU="${H3_LORA_OPT_GPU:-0}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
SPARGE_VENDOR="${H3_SPARGE_VENDOR:-$REPO/experiments/sparse_attn/vendor/SpargeAttn}"
SLA_VENDOR="$REPO/experiments/sla/vendor"
NODE_OPT_SRC="$EXP/custom_nodes/ComfyUI-LoRA-Optimizer"
NODE_OPT_LINK="$CP/custom_nodes/ComfyUI-LoRA-Optimizer"
NODE_SLA_SRC="$REPO/experiments/sla/custom_nodes/H3-SLAAttn-Exp"
NODE_SLA_LINK="$CP/custom_nodes/H3-SLAAttn-Exp"
EXTRA_YAML="$EXP/extra_model_paths.yaml"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped lora-opt ComfyUI :${PORT}"
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
if [[ ! -e "$SPARGE_VENDOR/spas_sage_attn" ]]; then
  echo "missing SpargeAttn vendor (SLA sage2 kernel): $SPARGE_VENDOR" >&2
  exit 1
fi

mkdir -p "$EXP/logs" "$CP/user"
ln -sfn "$NODE_OPT_SRC" "$NODE_OPT_LINK"
ln -sfn "$NODE_SLA_SRC" "$NODE_SLA_LINK"

cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${SLA_VENDOR}:${SPARGE_VENDOR}${PYTHONPATH:+:$PYTHONPATH}"
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-classic \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_lora_opt_${PORT}.db" \
  > "$EXP/logs/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$EXP/logs/comfyui_${PORT}.pid"
echo "lora-opt  ComfyUI-master_cp GPU${GPU}  http://127.0.0.1:${PORT}"
echo "log: $EXP/logs/comfyui_${PORT}.log"
