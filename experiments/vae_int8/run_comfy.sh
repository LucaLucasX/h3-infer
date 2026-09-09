#!/usr/bin/env bash
# Kijai int8_convrot VAE — L0 stack on ComfyUI-master_cp (0.34 + comfy-kitchen).
set -euo pipefail
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
CP="$REPO/ComfyUI-master_cp"
PORT="${H3_VAE_INT8_PORT:-8194}"
GPU="${H3_VAE_INT8_GPU:-1}"
VENV="${VENV:-$REPO/.venv_sage_bench}"
SPARGE_VENDOR="${H3_SPARGE_VENDOR:-$REPO/experiments/sparse_attn/vendor/SpargeAttn}"
NODE_SRC="$REPO/dev/sage_lora_sparge/custom_nodes/H3-SpargeAttn-Exp"
NODE_LINK="$CP/custom_nodes/H3-SpargeAttn-Exp"
EXTRA_YAML="$EXP/extra_model_paths.yaml"
LOG_DIR="$EXP/logs"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "${VENV}/bin/python main.py --listen 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "stopped vae_int8 ComfyUI :${PORT}"
  exit 0
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing $VENV" >&2
  exit 1
fi
if [[ ! -f "$CP/main.py" ]]; then
  echo "missing $CP/main.py" >&2
  exit 1
fi

ln -sfn "$NODE_SRC" "$NODE_LINK"
mkdir -p "$LOG_DIR" "$CP/user"
mkdir -p "$CP/models"/{checkpoints,clip,clip_vision,controlnet,diffusion_models,loras,text_encoders,upscale_models,vae,vae_approx,unet,latent_upscale_models}
cd "$CP"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SPARGE_VENDOR}${PYTHONPATH:+:$PYTHONPATH}"

# --cache-none: benchmark runs must not hit classic cache (false sub-second timings).
nohup "$VENV/bin/python" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --cuda-device "$GPU" \
  --preview-method none \
  --cache-none \
  --extra-model-paths-config "$EXTRA_YAML" \
  --database-url "sqlite:///${CP}/user/comfyui_vae_int8_${PORT}.db" \
  > "$LOG_DIR/comfyui_${PORT}.log" 2>&1 &
echo $! | tee "$LOG_DIR/comfyui_${PORT}.pid"
echo "vae_int8 L0 GPU${GPU} http://127.0.0.1:${PORT} (no lossless-mem, cache-none)"
echo "log: $LOG_DIR/comfyui_${PORT}.log"
