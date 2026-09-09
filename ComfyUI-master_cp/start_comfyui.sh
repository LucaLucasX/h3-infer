#!/usr/bin/env bash
# Start ONE ComfyUI-master_cp (MiniMax-H3). Single GPU (cuda:0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p output input temp logs user/default/workflows
PORT="${COMFY_PORT:-8188}"

echo "ComfyUI-master_cp MiniMax-H3 -> http://0.0.0.0:${PORT}"
echo "Workflows: workflows/"
echo "Models:    /big_models/comfyui-minimax-H3/ (extra_model_paths.yaml)"
echo "GPU:       --cuda-device ${COMFY_CUDA_DEVICE:-0}"

exec python main.py \
  --listen 0.0.0.0 \
  --port "${PORT}" \
  --cuda-device "${COMFY_CUDA_DEVICE:-0}" \
  --preview-method none \
  ${COMFY_ARGS:-}
