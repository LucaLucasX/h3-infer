#!/usr/bin/env bash
# MiniMax-H3 FL2VA on 2×RTX5090 — isolated env, fastest that fits 32GB.
# TP=2, Ulysses=1, SageAttention, Turbo-FP8.
#
# Pure all-resident speed OOMs (DiT TP2 shard ~27GB/GPU).
# Bottleneck on prior runs: VAE decode ~84s under layerwise VAE.
# Fast path: DiT+TE layerwise (DiT can use high resident), VAE fully GPU-resident.
# Does NOT touch /mnt/mjb/*.
set -euo pipefail

ROOT=/mnt/luca/H3_infer/experiments/sglang_h3
OUT="${ROOT}/output"
mkdir -p "${OUT}"

VENV="${ROOT}/.venv"
MODEL=/big_models/MiniMax-H3
TRANSFORMER="${TRANSFORMER_PATH:-${MODEL}/FL2VA/transformer-turbo-FP8-fl-ckpt850}"
PORT="${PORT:-30010}"
MASTER_PORT="${MASTER_PORT:-30046}"
LOG="${OUT}/sglang_tp2_sage_turbo_speed.log"
PID_FILE="${OUT}/sglang_tp2.pid"
PERF_MODE="${PERF_MODE:-speed}"
# With VAE GPU-resident (~9.7GB), DiT resident>0 OOMs on 12s 1344×768 denoise.
# Denoise is compute-bound anyway; resident=0 is the working fast path.
RESIDENT_LAYERS="${DIT_RESIDENT_LAYERS:-0}"
PREFETCH="${DIT_PREFETCH:-1}"

if [[ ! -x "${VENV}/bin/sglang" ]]; then
  echo "Isolated venv missing: ${VENV}" >&2
  exit 1
fi
if [[ ! -d "${TRANSFORMER}" ]]; then
  echo "Transformer missing: ${TRANSFORMER}" >&2
  exit 1
fi

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
    rm -f "${PID_FILE}"
  fi
  ps -eo pid,cmd | awk -v v="${VENV}/bin/" '
    index($0, v) && /sglang serve/ && !/awk/ { print $1 }
  ' | xargs -r kill 2>/dev/null || true
  echo "stopped SGLang :${PORT}"
  exit 0
fi

if curl -sS -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "SGLang already up on :${PORT} — run: $0 stop   then relaunch"
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_USE_RUNAI_MODEL_STREAMER="${SGLANG_USE_RUNAI_MODEL_STREAMER:-0}"

cd "${OUT}"
nohup "${VENV}/bin/sglang" serve \
  --model-path "${MODEL}" \
  --model-variant fl2va \
  --transformer-weights-path "${TRANSFORMER}" \
  --num-gpus 2 \
  --tp-size 2 \
  --ulysses-degree 1 \
  --performance-mode "${PERF_MODE}" \
  --dit-cpu-offload true \
  --dit-layerwise-offload true \
  --dit-layerwise-resident-layers "${RESIDENT_LAYERS}" \
  --dit-offload-prefetch-size "${PREFETCH}" \
  --layerwise-offload-components dit text_encoder \
  --vae-cpu-offload false \
  --attention-backend sage_attn \
  --component-attention-backends "text_encoder=torch_sdpa,transformer=sage_attn" \
  --enable-torch-compile false \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --master-port "${MASTER_PORT}" \
  >"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "SGLang SPEED pid=$(cat "${PID_FILE}") port=${PORT} mode=${PERF_MODE}"
echo "venv=${VENV} resident=${RESIDENT_LAYERS} (VAE GPU-resident; DiT+TE layerwise)"
echo "log=${LOG}"
