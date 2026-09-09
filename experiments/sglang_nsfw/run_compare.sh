#!/usr/bin/env bash
# NSFW vs official MiniMax-H3, same 4 cases/mode. Reuses live FL2VA NSFW on :30020 if healthy.
set -euo pipefail

SGLANG=/mnt/sglang-fork/.venv/bin/sglang
PY=/mnt/sglang-fork/.venv/bin/python
MODEL=/big_models/MiniMax-H3
TE=/big_models/Qwen3-VL-32B-Instruct-FP8
OUT=/mnt/luca/H3_infer/ComfyUI-master_cp/output/exp_sglang_compare
EXP=/mnt/luca/H3_infer/experiments/sglang_nsfw

FL_NSFW=/mnt/gly/h3_transformer_nsfw/FL2VA/transformer-lora-fp8-v1.1-nsfw
FL_OFF=/big_models/MiniMax-H3/FL2VA/transformer-lora-fp8-v1.1
REF_NSFW=/mnt/gly/h3_transformer_nsfw/Ref2VA/transformer-lora-fp8-v0.1-nsfw
REF_OFF=/big_models/MiniMax-H3/Ref2VA/transformer-lora-fp8

mkdir -p "${OUT}/logs"

kill_sglang() {
  pkill -f "sglang serve" 2>/dev/null || true
  sleep 3
  pkill -9 -f "sglang serve" 2>/dev/null || true
  sleep 2
}

launch() {
  local variant="$1" dit="$2" port="$3" master="$4"
  local log="${OUT}/logs/${variant}_${port}_serve.log"
  local extra=()
  if [[ "${variant}" == "ref2va" ]]; then
    extra+=(
      --minimax-h3-reference-image-short-edge 1024
      --minimax-h3-reference-video-short-edge 540
    )
  fi
  echo "launch ${variant} port=${port} dit=${dit}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0,1
  mkdir -p "${OUT}"
  cd "${OUT}"
  nohup "${SGLANG}" serve \
    --model-path "${MODEL}" \
    --transformer-path "${dit}" \
    --text-encoder-path "${TE}" \
    --num-gpus 2 --tp-size 2 --ulysses-degree 1 \
    --performance-mode memory \
    --layerwise-offload-components dit,text_encoder \
    --dit-offload-prefetch-size 1 \
    --dit-layerwise-resident-layers 0 \
    --enable-torch-compile false \
    --attention-backend sage_attn \
    --component-attention-backends text_encoder=torch_sdpa \
    --host 0.0.0.0 --port "${port}" \
    --master-port "${master}" \
    --model-variant "${variant}" \
    "${extra[@]}" \
    >"${log}" 2>&1 &
  echo $! > "${OUT}/logs/serve.pid"
  echo "pid=$(cat "${OUT}/logs/serve.pid") log=${log}"
}

"${PY}" "${EXP}/build_compare_cases.py"

echo "===== 1/4 FL2VA NSFW ====="
kill_sglang
launch fl2va "${FL_NSFW}" 30020 30025
"${PY}" "${EXP}/run_compare.py" --base http://127.0.0.1:30020 --server fl2va --tag nsfw --wait-ready-sec 1200

echo "===== 2/4 FL2VA official ====="
kill_sglang
launch fl2va "${FL_OFF}" 30020 30025
"${PY}" "${EXP}/run_compare.py" --base http://127.0.0.1:30020 --server fl2va --tag official --wait-ready-sec 1200

echo "===== 3/4 Ref2VA NSFW ====="
kill_sglang
launch ref2va "${REF_NSFW}" 30030 30035
"${PY}" "${EXP}/run_compare.py" --base http://127.0.0.1:30030 --server ref2va --tag nsfw --wait-ready-sec 1200

echo "===== 4/4 Ref2VA official ====="
kill_sglang
launch ref2va "${REF_OFF}" 30030 30035
"${PY}" "${EXP}/run_compare.py" --base http://127.0.0.1:30030 --server ref2va --tag official --wait-ready-sec 1200

echo "ALL COMPARE DONE"
find "${OUT}" -name '*.mp4' | sort
