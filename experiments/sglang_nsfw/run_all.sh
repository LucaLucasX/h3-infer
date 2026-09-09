#!/usr/bin/env bash
# Sequential 2×5090 NSFW MiniMax-H3 SGLang test. Does not modify /mnt/sglang-fork.
set -euo pipefail

SGLANG=/mnt/sglang-fork/.venv/bin/sglang
MODEL=/big_models/MiniMax-H3
TE=/big_models/Qwen3-VL-32B-Instruct-FP8
OUT=/mnt/luca/H3_infer/ComfyUI-master_cp/output/exp_sglang_nsfw
EXP=/mnt/luca/H3_infer/experiments/sglang_nsfw
PY=/mnt/sglang-fork/.venv/bin/python

FL_DIT=/mnt/gly/h3_transformer_nsfw/FL2VA/transformer-lora-fp8-v1.1-nsfw
REF_DIT=/mnt/gly/h3_transformer_nsfw/Ref2VA/transformer-lora-fp8-v0.1-nsfw

mkdir -p "${OUT}/logs"

kill_port_tree() {
  local port="$1"
  local pids
  pids=$(ss -lptn 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 2
    kill -9 ${pids} 2>/dev/null || true
  fi
  pkill -f "sglang serve" 2>/dev/null || true
  sleep 2
}

launch() {
  local variant="$1" dit="$2" port="$3" master="$4"
  local log="${OUT}/logs/${variant}_serve.log"
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
  echo $! > "${OUT}/logs/${variant}.pid"
  echo "pid=$(cat "${OUT}/logs/${variant}.pid") log=${log}"
}

"${PY}" "${EXP}/build_cases.py"

echo "===== FL2VA ====="
kill_port_tree 30020 || true
kill_port_tree 30025 || true
launch fl2va "${FL_DIT}" 30020 30025
"${PY}" "${EXP}/run_cases.py" --base http://127.0.0.1:30020 --server fl2va --wait-ready-sec 1200
echo "stopping FL2VA"
kill_port_tree 30020 || true
sleep 5

echo "===== Ref2VA ====="
kill_port_tree 30030 || true
kill_port_tree 30035 || true
launch ref2va "${REF_DIT}" 30030 30035
"${PY}" "${EXP}/run_cases.py" --base http://127.0.0.1:30030 --server ref2va --wait-ready-sec 1200
echo "stopping Ref2VA"
kill_port_tree 30030 || true

echo "ALL DONE"
ls -lh "${OUT}"/*/ 2>/dev/null | head
