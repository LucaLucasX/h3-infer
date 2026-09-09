#!/usr/bin/env bash
set -euo pipefail

# Two-stage + LBH learned latent upscaler (12s, 1344x768, 53 cases)
# Recommended default:
# - total steps: 6
# - split_step: 4 (pass-1 keeps your 4-step turbo path, pass-2 does 2-step refine)
# - sparge topk=0.5 + audio_dense=true

ROOT="/mnt/luca/H3_infer"
PY="$ROOT/.venv_sage_bench/bin/python"
RUNNER="$ROOT/experiments/sparse_attn/run_textcases_twostage_lbh_upscaler.py"

LORA="${LORA:-lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors}"
UPS="${UPS:-minimax_h3_latent_upscaler_3d_fp16.safetensors}"

mkdir -p "$ROOT/experiments/sparse_attn/output"

echo "[1/2] launch GPU0 :8190"
nohup "$PY" -u "$RUNNER" \
  --comfy http://127.0.0.1:8190 \
  --steps 6 \
  --split-step 4 \
  --twostep-scale 2.0 \
  --topk 0.5 \
  --audio-dense \
  --upscaler-node 3d \
  --upscaler-model "$UPS" \
  --upscaler-precision fp16 \
  --turbo-lora "$LORA" \
  --out-root exp_textcases_12s_twostage_lbh3d_sparge_gpu0 \
  > "$ROOT/experiments/sparse_attn/output/textcases_twostage_lbh3d_gpu0_run.log" 2>&1 &
echo "gpu0_pid=$!"

echo "[2/2] launch GPU1 :8191"
nohup "$PY" -u "$RUNNER" \
  --comfy http://127.0.0.1:8191 \
  --steps 6 \
  --split-step 4 \
  --twostep-scale 2.0 \
  --topk 0.5 \
  --audio-dense \
  --upscaler-node 3d \
  --upscaler-model "$UPS" \
  --upscaler-precision fp16 \
  --turbo-lora "$LORA" \
  --out-root exp_textcases_12s_twostage_lbh3d_sparge_gpu1 \
  > "$ROOT/experiments/sparse_attn/output/textcases_twostage_lbh3d_gpu1_run.log" 2>&1 &
echo "gpu1_pid=$!"

echo "launched."

