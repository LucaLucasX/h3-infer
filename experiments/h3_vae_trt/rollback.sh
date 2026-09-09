#!/usr/bin/env bash
# Rollback the MiniMax-H3 VAE TRT experiment inside ComfyUI-master_cp.
# Does not touch ComfyUI core, extra_model_paths.yaml, existing workflows,
# production VAE safetensors, /root/agnes_2_5, GPU1 :8196, or .venv_sage_bench.
set -euo pipefail
CP="/mnt/luca/H3_infer/ComfyUI-master_cp"
echo "[1] remove plugin (and its local .deps)"
rm -rf "$CP/custom_nodes/ComfyUI-H3VAE_TRT"
echo "[2] remove ONNX / TRT engines added by this experiment"
rm -f \
  "$CP/models/vae/minimax_h3_vae_decoder.onnx" \
  "$CP/models/vae/minimax_h3_vae_decoder.onnx.data" \
  "$CP/models/vae/minimax_h3_vae_encoder.onnx" \
  "$CP/models/vae/minimax_h3_vae_decoder.engine" \
  "$CP/models/vae/minimax_h3_vae_encoder.engine" \
  "$CP/models/vae/minimax_h3_vae_decoder_w4a16_awq.onnx" \
  "$CP/models/vae/README.md" \
  "$CP/models/vae/.gitattributes"
rm -rf "$CP/models/vae/.cache"
rm -rf "$CP/user/default/workflows/exp_h3_vae_trt"
rm -f "$CP/user/default/workflows/exp_h3_vae_trt"*.json
echo "done. production VAE safetensors and Sparge/Sage/4-step graphs are unchanged."
