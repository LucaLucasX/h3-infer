#!/usr/bin/env bash
# Isolated TensorRT env for the H3 VAE TRT experiment.
# Does NOT activate extra packages into .venv_sage_bench.
DEPS="/mnt/luca/H3_infer/ComfyUI-master_cp/custom_nodes/ComfyUI-H3VAE_TRT/.deps"
export PYTHONPATH="${DEPS}${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${DEPS}/tensorrt_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
