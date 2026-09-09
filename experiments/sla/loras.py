"""LoRA filenames for SLA experiment (discovered via experiments/sla/extra_model_paths.yaml)."""

# SLA 4-step FL2V (ComfyUI bf16) — file lives under experiments/sla/ only
LORA_SLA_T2V = (
    "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
)

# Production baseline (for A/B; from big_models via Comfy's normal paths)
LORA_PROD_T2V = (
    "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"
)
