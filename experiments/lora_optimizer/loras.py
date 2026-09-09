"""LoRA filenames for offline-merge + TurboLoRA experiments."""

LORA_PROD_T2V = (
    "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"
)

LORA_SLA_T2V = (
    "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
)

# LoRA Optimizer analyze + SaveMergedLoRA + fc2 patch (see bake_optimizer_lora.py).
LORA_OPTIMIZER_BAKED = (
    "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_optimizer_baked_h3keys.safetensors"
)

# Fixed-ratio rank-concat reference only (NOT optimizer analysis).
LORA_MERGE_PROD07_SLA03 = (
    "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_merge_prod0.7_sla0.3_h3keys.safetensors"
)
