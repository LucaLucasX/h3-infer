"""lightx2v 4-step Turbo LoRAs used by this production stack."""

# t2v / fl2va (first-last): remapped keys for MiniMaxH3TurboLoRA
LORA_T2V = "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"

# r2v / ref2va (multi-ref images + videos)
LORA_R2V = "lightx2v_fl2v/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
