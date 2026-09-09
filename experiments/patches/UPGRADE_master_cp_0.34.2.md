# ComfyUI-master_cp upgrade 0.30.0 → 0.34.2

Date: 2026-09-01

## Backup
- Backup: removed after consolidating to single ComfyUI-master_cp (0.34.2).
- Local patch export: `experiments/patches/master_cp_0.30_local.patch`

## Method
- `rsync` ComfyUI upstream 0.34.2 → ComfyUI-master_cp, **preserving**:
  - `custom_nodes/`, `models/`, `output/`, `user/`, `input/`, `temp/`, `logs/`, `.git/`
- Re-applied local forks (timing + Sparge/MagCache hooks):
  - `comfy/ldm/minimax/model.py` — `h3_pack_segments`, `block_loop` / `_run_blocks`
  - `comfy/text_encoders/minimax.py` — `[H3_TE_TIMING]`
  - `nodes.py` — `[VAE_VIDEO_TIMING]`
  - `comfy_extras/nodes_audio.py` — `[VAE_AUDIO_TIMING]`
  - `comfy_extras/nodes_video.py` — `[CREATE_VIDEO_TIMING]`, `[SAVE_VIDEO_TIMING]`
  - `comfy_extras/nodes_minimax_h3.py` — `[H3_VAE_TIMING]`, `[H3_REF_VIDEO]`
- Restored `kill.sh`, `start_comfyui.sh`, `extra_model_paths.yaml` from backup
- `.venv_sage_bench`: comfy-kitchen 0.2.31, comfy-aimdo 0.4.15

## Custom nodes (unchanged symlinks)
- `H3-SpargeAttn-Exp` → `dev/sage_lora_sparge/custom_nodes/`
- `H3-SLAAttn-Exp` → `experiments/sla/custom_nodes/`
- `ComfyUI-MiniMax-H3-Turbo` (4-step LoRA)
- Other experimental nodes in `custom_nodes/` retained as-is

## Start
```bash
dev/sage_lora_sparge/scripts/run_comfy.sh   # :8190 GPU0
```
