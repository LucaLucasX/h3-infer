# PDD Acc experiment (isolated)

Parallel Decoding Distillation Acc LoRAs ([alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)) on **ComfyUI-latest only**.

## Isolation rules

- Does **not** touch `ComfyUI-master_cp`, production Turbo/Sparge graphs, or `:8191`.
- Custom node lives under `experiments/pdd/custom_nodes/` and is symlinked only into `ComfyUI-latest/custom_nodes/`.
- Weights live under `experiments/pdd/models/pdd_acc/` (not `/big_models`).
- Default port **8193**, GPU **1**.

## Setup

```bash
# 1) Download FL2VA Acc (into experiments/pdd/models/pdd_acc/)
hf download alibaba-pai/MiniMax-H3-Acc-LoRAs MiniMax-H3-FL2VA-Acc-8Step.safetensors \
  --local-dir /mnt/luca/H3_infer/experiments/pdd/models/pdd_acc

# 2) Start (ComfyUI-latest >= 0.33 required)
bash experiments/pdd/run_comfy.sh

# 3) Smoke
.venv_comfy_latest/bin/python experiments/pdd/scripts/run_smoke.py \
  --comfy http://127.0.0.1:8193 --length 124
```

Stop: `bash experiments/pdd/run_comfy.sh stop`

## Recipe (fixed)

| Item | Value |
|------|-------|
| UNET | `minimax_h3_fl2va_int8_convrot` (full, not pruned) |
| PDD | `MiniMax-H3-FL2VA-Acc-8Step.safetensors` |
| Sampler | euler |
| Sigmas | from `MiniMaxH3PDDAccApply` |
| CFG | 1.0 (BasicGuider) |
| SigmaShift | 12 / 3 |
| NFE | 8 (default) or 4 |
| Do **not** stack | Turbo LoRA, SLA LoRA, Sparge, Kitchen, TeaCache, Spectrum |

## Node pack

Vendored from [Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc).
