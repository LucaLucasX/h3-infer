# Sage2 + 4-step Turbo LoRA + Sparge（生产 / dev）

720p t2v 暂定生产方案。本目录只抽代码和启动方式，**不拷贝权重**。

实测稳定墙钟（`--cache-classic`，warmup 之后）：约 **125s / 12s 视频**（1344×768）。

不要用 Sage3。不要加 `--gpu-only` / `--highvram`（5090 32GB 会 OOM：CLIP ~15GB + UNet ~20GB）。

## 目录

```
dev/sage_lora_sparge/
  custom_nodes/H3-SpargeAttn-Exp/   # Sparge Comfy 节点
  vendor_SpargeAttn/                  # SpargeAttn 源码 + 已编译 .so（实体拷贝，非软链）
  scripts/run_comfy.sh
  scripts/run_t2v.py
  extra_model_paths.yaml            # 仅作路径说明
```

Comfy 本体仍用仓库里的 `ComfyUI-master_cp/`，venv 仍用 `.venv_sage_bench`。

## 环境

| 项 | 值 |
|---|---|
| 机器 | Linux，NVIDIA RTX 5090 32GB（sm_120） |
| Python | 3.11（`.venv_sage_bench`） |
| PyTorch | 2.13.0+cu130 |
| SageAttention | **2.2.0**（`sage_v2_memeff`，site-packages） |
| SpargeAttn | `spas_sage_attn`，本目录 `vendor_SpargeAttn` |
| ComfyUI | `/mnt/luca/H3_infer/ComfyUI-master_cp` |
| 启动 flag | `--preview-method none --cache-classic`（`NORMAL_VRAM` + DynamicVRAM；**不是** GPU 常驻 UNet/CLIP） |

依赖的 Comfy 节点（已在 `ComfyUI-master_cp/custom_nodes`，未拷进本目录）：

- `ComfyUI-MiniMax-H3-Turbo`（4 步 LoRA）
- `ComfyUI-KJNodes`（图脚手架里的 Sage patch 会被 Sparge 节点替换）
- MiniMax H3 官方节点（UNET / CLIP / VAE / ImageToVideo / Sampler）

## 权重路径（不拷贝）

Comfy 通过 `ComfyUI-master_cp/extra_model_paths.yaml` 读 `/big_models/comfyui-minimax-H3/`。

| 角色 | 文件名 | 绝对路径 | 约大小 |
|---|---|---|---|
| UNet | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `/big_models/comfyui-minimax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20G |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `/big_models/comfyui-minimax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15G |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `/big_models/comfyui-minimax-H3/vae/minimax_h3_video_vae_fp16.safetensors` | 4.9G |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `/big_models/comfyui-minimax-H3/vae/minimax_h3_audio_vae_fp32.safetensors` | 578M |
| t2v LoRA | `lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors` | `/big_models/comfyui-minimax-H3/loras/lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors` | 1.9G |
| r2v LoRA | `lightx2v_fl2v/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `/big_models/comfyui-minimax-H3/loras/lightx2v_fl2v/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.9G |

备用目录（同名文件也可能在）：`/mnt/myx/ComfyUI/models/`。

## 图参数

| 项 | 值 |
|---|---|
| 分辨率 | 1344×768 |
| 时长 | length=288，24fps（12s） |
| steps | **4** |
| Attention | `MiniMaxH3SpargeAttnPatchExp`（内部 dense 走 Sage2） |
| Sparge | `topk=0.5`，`audio_dense=true`，`dense_first_steps=0`，`num_layers=50` |
| LoRA | t2v：lightx2v fl2v 4step v1.0 h3keys；r2v：lightx2v ref2v 4step v0.1；strength=1.0 |

## 启动

```bash
cd /mnt/luca/H3_infer
H3_SPARGE_PORT=8190 H3_SPARGE_GPU=0 ./dev/sage_lora_sparge/scripts/run_comfy.sh
```

停：`./dev/sage_lora_sparge/scripts/run_comfy.sh stop`

UI：http://127.0.0.1:8190

## 命令行跑 case

```bash
cd /mnt/luca/H3_infer
.venv_sage_bench/bin/python -u dev/sage_lora_sparge/scripts/run_t2v.py \
  --comfy http://127.0.0.1:8190 \
  --only o02 o03 o04
```

先 warmup 再测。warmup 若和某条 prompt 相同，那条会被 `--cache-classic` 命中，时间会假短。
