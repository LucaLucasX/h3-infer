# SGLang MiniMax-H3（本仓库独立环境）

## 环境说明

- **独立 venv**：`experiments/sglang_h3/.venv`（从他人环境 **只读复制** 后改 shebang，**不修改** `/mnt/mjb/...`）
- 权重：`/big_models/MiniMax-H3`（只读）
- 当前推荐服务：TP=2 + Ulysses=1 + **SageAttention** + **Turbo-FP8（LoRA 已合并）**

> 说明：Comfy 的 `pruned_int8_convrot.safetensors` 是 Comfy 格式；SGLang 侧对应的是官方 `transformer-turbo-FP8-fl-ckpt850`（FP8 量化 + turbo LoRA）。

## 启动 / 停止

```bash
bash experiments/sglang_h3/launch_tp2.sh        # 用 luca 独立 venv
bash experiments/sglang_h3/launch_tp2.sh stop
```

## 12s 中文场景

```bash
# 8-step turbo（推荐）
experiments/sglang_h3/.venv/bin/python experiments/sglang_h3/run_cn_12s.py --steps 8 --duration 12 --seed 42
```

## 实测（2×5090，同一中文 12s，1344×768）

| 配置 | 墙钟 | 推理 | Denoise / Decode | 峰值显存 |
|--|--|--|--|--|
| 20-step | ~730s | ~720s | — | ~17.7GB |
| 8-step TurboFP8+Sage+TP2，`memory`+VAE layerwise | ~210s | ~201s | ~110s / ~83s | ~17.5GB |
| **8-step 同上，`speed` + VAE 常驻 + DiT layerwise** | **~140s** | **~134s** | **~112s / ~14s** | **~19GB** |

说明：2×32GB 上「全组件 GPU 常驻」会 OOM（DiT TP2 分片约 27GB/卡）。当前最快可跑路径是 **VAE 常驻、DiT/TE layerwise**（`launch_tp2.sh`）。

视频：`experiments/sglang_h3/output/cn_12s_seed42_steps8_speed_vae_res.mp4`  
对照：`exp_cn_accel_12s/13_sglang_tp2_turbo8_speed_vae_res_00001_.mp4`
