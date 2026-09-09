# H3 VAE TensorRT 实验（隔离，可整段回滚）

只在 `ComfyUI-master_cp` 里做**加法**，不改核心源码、不改 `extra_model_paths.yaml`、不改产线工作流、不碰 `/root/agnes_2_5`、不往 `.venv_sage_bench` 装包、不停 GPU1 `:8196`。

## 加了什么

| 路径 | 作用 | 回滚 |
|------|------|------|
| `ComfyUI-master_cp/custom_nodes/ComfyUI-H3VAE_TRT/` | 插件 + 本地 `.deps`（tensorrt/onnx） | 删整个目录 |
| `ComfyUI-master_cp/models/vae/minimax_h3_vae_{decoder,encoder}.onnx*` | fp16 ONNX（**没有** w4a16） | 删这些文件 |
| `ComfyUI-master_cp/models/vae/minimax_h3_vae_decoder.engine` | GPU0 编出来的 fp16 decoder | 删这个文件 |
| `experiments/h3_vae_trt/` | 本实验脚本 / 图 / 日志 | 可整目录删 |

产线 VAE safetensors 仍在 `/big_models/comfyui-minimax-H3/vae/`，没有被覆盖。

## 回滚

```bash
bash /mnt/luca/H3_infer/experiments/h3_vae_trt/rollback.sh
```

## 启动实验 Comfy（不要用产线端口）

GPU0 上已有 Agnes `:8198`（约 8GB）。完整 T2V 再开一份 DiT 可能 OOM。先跑 `bench_decode.py`，端到端另开端口且确认显存后再投。

```bash
source /mnt/luca/H3_infer/experiments/h3_vae_trt/env.sh
cd /mnt/luca/H3_infer/ComfyUI-master_cp
CUDA_VISIBLE_DEVICES=0 /mnt/luca/H3_infer/.venv_sage_bench/bin/python main.py \
  --listen 127.0.0.1 --port 8202 --cuda-device 0 \
  --extra-model-paths-config extra_model_paths.yaml
```

工作流：`experiments/h3_vae_trt/workflows/`（Sparge + Sage2 MemEff 稠密路径 + 4-step LoRA，只换视频 VAE decode）。

## Decode 微基准（GPU0，不加载 DiT）

随机 latent `(1,24,31,44,80)` ≈ 1280×704 / 5s：

| 后端 | 墙钟 | 峰值显存 |
|------|------|----------|
| 产线 PyTorch fp16 VAE | 10.55 s | 5.4 GB |
| TRT fp16 decoder | **6.34 s**（约 1.66×） | 6.4 GB |

TRT 输出帧数 107 vs PyTorch 103，时间轴拼 tile 和 Comfy 原生 VAE 不完全同一套；端到端画质还要在独立 Comfy 上对拍。GPU1 `:8196` 未动。
