# SLA 实验（隔离）

对比：**生产 Sparge(topk=0.5)+v1.0 LoRA** vs **LightX2V SLA(sparsity=0.85)+SLA LoRA**。

其余保持一致：1344×768、288 帧、4 step、TurboSampler、同一 seed。

## 权重

- HF：`lightx2v/Minimax-h3-Turbo-SLA`
- 本地：`experiments/sla/Minimax-h3-Turbo-SLA/..._sla_comfyui_bf16.safetensors`
- Comfy 发现：仅通过 `experiments/sla/extra_model_paths.yaml`（不写 `/big_models`）

## 启动

```bash
cd /mnt/luca/H3_infer
H3_SLA_PORT=8191 H3_SLA_GPU=0 ./experiments/sla/scripts/run_comfy.sh
```

## A/B

```bash
./.venv_sage_bench/bin/python -u experiments/sla/scripts/run_compare.py \
  --comfy http://127.0.0.1:8191 \
  --only o02 o03 o04
```

SLA attention = LightX2V `dynamic_sparse_attn`（`get_block_map` + sage2 sparse），**不是**完整 SparseLinear（无 linear 分支 / `proj_l`）。
