# VSA 实验（现有权重 + 纯稀疏）

把 Sparge 换成 VSA 粗阶段 Top-K + Sage2 block-sparse。**不用 FastH3 gate**（zero-init / 纯稀疏）。4-step LoRA 仍是产线 v1.0。

默认 keep 10% 视频–视频 tile（FastH3 的 90% sparse）。text/audio 按 VSA-H3 exempt：非视频 Q dense，非视频 K 全保留。

## 启动

```bash
cd /mnt/luca/H3_infer
H3_VSA_PORT=8197 H3_VSA_GPU=0 ./experiments/vsa/scripts/run_comfy.sh
```

## Case

```bash
./.venv_sage_bench/bin/python -u experiments/vsa/scripts/run_cases.py \
  --comfy http://127.0.0.1:8197 \
  --only o01 o02 o03 o04
```

成片：`ComfyUI-master_cp/output/exp_vsa_pure_sparse/`
