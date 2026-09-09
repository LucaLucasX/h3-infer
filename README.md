# H3 Infer

MiniMax-H3 本地推理实验仓：ComfyUI 工作流、自定义节点、API 封装和加速实验。

两份 ComfyUI 源码树都不含 `models/`、`input/`、`output/`、TensorRT `.deps` 和 venv。权重按 `extra_model_paths.yaml` 从共享模型目录加载。

## 两份 ComfyUI

- `ComfyUI-master_cp/`：生产树
- `ComfyUI-master_cp_te_exp/`：TE（文本编码器 / Qwen3-VL vision）优化实验树。DiT 采样路径与生产一致；主要改 TE 驻留、causal SDPA、BF16 embedding 和 dynamic-VRAM 记账。开关：`COMFY_H3_VISION_RESIDENT=1|0`。说明见该目录 `H3_TE_DIT_EXPERIMENT.md`。

## 主要入口

- `h3_graph.py`：从 `GenerateParams` 生成 ComfyUI `/prompt` 图
- `api/`：t2v / i2v / r2v HTTP 服务
- `dev/sage_lora_sparge/`：生产向 Sage2 + Turbo LoRA + Sparge 启动脚本和节点
- `experiments/`：分辨率、量化、VAE、attention、lowvram 等对照实验
- `H3-DMD-SLA/`：蒸馏 / 量化相关脚本

## 注意

本仓库排除 `.venv*`、`*.safetensors`、`**/output/` 和模型权重。克隆后需要自备 Python 环境和 H3 权重。
