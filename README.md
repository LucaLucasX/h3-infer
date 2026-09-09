# H3 Infer

MiniMax-H3 本地推理实验仓：ComfyUI 工作流、自定义节点、API 封装和加速实验。

`ComfyUI-master_cp/` 是当前使用的 ComfyUI 源码树（含自定义节点）。不包含 `models/`、`input/`、`output/`、TensorRT `.deps` 和 Python venv。权重仍按 `extra_model_paths.yaml` 从共享模型目录加载。

`ComfyUI-master_cp_te_exp/` 是重复实验拷贝，未入库。

## 主要入口

- `ComfyUI-master_cp/`：ComfyUI 本体 + H3 相关 custom nodes
- `h3_graph.py`：从 `GenerateParams` 生成 ComfyUI `/prompt` 图
- `api/`：t2v / i2v / r2v HTTP 服务
- `dev/sage_lora_sparge/`：生产向 Sage2 + Turbo LoRA + Sparge 启动脚本和节点
- `experiments/`：分辨率、量化、VAE、attention、lowvram 等对照实验
- `H3-DMD-SLA/`：蒸馏 / 量化相关脚本

## 注意

本仓库排除 `.venv*`、`*.safetensors`、`**/output/` 和模型权重。克隆后需要自备 Python 环境和 H3 权重。
