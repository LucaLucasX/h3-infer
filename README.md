# H3 Infer

MiniMax-H3 本地推理实验仓：ComfyUI 工作流、自定义节点、API 封装和加速实验。

不包含模型权重、ComfyUI 本体拷贝和 Python venv。权重仍按现有 `extra_model_paths.yaml` 从共享模型目录加载。

## 主要入口

- `h3_graph.py`：从 `GenerateParams` 生成 ComfyUI `/prompt` 图
- `api/`：t2v / i2v / r2v HTTP 服务
- `dev/sage_lora_sparge/`：生产向 Sage2 + Turbo LoRA + Sparge 启动脚本和节点
- `experiments/`：分辨率、量化、VAE、attention、lowvram 等对照实验
- `H3-DMD-SLA/`：蒸馏 / 量化相关脚本

## 注意

本仓库默认排除 `.venv*`、`ComfyUI-master_cp*`、`*.safetensors` 和 `**/output/`。克隆后需要自备 ComfyUI、Python 环境和 H3 权重。
