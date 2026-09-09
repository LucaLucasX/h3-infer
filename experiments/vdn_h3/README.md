# VDN-H3：8 步 T2V / Ref2V

节点：[ComfyUI-VDN-H3](https://github.com/Saganaki22/ComfyUI-VDN-H3)。在 MiniMax-H3 上把远处注意力换成 Video Delta Net 线性分支，checkpoint 自带 **8 步 turbo adapter**。

固定栈：

- 生成视频默认 **1280×704**（704p；720 短边对齐到 32）。可用 `--width` / `--height` 或 `--size` 改
- 采样 **8 步** `er_sde` + `beta`
- `ApplyVDNH3`：`stage-dmd-step-250`，`apply_turbo_adapter=on`，`lora_mode=merge`，`branch_weights=stream`
- **不要**再挂 Sage / Sparge / 4-step turbo LoRA（会抢同一条 `attn.forward`）

T2V 用 fl2va UNET，Ref2V 用 ref2va UNET，后面都接同一个 `ApplyVDNH3`。

## 路径

仓库根：`/mnt/luca/H3_infer`

| 用途 | 路径 |
|--|--|
| ComfyUI | `ComfyUI-master_cp/` |
| 启动 | `experiments/vdn_h3/run_comfy.sh` |
| Python | `.venv_sage_bench/` |
| VDN 节点 | `ComfyUI-master_cp/custom_nodes/ComfyUI-VDN-H3/` |
| VDN 权重 | `/big_models/comfyui-minimax-H3/vdn/stage-dmd-step-250/` |
| H3 UNET / CLIP / VAE | `/big_models/comfyui-minimax-H3/` |
| T2V 脚本 | `experiments/vdn_h3/scripts/run_t2v_cases.py` |
| Ref2V 脚本 | `experiments/vdn_h3/scripts/run_r2v_cases.py` |
| 成片 | `ComfyUI-master_cp/output/exp_vdn_h3/` |
| 日志 | `experiments/vdn_h3/logs/` |

权重必须保持官方目录名（节点按 `stage-dmd-step-250` 查找）：

```
/big_models/comfyui-minimax-H3/vdn/stage-dmd-step-250/
  model_spec.json
  linear_branch/model.safetensors
  adapters/
```

节点会在 **loras 目录的上一级** 下找 `vdn/`。当前 `extra_model_paths.yaml` 里 `minimax_h3.loras` 指向 `/big_models/comfyui-minimax-H3/loras`，因此能扫到上面这条路径。也可以放到 `ComfyUI-master_cp/models/vdn/`。

缺权重：

```bash
hf download OpenVDN/vdn-minimax-h3 stage-dmd-step-250 \
  --local-dir /big_models/comfyui-minimax-H3/vdn
```

## 启动

```bash
cd /mnt/luca/H3_infer
bash experiments/vdn_h3/run_comfy.sh
# GPU0  http://127.0.0.1:8190
# 日志  experiments/vdn_h3/logs/comfyui_8190.log
```

等健康检查通过：

```bash
curl -sf http://127.0.0.1:8190/system_stats >/dev/null && echo READY
```

换卡 / 换端口：

```bash
H3_VDN_GPU=1 H3_VDN_PORT=8192 bash experiments/vdn_h3/run_comfy.sh
```

停：

```bash
bash experiments/vdn_h3/run_comfy.sh stop
```

改了 `comfy_extras/nodes_minimax_h3.py` 之后必须重启 Comfy。`--cache-classic` 开着：同一条 prompt 第二次会极快，**不能当耗时**。换 seed 或重启再测。没有单独 warmup，第一条会偏冷。

脚本自己会把仓库根加进 `sys.path`，在仓库根用下面这个 Python 跑即可：

```bash
cd /mnt/luca/H3_infer
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py --help
```

## 生成视频分辨率（最终 MP4）

这是 **生成出来的视频宽高**，和参考图/参考视频怎么缩是两回事。

默认 **1280×704**。宽高必须是 32 的倍数；`--height 720` 会自动收到 704。

```bash
# 默认 704p
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py --comfy http://127.0.0.1:8190

# 改成 768p
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py --size 768p

# 自己写宽高
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_r2v_cases.py --only 8img_2vid --width 1376 --height 768
```

`--size` 可用：`704p`（1280×704）、`768p`（1376×768）、或 `1280x704` 这种 `宽x高`。写了 `--size` 就覆盖 `--width` / `--height`。

## 参考图 / 参考视频怎么缩（节点内部）

DiT 的 video patch 是 **2×2**，VAE 空间 16×，所以参考进 VAE 的宽高必须是 **32 的倍数**，否则采样会在 `patchify_video` 炸掉（例如短边 720 → latent 高 45，奇数）。

参考尺寸**不跟生成视频走**，短边钉死后再把长边 round 到 32：

| 输入 | 策略 |
|--|--|
| 参考图 `match`（默认） | 短边 **704**，长边按比例 round 到 32 |
| 参考图 `max` | 短边 **2048**，长边 round 到 32 |
| 参考视频 | 短边 **544**，长边 round 到 32（960×540 → **960×544**） |
| 首尾帧 / AddGuide | 先按图短边 704 缩，再 cover 到生成视频画布 |

2816×1536 的参考图会收到 **1280×704**；1080p 参考视频短边钉 544。

## 测 T2V

UNET：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`

**指定 case = 你自己的 prompt 或文件路径**，不是 `o01` 这种内置编号。

```bash
cd /mnt/luca/H3_infer

# 手写提示词
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py \
  --prompt "海边悬崖上最后一朵花在风中摇曳，电影感" \
  --name flower --duration 5

# 一个 txt（整份文件就是 prompt）
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py \
  --case text-cases/official-27/海边悬崖的最后一朵花-15s.txt

# 多个文件，或一个目录（目录里所有 .txt）
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py \
  --case /path/to/my_prompt.txt /path/to/another.txt

# json：{"prompt": "...", "title": "..."} 或这种对象的列表
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_t2v_cases.py \
  --case /path/to/case.json
```

什么都不写时才会跑内置 catalog 的 o01/o02/o03（12s）。`--only o07` 是从那份 catalog 里挑，一般不用。`--list` 能看 catalog id。

链路：

```
UNETLoader(fl2va) → ApplyVDNH3(stage-dmd-step-250, turbo=on)
  → MiniMaxH3TextToVideo → 8 步 er_sde/beta
```

本机 5090 32GB、12s @ 1376×768、未开 `--kj-lowvram`：o01 318s / o02 302s / o03 316s（平均约 312s）。  
日志：`experiments/vdn_h3/logs/t2v_12s_768_o01_o03.json`

那组是改默认分辨率之前的 768p 对照。现在不写 `--size` 就是 1280×704。

## 测 Ref2V

UNET：`minimax_h3_ref2va_pruned_int8_convrot.safetensors`

和 T2V 同一套 VDN 补丁与采样，只换 ref2va 基座 + `MiniMaxH3ReferenceToVideo`。生成视频默认同样是 **1280×704**。

```bash
cd /mnt/luca/H3_infer

# 手写 prompt + 自己的图/视频（绝对路径或 Comfy input/ 里的文件名都行）
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_r2v_cases.py \
  --prompt "Cinematic scene matching <Picture 1> <Video 1>." \
  --images /path/to/person.png \
  --videos /path/to/clip.mp4 \
  --name my_r2v --duration 5

# 一个文件夹：里面放图、视频，可选 prompt.txt
.venv_sage_bench/bin/python experiments/vdn_h3/scripts/run_r2v_cases.py \
  --case /path/to/my_r2v_case
```

不写 `--prompt` / `--case` / `--images` / `--videos` 时才跑内置冒烟（`input/` 里 8 图+2 视频）。`--only 8img_2vid` 是挑那条冒烟，不是指定你的 case。

链路：

```
UNETLoader(ref2va) → ApplyVDNH3(stage-dmd-step-250, turbo=on)
  → MiniMaxH3ReferenceToVideo → 8 步 er_sde/beta
```

本机 5090、5s @ 1280×704、8 图+2 视频、未开 `--kj-lowvram`：556s。  
成片：`ComfyUI-master_cp/output/exp_vdn_h3/ref2va_8img_2vid_5s_00001_.mp4`  
日志：`experiments/vdn_h3/logs/r2v_<duration>s_<宽>x<高>.json`

## UI 手跑

T2V 示例：`ComfyUI-master_cp/custom_nodes/ComfyUI-VDN-H3/example_workflows/vdn_h3_t2v_8step.json`

Ref2V 没有上游示例图。API 构图导出：`experiments/vdn_h3/workflows/vdn_h3_r2v_8step_api.json`

加载对应 UNET → `Apply VDN-H3`（checkpoint `stage-dmd-step-250`，turbo 打开）→ 条件节点 → 8 步 `er_sde`+`beta`。

## 常见问题

- **`VDN checkpoint ... not found`**：确认 `stage-dmd-step-250/linear_branch/model.safetensors` 在 loras 上一级的 `vdn/` 下。
- **block 数对不上**：VDN stage 必须配 MiniMax-H3 同深度基座（本机 T2V 用 pruned int8 fl2va，Ref2V 用 pruned int8 ref2va）。
- **`shape '[1, 24, …]' is invalid`**：参考图/视频宽高不是 32 倍数。节点已把图短边钉 704、视频短边钉 544；改节点后要重启 Comfy。
- **8 步动作怪**：turbo adapter 必须开；关了要用 ~50 步。
- **OOM**：加 `--kj-lowvram`，或缩短时长。
- **看起来像没开 VDN**：太短的片段窗口会盖住全部时间轴，会退回稠密注意力。脚本已开 `verbose`，日志里应有 `[vdn] layout:`。
