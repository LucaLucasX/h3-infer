# MiniMax H3 单卡无损显存扩容实验报告

> 目标：在 **不改基线图 / 不改 `h3_graph.py` / 不换 Sage3 / 不压参考画质** 的前提下，把 RTX 5090 32GB 上「长参考视频」的 OOM 悬崖往后推，并标清每一档的容量收益和墙钟代价。
>
> 日期：2026-08-28（capacity 消融）；**2026-08-31 阶梯测试终报见 [`LADDER_REPORT.md`](LADDER_REPORT.md)**。所有正式对照均为 **12s 输出 @ 1376×768、24 FPS、4-step turbo LoRA、Sage2 MemEff、Sparge topk=0.5、INT8 ConvRot UNET**。

## 结论摘要

- 基线 12s 安全区：无参考视频（含 9 图）；单路参考视频 ≤8s；9 图 + 单路 ≤5s。单路 ≥10s、9 图+8s 在 TE 后、采样前 OOM。
- L1 不是一块开关。实验顺序就是三档叠上去的：**先切 Linear（推不动 10s）→ 后来才开 Sparge 分 head（10s/9图+8s 才过）→ 再后来才预分配（3×5s 才过）**。当前 `nodes.py` 把这三档绑在一起，但请求级可以拆。
- **L1-full 上限（阶梯 seed=42）**：单路 8–15s；9图+8s/15s；9图+3×3s/5s；**9图+2×8s**（653s）。卡在 **9图+2×10s**、**9图+3×8s**。
- **L2**（不物化整段融合 QKV）相对 L1 再慢 ~1.6–1.8×；阶梯在 13/18 时终止（9图+2×12s 已 ~24min），**不再推进**。残差卸 CPU（约 7×）不做。
- **2026-08-31 阶梯测试（seed=42，GPU0）**：详见 [`LADDER_REPORT.md`](LADDER_REPORT.md)；完整 L1/L2 数据已并入下文主结果表。
- 节点 `MiniMaxH3LosslessMemPatch` 接在 Turbo LoRA 之后。图里没有这个节点，前向与基线相同。

## 统一配置与计时口径

| 项 | 值 |
|---|---|
| GPU | RTX 5090 32GB（实验在 ComfyUI-latest；基线对照为 `dev/test/H3_COMPLETE_BENCHMARK_REPORT.md`） |
| 启动 | `--preview-method none --cache-classic`，**NORMAL_VRAM**，无 `--gpu-only` / `--highvram` |
| 最终推荐启动多的唯一项 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 模型 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| LoRA | `lightx2v_fl2v/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`，strength 1.0 |
| 注意力 | `sage_v2_memeff` + Sparge topk 0.5（**不是 Sage3**） |
| 输出 | 1376×768，294 帧 / 12.25s |
| E2E | 脚本提交 `/prompt` 到 history 成功的墙钟 |
| 单步 | Comfy 日志 `H3TURBO step N ... elapsed=` |
| 峰值 | 采样期间 `nvidia-smi memory.used` 最大值（含权重） |

**峰值不可跨「是否 reserve-vram」直接比。** 早期 GPU1 部分成功 case 开过 `--reserve-vram 3 --vram-headroom 3`，smi 读数会被压到约 25–27GB，但激活空间被饿死，同样的 10s/9图+8s 在只加 reserve、没有 head-chunk 时反而 OOM。后面的 L1-full / L2 主结果均 **无 reserve**。

## 根因（为什么砍分辨率解决不了）

H3 packed 序列是 `[text | 参考块 | audio | video]`。参考 VAE latent **每个采样 step 都重新注入**，和生成 token 做全程 self-attention。峰值在：

| 张量 | 量级（长序列） | 备注 |
|---|---|---|
| MLP `fc1` 激活 | 随 S 线性 | INT8 per-row，沿 token 切 **bit-identical**（禁止尾巴长度 1，避免 Kitchen `m==1` kernel） |
| 融合 QKV `[S, 3HD]` | 3×5s+9 图约 8.93GB；2×15s+9 图约 13GB | 物化后再 split |
| Sparge `v_transposed` | 56 head 约 2.1–2.5GB | 按 head 切 **无损**（head 独立） |
| `list` + `torch.cat` | 再一份完整输出 | 3×5s 直接炸在这里 |

沿 **token** 切 attention 会改连通性，不是无损。LoRA 是 `BypassForwardHook` 打在 Linear 上，不能用裸 `F.linear` 切权重绕过模块。

---

## 分级：方法、代价、是否进默认

从便宜到贵。容量收益不与时间代价同序。

### L0 — 基线原图（对照）

什么都不加。sage2 + Sparge 0.5 + 4-step turbo。

- **容量**：见下一节基线悬崖。
- **代价**：0。
- **用途**：无参考视频、单路 ≤8s、9 图+单路 ≤5s。

### L0+ — 分配器 `expandable_segments`

环境变量，不是算法。

- **容量**：减碎片，**单独推不动** 10s / 9图+8s 悬崖。
- **代价**：测不出。
- **用途**：L1/L2 实验服默认带上；对基线无害。

### 反例 — `--reserve-vram` / `--vram-headroom`

- **容量**：把激活空间饿死。`capacity_gpu1_headroom3.json`：10s、12s、9图+8s、15s **全部 OOM**。
- **代价**：无速度收益。
- **结论**：扩容实验 **不要** 预留 3+3GB。

### L1 三档（按实验加上去的顺序，可以拆开）

不是一次做完的。先有 token 切块， Sparge 分 head 是后面才开的，预分配更晚。

| 子档 | 何时加上 | 挡住什么 | 单独解锁 | 时间 |
|---|---|---|---|---|
| **L1-token** | 最先 | MLP/`qkv_proj` 整段激活 | **推不动** 10s / 9图+8s | 几乎 0（12s+3s：166s vs 基线 161s） |
| **L1-sparge** | 其后 | Sparge `v_transposed` ~2.1–2.5GB | 单路 10/12/15s、9图+8s | **L1 里唯一的主税** |
| **L1-prealloc** | 再其后 | `torch.cat` 再要一份输出（3×5s 曾 8.93GB） | 9图+3路5s | ~0，应随切块常开 |

#### L1-token — 沿 token 切 Linear/MLP

`chunk_rows=4096`。当时结果仍 `list+cat`，Sparge 仍 56 head 一次。`capacity_gpu1_chunk_only.json`：3s ✅；**10s / 12s / 15s / 9图+8s 仍 OOM**。

INT8 per-row，切 token 维 bit-identical（禁尾巴长度 1）。短序列 `S≤4096` 等于没切。

#### L1-sparge — Sparge 按 head 组切（`head_chunks=4` → 14 组）

**这才是把基线 8s 墙推开的那一档。** Attention 仍全局，QKV 当时仍整段物化。Head 独立，无损。

早期 GPU1（当时还带着 reserve，峰值读数偏低）：10s 359s / 12s 418s / 15s 446s / 9图+8s 393s，全部从 OOM 变成成功。去掉 reserve 后 9图+8s 仍成功（409s / 31.6GB / 单步 ~70s）。

代价：同样 FLOP，Sparge kernel 1→14，GPU ~100%。基线能跑的 12s+3s：单步 30s → 34s（**+13%**）。这档税出在这里，不在 token 切块。

#### L1-prealloc — 预分配，禁止 `cat` 双缓冲

L1-token + L1-sparge 之后，9图+3路5s 仍炸在 `chunked_call` 的 `torch.cat`（`stress_3vid_9img.json`）。改成预分配写入后同 case 成功（587s / 31.1GB）。kernel 次数不变，数学不变。只要开了切块就该带上，不必单独当「慢档」。

#### L1-full — 三档绑在一起（当前 `nodes.py`）

磁盘上的 step1-only = L1-token + L1-sparge + L1-prealloc。attn：token 切 `qkv_proj` → 整段 QKV → RoPE → head-chunk Sparge → token 切 `out_proj`。

- **容量**：单路 10/12/15s；9图+8s；9图+15s；9图+3路5s。**9图+2路15s 仍 OOM**。
- **代价**：几乎全是 L1-sparge 的税。9图+8s 单步约 70s vs 基线无图 8s 的 53s（约 +32%，内含 9 图加长序列）。
- **动态**：过悬崖至少开 L1-sparge（外加几乎免费的 token+prealloc）；不必对短请求开 Sparge 14 组。

### L2 — 不物化整段融合 QKV（实验档，当前代码已关掉）

按 head 组：全宽 `qkv_proj`（LoRA hook 仍触发）→ 只留 4/56 head → RoPE → Sparge → 写入预分配 attn → 最后一次 `out_proj`。

- **容量**：峰值只留 4/56 的 QKV。解锁 **9图+2路15s**（L1-full 在 Sparge `k.mean` 再要 127MB 时炸，已分配 ~29.7GiB）。
- **代价**：为保 INT8+LoRA，**不是** 1×FLOP 窄 GEMM，而是全宽 QKV ×14。相对 L1-full 约 **1.5× E2E / 1.8× 单步**。GPU 仍 ~100%。
- **结论**：只给 2×15s 这类极端请求。不要当默认。

### L3 — 残差 / QKV 卸 CPU（否决）

- **容量**：理论最大。
- **代价**：warmup 12s+3s 单步从约 50s 到 **340–360s（约 7×）**，GPU 利用率约 30%，PCIe 搬运。
- **结论**：默认不做。

---

## 基线（L0）

来源：`dev/test/H3_COMPLETE_BENCHMARK_REPORT.md`。12s 输出。

### 单路参考视频

| Case | 状态 | E2E(s) | 单步(s) | 峰值(MB) |
|---|---|---:|---|---:|
| 12s + 1 路 3s | ✅ | 160.6 | 30.2 / 30.4 / 30.5 / 30.5 | 31875 |
| 12s + 1 路 5s | ✅ | 230.3 | 40.3 / 38.8 / 38.4 / 38.5 | 31809 |
| 12s + 1 路 8s | ✅ | 328.6 | 54.8 / 53.1 / 53.2 / 53.2 | 31875 |
| 12s + 1 路 10s | ❌ OOM | 76.2 | — | 32097 |
| 12s + 1 路 12s | ❌ OOM | 94.2 | — | 31957 |
| 12s + 1 路 15s | ❌ OOM（输入另有 temporal truncation） | 102.2 | — | 31957 |

### 9 图 + 1 路视频

| Case | 状态 | E2E(s) | 单步(s) | 峰值(MB) |
|---|---|---:|---|---:|
| 12s + 9图 + 3s | ✅ | 196.3 | 38.8 / 39.0 / 39.1 / 39.2 | 32097 |
| 12s + 9图 + 5s | ✅ | 292.5 | 49.6 / 47.9 / 47.9 / 47.8 | 31073 |
| 12s + 9图 + 8s | ❌ OOM（TE 后、采样前） | 92.3 | — | 31553 |

无参考视频的 12s 多图多音频矩阵 21/21 成功，不在本实验扩容范围内。

---

## 消融：同一批会 OOM 的 case

均为 12s 输出，实验图注入 node 22。早期 GPU1，部分带 reserve（见上）。

| 配置 | 3s | 10s | 12s | 15s | 9图+8s |
|---|---|---|---|---|---|
| 只 L1-token | ✅ 166s / 32.1GB | ❌ | ❌ | ❌ | ❌ |
| L1-token + reserve 3+3 | — | ❌ | ❌ | ❌ | ❌ |
| L1-token + **L1-sparge**（早期，带 reserve） | ✅ 196s / 24.0GB | ✅ 359s / 26.4GB | ✅ 418s / 26.4GB | ✅ 446s / 26.0GB | ✅ 393s / 25.0GB |

读法：只切 Linear 救不了 10s；**加上 Sparge 分 head 之后** 基线 OOM 的单路 10–15s 和 9图+8s 才全部出片。3×5s 要再等到 L1-prealloc。后续主测去掉 reserve，峰值回到 ~31GB。

---

## L1-full 主结果（无 reserve；三档都开）

实现：L1-token + L1-sparge + L1-prealloc（`mode=step1`, `head_chunks=4`）。

### 早期 capacity 实验

日志：`capacity_gpu0_step1.json`、`capacity_gpu1_step1_9img_8s.json`、`capacity_gpu0_2x15s.json`。

| Case | 基线 | L1-full | E2E(s) | 单步(s) | 峰值(MB) |
|---|---|---|---:|---|---:|
| warmup 12s+3s | ✅ ~161s / ~30s | ✅ | 208–214 | 34.7 / 34.4 / 33.3 / 33.8 | 26583–27545 |
| 12s + 9图 + 8s | ❌ OOM | ✅ | 409.2 | 70.5 / 70.1 / 70.5 / 70.1 | 31617 |
| 12s + 9图 + 1 路 15s | 基线未测；单路 15s 已 OOM | ✅ | 540.8 | ~91 | 31969 |
| 12s + 9图 + 3 路 5s | 未进基线矩阵；L1c 前 OOM | ✅ | 586.9 | ~100 | 31873 |
| 12s + 9图 + 2 路 15s | — | ❌ OOM | 260.2 | — | 31457 |

9图+8s 对照：

- 相对基线 **同输出、无图、8s**（能跑的最近视频对照）：E2E 409 vs 329（+24%），单步 70 vs 53（+32%）。
- 相对基线 **9图+5s**（能跑的最近 9 图对照）：序列更长，不能单看成 patch 税。

3×5s 在 **L1-prealloc 之前**（当时已有 token + Sparge 分 head）：E2E 154s 即 OOM，峰值 29971MB，栈在 `torch.cat`。加上预分配后 587s 跑完。

### 阶梯测试 L1（seed=42，GPU0，2026-08-31）

日志：`logs/ladder_gpu0_seed42/L1.json`、`run_all.log`。  
成片：`ComfyUI-latest/output/exp_vram_lossless/ladder/L1/results/`。

| Case | 基线 | L1 | E2E(s) | 单步(s) | 峰值(MB) |
|---|---|---|---:|---|---:|
| 12s + 1 路 3s | ✅ | ✅ | 8.2† | 33.7 / 32.4 / 32.6 / 33.7 | 13153 |
| 12s + 1 路 8s | ❌ OOM | ✅ | 316.7 | 56.2 / 55.3 / 55.6 / 55.5 | 31233 |
| 12s + 1 路 8s_b | ❌ OOM | ✅ | 318.5 | 55.0 / 55.6 / 55.4 / 55.7 | 29921 |
| 12s + 1 路 10s | ❌ OOM | ✅ | 356.9 | 61.9 / 62.5 / 62.6 / 62.4 | 30979 |
| 12s + 1 路 12s | ❌ OOM | ✅ | 417.8 | 73.3 / 73.4 / 73.8 / 73.3 | 31169 |
| 12s + 1 路 15s | ❌ OOM | ✅ | 442.7 | 77.0 / 76.9 / 77.6 / 78.8 | 32097 |
| 12s + 9图 + 8s | ❌ OOM | ✅ | 391.3 | 67.2 / 66.4 / 66.6 / 67.3 | 31617 |
| 12s + 9图 + 15s | ❌ OOM | ✅ | 531.1 | 92.7 / 90.8 / 91.1 / 90.8 | 32101 |
| 12s + 9图 + 3×3s | ❌ OOM | ✅ | 380.9 | 62.7 / 62.6 / 63.7 / 62.5 | 30819 |
| 12s + 9图 + 3×5s | ❌ OOM | ✅ | 581.2 | 100.1 / 100.0 / 99.7 / 99.7 | 32099 |
| **12s + 9图 + 2×8s** | ❌ OOM | ✅ | **653.1** | **115.6 / 114.4 / 115.2 / 113.7** | **30403** |
| 12s + 9图 + 2×10s | — | ❌ OOM | 198.2 | — | 32099 |
| 12s + 9图 + 3×8s | ❌ OOM | ❌ OOM | 244.2 | — | 31299 |

† warmup 后全缓存，E2E 不代表真实采样成本。

**L1 阶梯上限**：9图+2×8s（E2E ~11min）。再往上 2×10s / 3×8s 仍 OOM，需 L2（但 L2 墙钟不可接受，见下节）。

单路 8s 对照：基线 E2E 329s / 单步 53s → L1 阶梯 317s / 56s（约 **+5%**）。

---

## L2 主结果（无 reserve；相对 L1-full 同 case）

实现：L1-full + 按 head 组流水线，全宽 QKV ×14（`mode=step2`, `head_chunks=4`）。

### 早期 capacity 实验

日志：`capacity_gpu0_step2.json`、`comfyui_8193.log`。

| Case | L1-full E2E / 峰值 | L2 E2E / 峰值 | L2 单步(s) | L2 / L1-full |
|---|---|---|---|---:|
| warmup 12s+3s | 214s / 26.6GB | 350.7s / 31.0GB | 71.8 / 69.2 / 69.7 / 70.8 | ~1.6×（短序列也被 ×14） |
| 9图 + 1 路 15s | 540.8s / 31.2GB | 816.8s / 30.7GB | 162.4 / 161.7 / 161.9 / 161.8 | E2E 1.51×，单步 ~1.78× |
| 9图 + 3 路 5s | 586.9s / 31.1GB | 885.4s / 30.9GB | 174.3 / 175.9 / 176.0 / 175.9 | E2E 1.51×，单步 ~1.75× |
| 9图 + 2 路 15s | ❌ OOM | ✅ **1450.1s / 31.4GB** | 288.6 / 289.6 / 289.4 / 289.4 | 新容量档 |

### 阶梯测试 L2（seed=42，GPU0；13/18 时终止）

日志：`logs/ladder_gpu0_seed42/run_all.log`。  
成片：`ComfyUI-latest/output/exp_vram_lossless/ladder/L2/results/`。

| Case | L1 E2E | L2 E2E | L2 单步(s) | L2/L1 | 峰值(MB) |
|---|---:|---:|---|---:|---:|
| 12s + 1 路 8s | 316.7 | 570.3 | 118.1 / 119.9 / 117.3 / 118.6 | 1.80× | 32101 |
| 12s + 9图 + 8s | 391.3 | 672.3 | 136.5 / 136.7 / 137.0 / 137.8 | 1.72× | 32005 |
| 12s + 9图 + 15s | 531.1 | 874.4 | 176.2 / 176.4 / 177.3 / 178.0 | 1.65× | 32101 |
| 12s + 9图 + 3×5s | 581.2 | 938.5 | 188.7 / 191.3 / 188.8 / 189.8 | 1.61× | 31941 |
| **12s + 9图 + 2×8s** | 653.1 | **1114.7** | **231.8 / 228.4 / 226.5 / 226.7** | **1.71×** | **32101** |
| 12s + 9图 + 2×10s | ❌ OOM | 1208.6 | 242.5 / 245.0 / 246.4 / 247.8 | — | 32037 |
| 12s + 9图 + 2×12s | ❌ OOM | 1436.7 | 293.3 / 291.4 / 293.6 / 292.4 | — | 30949 |

L2 相对 L1 稳定 **~1.6–1.8×**；9图+2×12s 已 **~24min E2E**，用户终止后续 case。**不再推进 L2**。

L2 峰值并不明显低于 L1-full（仍贴 31GB），换来的是 **融合 QKV 不再和 Sparge 工作区叠满**。早期 capacity 实验里 2×15s 是 L2 唯一必须开的理由，但墙钟 ~24min 不可接受。

---

## 时间 / 容量性价比排序

**容量（谁真的移动了 OOM 悬崖）**

1. **L1-sparge** — 10s / 12s / 15s / 9图+8s（L1 里真正搬家的那档）
2. **L1-prealloc** — 9图+3路5s（白捡）
3. **L1-token** — 单独不够；开切块时一起带着
4. L2 去融合 QKV — 仅 9图+2路15s
5. expandable_segments — 辅助
6. L3 CPU — 否决
7. reserve-vram — 负收益

**时间（谁在变慢）**

1. L1-prealloc、expandable_segments — ~0
2. L1-token — 小；`S≤4096` 走原路径
3. **L1-sparge** — L1 主税（+13% ~ +32%）
4. L2 全宽 QKV×14 — 再 ×1.5–1.8
5. L3 — ×7

默认：未过悬崖全关；过 10s/9图+8s 开 L1-sparge（token+prealloc 顺带开，几乎免费）；3 路长参考确保 prealloc；2×15s 再开 L2。

---

## 按请求动态开关

节点按 **每次 `/prompt` 构图** 决定，不是进程启动写死。不改 `h3_graph.py`：`build_graph()` 之后调用 `inject_lossless_mem` / `should_inject_lossless_mem`（`experiments/vram_lossless/scripts/inject_mem_patch.py`）。

| 请求（12s 输出） | 基线 | 开哪一档 |
|---|---|---|
| 无参考视频（含 9 图） | 不 OOM | 全关 |
| 1 路 ≤8s、无 9 图 | ✅ ~31.8GB | 全关 |
| 1 路 ≥10s | OOM | **L1-sparge**（+ token/prealloc） |
| 9 图 + 1 路 ≤5s | ✅ | 全关 |
| 9 图 + 1 路 ≥8s | OOM | **L1-sparge**（+ token/prealloc） |
| 9 图 + 3 路 5s | L1-sparge 仍可能 cat OOM | 再加 **L1-prealloc** |
| 9 图 + 2 路 8s | OOM | **L1-full**（阶梯 653s / ~11min） |
| 9 图 + 2 路 ≥10s | L1-full 仍 OOM | L2 能过但墙钟不可接受（阶梯 2×12s ~24min，**否决**） |

`head_chunks=56`（或关掉 head 循环）= 只有 L1-token/prealloc，等价于实验里救不了 10s 的那档。`head_chunks=4` 才是 L1-sparge。当前节点把三档写死在一起；要按请求拆，图参数里分开关即可。

---

## 代码入口

| 路径 | 作用 |
|---|---|
| `experiments/vram_lossless/custom_nodes/ComfyUI-H3-LosslessMem/nodes.py` | 当前 **L1-full**（token + sparge head=4 + 预分配） |
| `.../chunk_util.py` | 禁尾巴长度 1；预分配 `chunked_call` |
| `experiments/vram_lossless/scripts/inject_mem_patch.py` | 注入 node 22；`should_inject_lossless_mem` |
| `experiments/vram_lossless/scripts/run_capacity.py` | 容量脚本；`--no-mem-patch` 为 L0 对照 |
| `experiments/vram_lossless/run_comfy.sh` | 实验服启动（**不要**对正在跑的对照服端口执行 stop） |

硬约束：节点必须在 `MiniMaxH3TurboLoRA` **之后**。基线图不加 node 22 则完全不变。

## 原始数据

- **阶梯测试终报**：`LADDER_REPORT.md`；数据 `logs/ladder_gpu0_seed42/`（L1 全量 + L2 部分）
- 基线：`dev/test/H3_COMPLETE_BENCHMARK_REPORT.md`
- L1-token：`logs/capacity_gpu1_chunk_only.json`
- L1-sparge：`logs/capacity_gpu1_headchunk.json`、`logs/capacity_gpu1_headchunk_rest.json`
- L1-prealloc / L1-full：`logs/capacity_gpu0_step1.json`、`logs/capacity_gpu1_step1_9img_8s.json`、`logs/stress_3vid_9img.json`（prealloc 前 3×5s OOM）、`logs/capacity_gpu0_2x15s.json`（L1-full 下 2×15s OOM）
- L2：`logs/capacity_gpu0_step2.json`
- 成片：`ComfyUI-latest/output/exp_vram_lossless/results/`

## 已知限制

- L2 为保 INT8 + LoRA hook，QKV GEMM 重复 14 次；若以后能按输出通道切 hooked Linear 且数值等价，L2 税应下降。
- 15s 参考在对照配置侧已有 temporal truncation；本实验解的是 **显存**，不是截断策略。
- 未做像素级 vs 基线 bit-exact 回归（INT8 切块与 head 独立在原理上等价；L2 全宽重复 GEMM 在数学上应与一次全宽再切片相同，但浮点累加顺序可能有 ulp 差）。
- 5s 输出矩阵基线已能过绝大多数视频组合，不是本实验主战场。
