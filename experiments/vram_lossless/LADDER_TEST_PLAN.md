# GPU0 阶梯测试计划（seed=42）

## 策略档（4 档，无 L1-sparge/L1-full 重复命名）

| 档 | patch | mode | head_chunks |
|---|---|---|---|
| L0 | 关 | — | — |
| L1-token | 开 | step1 | 56（Sparge 不切 head） |
| L1 | 开 | step1 | 4 |
| L2 | 开 | step2 | 4 |

## 早停规则（省时间）

Case 按 **5 条单调链** 从易到难排列。同一档位、同一链内：

- 某一 case **OOM / error** → 该链后续 case 标 `skipped`，**不再提交**
- 不同链互不影响（例如单路 10s OOM 不影响 9图+3×3s 链）

链顺序见 `run_ladder.py` 的 `CASE_CHAINS`。

## L2 后探测

- 矩阵跑完后：若 `12s_9img_3video_15s` **已在 L2 矩阵成功** → **不再跑 P1**（3×15s 即上限）
- 若被早停跳过、前面 3×12s 已过 → 单独补跑 **P1**（同 9图+3×15s）
- **3×15s 成功即封顶**，不再往上探

## 执行

- GPU0，`ComfyUI-latest` `:8193`，`expandable_segments`，无 reserve
- **seed=42**（warmup + 全部 case）
- 换档前重启 Comfy（重新 import 节点）
- 日志：`logs/ladder_gpu0_seed42/{L0,L1-token,L1,L2,L2_P1}.json`

```bash
# 重启后单档
python experiments/vram_lossless/scripts/run_ladder.py --tier L0

# 全流程（L0→L1-token→L1→L2→按需 P1）
python experiments/vram_lossless/scripts/run_ladder.py --tier all
```

## 报告字段

每 case：`status` / `end_to_end_sec` / `step_0..3_sec` / `peak_gpu_mem_mb` / `skipped`+`skip_reason`

汇总：每档「最高通过 case」、相对 L0 / 上一档 ΔE2E、Δ单步。
