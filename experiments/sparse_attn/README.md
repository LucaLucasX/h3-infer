# SpargeAttn / LoSA 隔离实验

**硬规则：不改主流程默认、不污染 `.venv_sage_bench` site-packages、不占用 8188。**

## 布局

```
experiments/sparse_attn/
  vendor/SpargeAttn/     # 源码 + 原地编译 .so（PYTHONPATH 注入）
  custom_nodes/H3-SpargeAttn-Exp/  # 实验节点（import 失败则空注册）
  scripts/               # 启动 / 安装
  output/                # 实验产出（可删）
```

日常 8188 继续用 Sage2 MemEff。实验走 **:8190** + 可选 symlink 节点。

## 阶段

1. **SpargeAttn** — `spas_sage2_attn_meansim_topk_cuda(topk=…)` 替换 attention
2. **LoSA-style** — 早期步估 block mass → `block_sparse_sage2_attn_cuda` 冻结 mask

## 清理

```bash
./experiments/sparse_attn/scripts/teardown_exp.sh
```

卸 symlink、停 8190；vendor 可留可删。主代码与 8188 不受影响。
