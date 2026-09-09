# H3 官方手册·文生视频 Cases

共 9 个可直接调用的纯文生视频 case。

## 时长上限

全部可调用任务的目标时长不超过 12 秒。原始时长超过 12 秒的 Prompt 时间轴已按比例压缩，`cases.json`、`manifest.tsv` 与 `tasklist_all.list` 已同步；变更清单见 `duration-cap-12s.tsv`。

## Prompt 版本

根目录的 `H3官方-*.txt` 已优化为 H3-Context-IR 三段式：

1. `integrated_multimodal_description`
2. `overall_soundscape`
3. `non_diegetic_music`

优化版本：`h3-context-ir-v2-20260814`。原始 Prompt 逐字保存在 `prompts-raw/`，可随时回退或对照。

```bash
./run_cases.sh 30010
```

如果系统 `python3` 没有 `requests`，指定项目虚拟环境：

```bash
PY=/path/to/venv/bin/python3 ./run_cases.sh 30010
```

单例调用：

```bash
python3 ../../fl2va.py --prompt-file 'H3官方-007-海螺_视频_15秒、16_9の横_538390013432676361.txt' \
  --duration 15 --aspect-ratio 16:9 --port 30010
```

- `manifest.tsv`：批处理元数据
- `cases.json`：机器可读全量定义
- `tasklist_all.list`：默认任务清单
- `examples/`：官方生成示例
