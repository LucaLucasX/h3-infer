# HappyHorse 文生视频 · MiniMax H3 调用版

这里包含 17 个可直接交给 MiniMax H3 `fl2va.py` 的纯文生视频任务。根目录的 `HappyHorse-*.txt` 已转换为 H3-Context-IR 三段式，原始 HappyHorse Prompt 保存在 `prompts-raw/`。

全部可调用任务的目标时长不超过 12 秒。原始时长超过 12 秒的 Prompt 时间轴已按比例压缩，相关元数据与任务清单已同步；变更清单见 `duration-cap-12s.tsv`。

## 直接运行

```bash
./run_cases.sh 30010
```

如果系统 `python3` 没有 `requests`，请指定项目虚拟环境：

```bash
PY=/path/to/venv/bin/python3 ./run_cases.sh 30010
```

单个任务示例：

```bash
python3 ../../fl2va.py \
  --prompt-file HappyHorse-t2v-10-01.txt \
  --duration 15 --aspect-ratio 16:9 --port 30010
```

## 文件说明

- `HappyHorse-*.txt`：H3-Context-IR 三段式调用 Prompt。
- `prompts-raw/`：逐字保留的原始 HappyHorse Prompt。
- `examples/<task>/`：HappyHorse 参考成片，使用硬链接避免重复占用空间。
- `cases.json`：全部 MiniMax H3 任务及来源映射。
- `manifest.tsv`：适合表格查看的任务索引。
- `tasklist_all.list`：`run_cases.sh` 默认批处理清单。
- `t2v-1.0/`、`t2v-1.1/`、`cases.jsonl`：保留的原始资料与 DashScope 请求。
- `manifest-source.tsv`：整理前的原始索引。

调用会产生模型费用。建议先用 `fl2va.py --dry-run` 检查单个任务。
