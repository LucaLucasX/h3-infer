# H3 FastAPI 服务

这个服务封装当前生产配置，支持三种模式：

- `t2v`：只传文字
- `i2v`：上传 1 张首帧
- `fl2v`：上传 2 张首帧+尾帧（`i2v` 的首尾帧变体，底层实现相同）
- `r2v`：最多 9 张参考图 + 最多 3 个独立音频
- 明确不接受视频输入
- `t2v`/`i2v` 使用 `lightx2v_fl2v` LoRA
- `r2v` 使用 `lightx2v_ref2v` LoRA
- `mode` 和 `duration`（秒）都是必填参数
- Sage2 MemEff + Sparge topk 0.5
- 默认 1344×768、288 帧、24 FPS（约 12 秒）
- 服务异步返回任务 ID，避免 HTTP 请求长时间阻塞
- 上传和 ComfyUI 提交不会阻塞 FastAPI 事件循环

## 启动

先启动 ComfyUI（GPU0 示例）：

```bash
cd /mnt/luca/H3_infer
H3_SPARGE_PORT=8190 H3_SPARGE_GPU=0 \
  ./dev/sage_lora_sparge/scripts/run_comfy.sh
```

另开终端启动 API：

```bash
cd /mnt/luca/H3_infer
H3_COMFY_URL=http://127.0.0.1:8190 \
  ./.venv_sage_bench/bin/python -m uvicorn \
  api.h3_r2v_service:app --host 0.0.0.0 --port 8200
```

如果只想直接启动模块：

```bash
./.venv_sage_bench/bin/python api/h3_r2v_service.py
```

## 调用

健康检查：

```bash
curl http://127.0.0.1:8200/health
```

### t2v：只传文字

```bash
curl -X POST http://127.0.0.1:8200/v1/generate \
  -F 'mode=t2v' \
  -F 'duration=12' \
  -F 'prompt=Cinematic rainy Tokyo street, a woman walks toward camera, no text.'
```

### i2v/fl2v：首帧或首帧+尾帧

`i2v` 上传 1 张图片作为首帧；`fl2v` 上传 2 张图片，按顺序解释为首帧和尾帧。

```bash
curl -X POST http://127.0.0.1:8200/v1/generate \
  -F 'mode=i2v' \
  -F 'duration=12' \
  -F 'prompt=The woman walks forward and opens the umbrella, preserve both keyframes.' \
  -F 'images=@/path/to/first.png' \
  -F 'images=@/path/to/last.png'
```

### r2v：多图+多音频

```bash
curl -X POST http://127.0.0.1:8200/v1/generate \
  -F 'mode=r2v' \
  -F 'duration=12' \
  -F 'prompt=Using the references to keep the same character identity. Cinematic 12-second video, natural motion, ambient sound, no text.' \
  -F 'images=@/path/to/ref_01.jpg' \
  -F 'images=@/path/to/ref_02.jpg' \
  -F 'images=@/path/to/ref_03.jpg' \
  -F 'audios=@/path/to/voice.wav' \
  -F 'seed=42'
```

`mode` 是必填字段，不传会返回 HTTP 422。可选值为 `t2v`、`i2v`、`fl2v`、`r2v`。
首尾帧必须显式填写 `mode=fl2v`。

响应会返回 `job_id` 和 `status_url`，然后轮询：

```bash
curl http://127.0.0.1:8200/v1/jobs/JOB_ID
```

完成后响应的 `outputs[].url` 就是 ComfyUI 视频下载地址：

```bash
curl -L 'http://127.0.0.1:8190/view?...' -o result.mp4
```

也可以直接打开自动生成的 Swagger：

```text
http://127.0.0.1:8200/docs
```

`duration` 使用秒，服务按 24 FPS 转换，并自动对齐到 H3 的 `17k+5` 帧约束。

## 并发说明

服务可以在**单个 FastAPI worker**内并发接收上传和提交多个任务；ComfyUI
负责实际 GPU 队列。不要直接使用 `--workers 4`，因为当前任务索引保存在
进程内存中，多 worker 之间不会共享 job 状态。需要多 worker 或多 API 实例时，
应把 `JOBS` 替换为 Redis/数据库，并为不同实例配置独立的任务存储。
