#!/usr/bin/env python3
"""FastAPI service for production H3 image+audio reference-to-video generation."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "dev" / "sage_lora_sparge"
sys.path[:0] = [str(ROOT), str(DEV), str(ROOT / "report")]

import _bootstrap_h3_graph  # noqa: F401  # type: ignore[reportMissingImports]
from comfy_client import extract_outputs, get_history, health, submit_prompt, view_url
from h3_graph import GenerateParams, build_graph
from loras import LORA_R2V, LORA_T2V  # type: ignore[reportMissingImports]


COMFY_URL = os.getenv("H3_COMFY_URL", "http://127.0.0.1:8190").rstrip("/")
INPUT_DIR = ROOT / "ComfyUI-master_cp" / "input"
MAX_IMAGES = 9
MAX_AUDIOS = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

app = FastAPI(
    title="MiniMax H3 Generation API",
    version="1.0.0",
    description=(
        "Production H3 API for t2v, i2v/fl2v and r2v. Video inputs are unsupported."
    ),
)

# The API process is intentionally stateless apart from this small prompt map.
# ComfyUI remains the source of truth for queue/history/output data.
JOBS: dict[str, dict[str, Any]] = {}


def _duration_to_length(duration: float, fps: float = 24.0) -> int:
    """Convert seconds to the nearest H3-compatible 17k+5 frame count."""
    if duration <= 0 or duration > 120:
        raise HTTPException(status_code=400, detail="duration 必须在 0 到 120 秒之间")
    requested = max(5, round(duration * fps))
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * k + 5)]
    if k > 0:
        candidates.append(17 * (k - 1) + 5)
    candidates.append(17 * (k + 1) + 5)
    return min(candidates, key=lambda value: abs(value - requested))


def _timing(job: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    submitted = job.get("submitted_at")
    result: dict[str, Any] = {
        "submitted_at": submitted,
        "queue_sec": (
            round(max(0.0, time.time() - submitted), 3)
            if submitted is not None
            else None
        ),
        "generation_sec": None,
        "total_sec": (
            round(max(0.0, time.time() - submitted), 3)
            if submitted is not None
            else None
        ),
    }
    messages = (entry or {}).get("status", {}).get("messages", [])
    timestamps = {
        message[0]: message[1].get("timestamp") / 1000.0
        for message in messages
        if isinstance(message, list)
        and len(message) > 1
        and isinstance(message[1], dict)
        and isinstance(message[1].get("timestamp"), (int, float))
    }
    started = timestamps.get("execution_start")
    completed = timestamps.get("execution_success")
    if started is not None:
        result["started_at"] = started
        if submitted is not None:
            result["queue_sec"] = round(max(0.0, started - submitted), 3)
    if completed is not None:
        result["completed_at"] = completed
        if started is not None:
            result["generation_sec"] = round(max(0.0, completed - started), 3)
        if submitted is not None:
            result["total_sec"] = round(max(0.0, completed - submitted), 3)
    return result


def _validate_upload(upload: UploadFile, allowed: set[str], kind: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的{kind}格式: {suffix or '(无扩展名)'}",
        )
    return suffix


async def _save_upload(upload: UploadFile, allowed: set[str], kind: str, job_id: str, index: int) -> str:
    suffix = _validate_upload(upload, allowed, kind)
    filename = f"api_{job_id}_{kind}_{index:02d}{suffix}"
    path = INPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    def copy_upload() -> None:
        # Keep large media out of the event loop and avoid loading it all into RAM.
        upload.file.seek(0)
        with path.open("wb") as output:
            shutil.copyfileobj(upload.file, output, length=1024 * 1024)

    await asyncio.to_thread(copy_upload)
    return filename


def _graph(
    *,
    mode: str,
    prompt: str,
    image_names: list[str],
    audio_names: list[str],
    seed: int,
    length: int,
    width: int,
    height: int,
    job_id: str,
) -> dict[str, dict]:
    is_r2v = mode == "r2v"
    params = GenerateParams(
        mode=mode,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        fps=24.0,
        steps=4,
        seed=seed,
        turbo=True,
        turbo_lora=LORA_R2V if is_r2v else LORA_T2V,
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=True,
        sparge_topk=0.5,
        sparge_dense_first_steps=0,
        sparge_num_layers=50,
        sparge_audio_dense=True,
        image=image_names[0] if mode == "i2v" else "",
        last_frame=image_names[1] if mode == "i2v" and len(image_names) > 1 else None,
        ref_images=image_names if is_r2v else [],
        ref_videos=[],
        ref_audios=audio_names if is_r2v else [],
        output_prefix=f"api_r2v/{job_id}",
    )
    return build_graph(params)


@app.get("/health")
def service_health() -> dict[str, Any]:
    try:
        stats = health(COMFY_URL)
        return {"status": "ok", "comfy": COMFY_URL, "comfy_system_stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ComfyUI 不可用: {exc}") from exc


@app.post("/v1/generate")
async def generate(
    mode: str = Form(...),
    prompt: str = Form(..., min_length=1),
    images: list[UploadFile] | None = File(default=None),
    audios: list[UploadFile] | None = File(default=None),
    seed: int = Form(default=42),
    duration: float = Form(...),
    width: int = Form(default=1376),
    height: int = Form(default=768),
) -> dict[str, Any]:
    image_uploads = images or []
    audio_uploads = audios or []
    requested_mode = mode
    graph_mode = "i2v" if mode == "fl2v" else mode
    if graph_mode not in {"t2v", "i2v", "r2v"}:
        raise HTTPException(status_code=400, detail="mode 必须是 t2v、i2v/fl2v 或 r2v")
    if graph_mode == "t2v" and (image_uploads or audio_uploads):
        raise HTTPException(status_code=400, detail="t2v 不接受图片或音频输入")
    if graph_mode == "i2v" and len(image_uploads) not in (1, 2):
        raise HTTPException(status_code=400, detail="i2v/fl2v 需要 1 张首帧，或 2 张首帧+尾帧")
    if graph_mode == "i2v" and audio_uploads:
        raise HTTPException(status_code=400, detail="i2v/fl2v 当前不接受独立音频参考")
    if graph_mode == "r2v" and not image_uploads:
        raise HTTPException(status_code=400, detail="r2v 至少需要 1 张参考图片")
    if graph_mode == "r2v" and len(image_uploads) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"r2v 最多上传 {MAX_IMAGES} 张图片")
    if graph_mode == "r2v" and len(audio_uploads) > MAX_AUDIOS:
        raise HTTPException(status_code=400, detail=f"r2v 最多上传 {MAX_AUDIOS} 个音频")
    length = _duration_to_length(duration)
    if width % 32 or height % 32:
        raise HTTPException(status_code=400, detail="width 和 height 必须是 32 的倍数")

    job_id = uuid.uuid4().hex
    image_names = [
        await _save_upload(upload, IMAGE_EXTS, "image", job_id, i)
        for i, upload in enumerate(image_uploads, 1)
    ]
    audio_names = [
        await _save_upload(upload, AUDIO_EXTS, "audio", job_id, i)
        for i, upload in enumerate(audio_uploads, 1)
    ]
    try:
        submitted_at = time.time()
        prompt_id = await asyncio.to_thread(
            submit_prompt,
            COMFY_URL,
            _graph(
                mode=graph_mode,
                prompt=prompt,
                image_names=image_names,
                audio_names=audio_names,
                seed=seed,
                length=length,
                width=width,
                height=height,
                job_id=job_id,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"提交 ComfyUI 失败: {exc}") from exc

    JOBS[job_id] = {
        "job_id": job_id,
        "prompt_id": prompt_id,
        "mode": requested_mode,
        "graph_mode": graph_mode,
        "comfy": COMFY_URL,
        "images": len(image_names),
        "audios": len(audio_names),
        "duration_sec": duration,
        "length_frames": length,
        "steps": 4,
        "lora": LORA_R2V if graph_mode == "r2v" else LORA_T2V,
        "submitted_at": submitted_at,
    }
    return {
        **JOBS[job_id],
        "status": "queued",
        "status_url": f"/v1/jobs/{job_id}",
    }


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id 不存在（服务重启后内存任务记录会丢失）")
    entry = get_history(job["comfy"], job["prompt_id"])
    if not entry:
        return {**job, "status": "queued", **_timing(job, None)}

    status_obj = entry.get("status") or {}
    status = status_obj.get("status_str") or "running"
    result: dict[str, Any] = {**job, "status": status, **_timing(job, entry)}
    if status == "success":
        outputs = extract_outputs(entry)
        result["outputs"] = [
            {
                **item,
                "url": view_url(
                    job["comfy"],
                    item["filename"],
                    item.get("subfolder", ""),
                ),
            }
            for item in outputs
        ]
    elif status == "error":
        result["messages"] = status_obj.get("messages", [])
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.h3_r2v_service:app",
        host=os.getenv("H3_API_HOST", "0.0.0.0"),
        port=int(os.getenv("H3_API_PORT", "8200")),
        reload=False,
    )
