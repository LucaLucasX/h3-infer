#!/usr/bin/env python3
"""Real 100-request concurrency test for the H3 FastAPI service."""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8200"
INPUT = ROOT / "ComfyUI-master_cp" / "input"
TOTAL = 100
DURATIONS = (5, 10, 12)

ASSETS = {
    "i2v": [
        ("first.png", INPUT / "h3rb_c9c5722394db_kf_cliff_girl_first.png", "image/png"),
    ],
    "fl2v": [
        ("first.png", INPUT / "h3rb_c9c5722394db_kf_cliff_girl_first.png", "image/png"),
        ("last.png", INPUT / "h3rb_d05c2431d6d7_kf_cliff_girl_last.png", "image/png"),
    ],
    "r2v": [
        ("ref1.jpg", INPUT / "realref_full_01.png", "image/png"),
        ("ref2.jpg", INPUT / "realref_full_02.png", "image/png"),
        ("ref3.jpg", INPUT / "realref_full_03.png", "image/png"),
        ("audio.mp3", INPUT / "h3rb_ad6dc2547ff4_music_12s.mp3", "audio/mpeg"),
    ],
}


def request_spec(index: int) -> tuple[str, int]:
    mode = ("t2v", "i2v", "fl2v", "r2v")[index % 4]
    duration = DURATIONS[(index // 4) % len(DURATIONS)]
    return mode, duration


def make_files(mode: str) -> list[tuple[str, tuple[str, bytes, str]]]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for filename, path, content_type in ASSETS.get(mode, []):
        files.append(("images" if content_type.startswith("image/") else "audios",
                      (filename, path.read_bytes(), content_type)))
    return files


async def submit(client: httpx.AsyncClient, index: int) -> dict[str, Any]:
    mode, duration = request_spec(index)
    started = time.perf_counter()
    data = {
        "mode": mode,
        "duration": str(duration),
        "prompt": (
            f"Stress test request {index:03d}. Cinematic photorealistic scene, "
            "natural motion, detailed lighting, no text or logos."
        ),
        "seed": str(1000 + index),
    }
    try:
        response = await client.post(
            "/v1/generate",
            data=data,
            files=make_files(mode),
        )
        response.raise_for_status()
        body = response.json()
        return {
            "index": index,
            "mode": mode,
            "duration": duration,
            "submit_sec": round(time.perf_counter() - started, 3),
            **body,
        }
    except Exception as exc:
        return {
            "index": index,
            "mode": mode,
            "duration": duration,
            "submit_sec": round(time.perf_counter() - started, 3),
            "status": "submit_error",
            "error": str(exc),
        }


async def poll_all(
    client: httpx.AsyncClient, jobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pending = {job["job_id"]: job for job in jobs if "job_id" in job}
    done: list[dict[str, Any]] = [
        job for job in jobs if "job_id" not in job
    ]
    while pending:
        async def poll(job_id: str) -> tuple[str, dict[str, Any]]:
            try:
                response = await client.get(f"/v1/jobs/{job_id}")
                return job_id, response.json()
            except Exception as exc:
                return job_id, {"status": "poll_error", "error": str(exc)}

        results = await asyncio.gather(*(poll(job_id) for job_id in pending))
        for job_id, status in results:
            if status.get("status") in {"success", "error", "poll_error"}:
                done.append({**pending.pop(job_id), **status})
        print(f"completed={len(done)}/{len(jobs)} pending={len(pending)}", flush=True)
        if pending:
            await asyncio.sleep(10)
    return sorted(done, key=lambda item: item["index"])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["mode"], result["duration"])].append(result)

    summary: dict[str, Any] = {}
    for (mode, duration), rows in sorted(groups.items()):
        key = f"{mode}/{duration}s"
        successful = [row for row in rows if row.get("status") == "success"]
        summary[key] = {
            "requests": len(rows),
            "success": len(successful),
            "errors": len(rows) - len(successful),
            "avg_submit_sec": round(
                statistics.mean(row["submit_sec"] for row in rows), 3
            ),
            "avg_queue_sec": round(
                statistics.mean(row["queue_sec"] for row in successful), 3
            ) if successful else None,
            "avg_generation_sec": round(
                statistics.mean(row["generation_sec"] for row in successful), 3
            ) if successful else None,
            "avg_total_sec": round(
                statistics.mean(row["total_sec"] for row in successful), 3
            ) if successful else None,
        }
    return summary


async def main() -> None:
    limits = httpx.Limits(max_connections=TOTAL, max_keepalive_connections=TOTAL)
    timeout = httpx.Timeout(connect=30, read=30, write=120, pool=120)
    async with httpx.AsyncClient(base_url=API, limits=limits, timeout=timeout) as client:
        health = await client.get("/health")
        health.raise_for_status()
        started = time.perf_counter()
        submissions = await asyncio.gather(
            *(submit(client, index) for index in range(TOTAL))
        )
        print(f"submitted={len(submissions)} in {time.perf_counter() - started:.2f}s")
        results = await poll_all(client, submissions)

    payload = {
        "total_requests": TOTAL,
        "distribution": "4 modes x 3 durations, repeated across 100 requests",
        "api": API,
        "summary": summarize(results),
        "results": results,
    }
    output = ROOT / "api" / "logs" / "stress_100_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
