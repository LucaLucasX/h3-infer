#!/usr/bin/env python3
"""Submit the CN 12s Bund dialogue scene to local SGLang MiniMax-H3 (TP2)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from benchmark_cn_accel_12s import CN_VOICE_TEXT  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30010")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wait-ready-sec", type=int, default=900)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t_ready0 = time.time()
    while True:
        try:
            h = _get(f"{args.base}/health")
            print("health", h, flush=True)
            break
        except Exception as e:
            if time.time() - t_ready0 > args.wait_ready_sec:
                raise SystemExit(f"server not ready: {e}")
            print(f"waiting server... ({e})", flush=True)
            time.sleep(5)

    payload = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": CN_VOICE_TEXT,
        "task": "t2va",
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": float(args.duration),
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": int(args.steps),
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": int(args.seed),
    }
    print(
        f"POST {args.base}/v1/videos steps={args.steps} duration={args.duration}s",
        flush=True,
    )
    t0 = time.time()
    try:
        resp = _post(f"{args.base}/v1/videos", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise SystemExit(f"submit failed {e.code}: {body[:2000]}")
    print(json.dumps(resp, ensure_ascii=False, indent=2)[:2000], flush=True)
    job = resp.get("id")
    if not job:
        raise SystemExit(f"no job id: {resp}")

    while True:
        info = _get(f"{args.base}/v1/videos/{job}")
        status = (info.get("status") or "").lower()
        print(
            f"[{time.strftime('%H:%M:%S')}] status={status} progress={info.get('progress')}",
            flush=True,
        )
        if status in ("completed", "succeeded", "success"):
            out = OUT / f"cn_12s_seed{args.seed}_steps{args.steps}.mp4"
            meta = OUT / f"cn_12s_seed{args.seed}_steps{args.steps}.json"
            src = info.get("file_path") or info.get("url")
            if isinstance(src, str) and src.startswith("http"):
                urllib.request.urlretrieve(src, out)
            elif isinstance(src, str) and Path(src).exists():
                shutil.copy2(src, out)
            else:
                try:
                    urllib.request.urlretrieve(f"{args.base}/v1/videos/{job}/content", out)
                except Exception as e:
                    raise SystemExit(f"cannot fetch video: {e}; info={info}")
            elapsed = round(time.time() - t0, 2)
            meta.write_text(
                json.dumps(
                    {"elapsed_sec": elapsed, "request": payload, "response": info},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            print(f"saved {out} ({out.stat().st_size} bytes) elapsed={elapsed}s", flush=True)
            return
        if status in ("failed", "error"):
            raise SystemExit(json.dumps(info, ensure_ascii=False, indent=2))
        time.sleep(10)


if __name__ == "__main__":
    main()
