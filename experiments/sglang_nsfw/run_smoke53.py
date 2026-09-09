#!/usr/bin/env python3
"""NSFW FL2VA: first 3 of the usual 53 textcases (o01/o02/o03), 12s t2va."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path("/mnt/luca/H3_infer")
sys.path.insert(0, str(REPO / "report"))
from run_textcases_comfy4_tc_sparge import load_cases  # noqa: E402

OUT = REPO / "ComfyUI-master_cp" / "output" / "exp_sglang_nsfw_smoke53"


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def _get(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(base: str, timeout_s: int) -> None:
    t0 = time.time()
    while True:
        try:
            h = _get(f"{base}/health")
            print("health", h, flush=True)
            return
        except Exception as e:
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"server not ready: {e}")
            print(f"waiting server... ({e})", flush=True)
            time.sleep(5)


def save_video(base: str, job: str, info: dict, dest: Path) -> None:
    src = info.get("file_path") or info.get("url")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(src, str) and src.startswith("http"):
        urllib.request.urlretrieve(src, dest)
        return
    if isinstance(src, str) and Path(src).exists():
        shutil.copy2(src, dest)
        return
    urllib.request.urlretrieve(f"{base}/v1/videos/{job}/content", dest)


def run_case(base: str, case: dict, steps: int, duration: float, seed: int) -> dict:
    dest = OUT / f"{case['id']}_{_slug(case['title'])}.mp4"
    meta_path = dest.with_suffix(".json")
    payload = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": case["prompt"],
        "task": "t2va",
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": float(duration),
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": int(steps),
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": int(seed),
    }
    print(
        f"POST {case['id']} {case['title']} steps={steps} duration={duration}s",
        flush=True,
    )
    t0 = time.time()
    try:
        resp = _post(f"{base}/v1/videos", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        rec = {
            "id": case["id"],
            "title": case["title"],
            "status": "submit_failed",
            "http": e.code,
            "body": body[:4000],
            "elapsed_sec": round(time.time() - t0, 2),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        print(f"submit failed {e.code}: {body[:800]}", flush=True)
        return rec

    job = resp.get("id")
    if not job:
        rec = {
            "id": case["id"],
            "title": case["title"],
            "status": "no_job",
            "response": resp,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        return rec

    while True:
        info = _get(f"{base}/v1/videos/{job}", timeout=120)
        status = (info.get("status") or "").lower()
        print(
            f"[{time.strftime('%H:%M:%S')}] {case['id']} status={status} "
            f"progress={info.get('progress')}",
            flush=True,
        )
        if status in ("completed", "succeeded", "success"):
            save_video(base, job, info, dest)
            rec = {
                "id": case["id"],
                "title": case["title"],
                "status": "ok",
                "elapsed_sec": round(time.time() - t0, 2),
                "path": str(dest),
                "bytes": dest.stat().st_size,
            }
            meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
            print(f"saved {dest} ({dest.stat().st_size} bytes) {rec['elapsed_sec']}s", flush=True)
            return rec
        if status in ("failed", "error"):
            rec = {
                "id": case["id"],
                "title": case["title"],
                "status": "failed",
                "elapsed_sec": round(time.time() - t0, 2),
                "response": info,
            }
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
            print(f"FAILED {case['id']}: {json.dumps(info, ensure_ascii=False)[:800]}", flush=True)
            return rec
        time.sleep(8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30020")
    ap.add_argument("--wait-ready-sec", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    cases = load_cases()[: args.n]
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"smoke53 n={len(cases)} ids={[c['id'] for c in cases]} "
        f"titles={[c['title'] for c in cases]}",
        flush=True,
    )
    wait_ready(args.base, args.wait_ready_sec)
    results = [run_case(args.base, c, args.steps, args.duration, args.seed) for c in cases]
    summary = OUT / "summary.json"
    ok = sum(1 for r in results if r.get("status") == "ok")
    summary.write_text(
        json.dumps({"ok": ok, "n": len(results), "results": results}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print(f"done ok={ok}/{len(results)} {summary}", flush=True)
    if ok == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
