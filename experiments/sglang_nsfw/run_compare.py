#!/usr/bin/env python3
"""Run compare cases against a live SGLang server. Writes to exp_sglang_compare/<tag>/<mode>/."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/mnt/luca/H3_infer/ComfyUI-master_cp/output/exp_sglang_compare")
CASES = ROOT / "cases.json"


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


def run_case(base: str, tag: str, case: dict) -> dict:
    out_dir = ROOT / tag / case["mode"]
    dest = out_dir / f"{case['id']}.mp4"
    meta_path = out_dir / f"{case['id']}.json"
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"skip existing {dest}", flush=True)
        return {"id": case["id"], "tag": tag, "status": "skipped", "path": str(dest)}

    payload = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": case["prompt"],
        "task": case["task"],
        "target": case["target"],
        "conditions": case.get("conditions") or [],
        "num_outputs_per_prompt": 1,
        "num_inference_steps": int(case["num_inference_steps"]),
        "flow_shift": float(case["flow_shift"]),
        "audio_flow_shift": float(case["audio_flow_shift"]),
        "seed": int(case["seed"]),
    }
    print(
        f"POST {tag}/{case['id']} task={case['task']} tone={case['tone']} "
        f"steps={payload['num_inference_steps']}",
        flush=True,
    )
    t0 = time.time()
    try:
        resp = _post(f"{base}/v1/videos", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        rec = {
            "id": case["id"],
            "tag": tag,
            "status": "submit_failed",
            "http": e.code,
            "body": body[:4000],
            "elapsed_sec": round(time.time() - t0, 2),
            "request": payload,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        print(f"submit failed {e.code}: {body[:800]}", flush=True)
        return rec

    job = resp.get("id")
    if not job:
        rec = {
            "id": case["id"],
            "tag": tag,
            "status": "no_job",
            "response": resp,
            "elapsed_sec": round(time.time() - t0, 2),
            "request": payload,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        return rec

    while True:
        info = _get(f"{base}/v1/videos/{job}", timeout=120)
        status = (info.get("status") or "").lower()
        print(
            f"[{time.strftime('%H:%M:%S')}] {tag}/{case['id']} status={status} "
            f"progress={info.get('progress')}",
            flush=True,
        )
        if status in ("completed", "succeeded", "success"):
            save_video(base, job, info, dest)
            rec = {
                "id": case["id"],
                "tag": tag,
                "status": "ok",
                "elapsed_sec": round(time.time() - t0, 2),
                "path": str(dest),
                "bytes": dest.stat().st_size,
                "request": payload,
                "response": info,
            }
            meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
            print(f"saved {dest} ({dest.stat().st_size} bytes) {rec['elapsed_sec']}s", flush=True)
            return rec
        if status in ("failed", "error"):
            rec = {
                "id": case["id"],
                "tag": tag,
                "status": "failed",
                "elapsed_sec": round(time.time() - t0, 2),
                "request": payload,
                "response": info,
            }
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
            print(f"FAILED {case['id']}: {json.dumps(info, ensure_ascii=False)[:800]}", flush=True)
            return rec
        time.sleep(8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--server", required=True, choices=("fl2va", "ref2va"))
    ap.add_argument("--tag", required=True, choices=("nsfw", "official"))
    ap.add_argument("--wait-ready-sec", type=int, default=1200)
    args = ap.parse_args()

    blob = json.loads(CASES.read_text())
    cases = [c for c in blob["cases"] if c["server"] == args.server]
    if not cases:
        raise SystemExit(f"no cases for server={args.server}")

    wait_ready(args.base, args.wait_ready_sec)
    results = []
    for case in cases:
        try:
            results.append(run_case(args.base, args.tag, case))
        except Exception as e:
            rec = {"id": case["id"], "tag": args.tag, "status": "exception", "error": repr(e)}
            print(f"exception {case['id']}: {e}", flush=True)
            results.append(rec)
            p = ROOT / args.tag / case["mode"] / f"{case['id']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")

    summary = ROOT / f"summary_{args.tag}_{args.server}.json"
    ok = sum(1 for r in results if r.get("status") in {"ok", "skipped"})
    summary.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "server": args.server,
                "ok": ok,
                "n": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"done {args.tag}/{args.server} ok={ok}/{len(results)} {summary}", flush=True)
    if ok == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
