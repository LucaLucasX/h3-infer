#!/usr/bin/env python3
"""Full T2V 5s before/after VAE on the experimental Comfy (port 8202)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/luca/H3_infer")
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt  # noqa: E402

BASE = "http://127.0.0.1:8202"
WF = Path("/mnt/luca/H3_infer/experiments/h3_vae_trt/workflows")
OUT = Path("/mnt/luca/H3_infer/experiments/h3_vae_trt/logs")
COMFY_OUT = Path("/mnt/luca/H3_infer/ComfyUI-master_cp/output")
LOG = Path("/mnt/luca/H3_infer/experiments/h3_vae_trt/logs/comfy_8202.log")


def _graph(name: str) -> dict:
    return json.loads((WF / name).read_text(encoding="utf-8"))


def _executed_times() -> list[float]:
    if not LOG.exists():
        return []
    text = LOG.read_text(encoding="utf-8", errors="replace")
    return [float(x) for x in re.findall(r"Prompt executed in ([0-9.]+) seconds", text)]


def _vae_decode_times() -> list[tuple[float, str]]:
    if not LOG.exists():
        return []
    text = LOG.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"\[VAE_VIDEO_TIMING\] decode=([0-9.]+)s shape=(\([^)]+\))", text)


def _resolve_video(item: dict) -> Path | None:
    name = item.get("filename") or ""
    sub = item.get("subfolder") or ""
    if not name:
        return None
    p = COMFY_OUT / sub / name if sub else COMFY_OUT / name
    if p.exists() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".gif"}:
        return p
    return p if p.exists() and item.get("type") in ("videos", "video", "gifs", "images") else None


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,width,height,duration,avg_frame_rate",
        "-of", "json", str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    st = (json.loads(raw).get("streams") or [{}])[0]
    return {
        "width": st.get("width"),
        "height": st.get("height"),
        "nb_frames": st.get("nb_frames"),
        "duration": st.get("duration"),
        "avg_frame_rate": st.get("avg_frame_rate"),
        "path": str(path),
        "bytes": path.stat().st_size,
    }


def psnr_pair(a: Path, b: Path) -> dict:
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(a), "-i", str(b),
        "-lavfi", "[0:v][1:v]psnr=stats_file=-",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    err = proc.stderr
    avg = None
    m = re.search(r"average:([0-9.]+)", err)
    if m:
        avg = float(m.group(1))
    return {"psnr_avg": avg, "ffmpeg_rc": proc.returncode, "tail": err[-1500:]}


def run_one(label: str, graph: dict) -> dict:
    n_before = len(_executed_times())
    vae_before = len(_vae_decode_times())
    t0 = time.perf_counter()
    pid = submit_prompt(BASE, graph)
    print(f"[{label}] prompt_id={pid}", flush=True)
    hist = wait_prompt(BASE, pid, timeout=1800, poll_interval=2)
    elapsed = round(time.perf_counter() - t0, 2)
    st = (hist.get("status") or {}).get("status_str")
    err = None
    for msg in (hist.get("status") or {}).get("messages") or []:
        if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
            payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
            err = (payload.get("exception_message") or str(payload))[:4000]
    times = _executed_times()
    comfy_s = times[-1] if len(times) > n_before else None
    vae_times = _vae_decode_times()
    if len(vae_times) > vae_before:
        vae_decode_s = float(vae_times[-1][0])
        vae_shape = vae_times[-1][1]
    else:
        vae_decode_s, vae_shape = None, None
    outs = extract_outputs(hist)
    videos = []
    for item in outs:
        p = _resolve_video(item)
        if p is not None and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".gif"}:
            videos.append({"item": item, "probe": ffprobe(p)})
    return {
        "label": label,
        "status": "success" if st == "success" else f"comfy_{st}",
        "elapsed_sec": elapsed,
        "comfy_executed_sec": comfy_s,
        "vae_decode_sec": vae_decode_s,
        "vae_shape": vae_shape,
        "prompt_id": pid,
        "outputs": outs,
        "videos": videos,
        "error": err,
        "cached_nodes": [
            m[1].get("nodes")
            for m in (hist.get("status") or {}).get("messages") or []
            if isinstance(m, (list, tuple)) and m and m[0] == "execution_cached"
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = health(BASE)
    print("comfy ok", {k: stats.get(k) for k in ("system",) if k in stats} or "system_stats", flush=True)

    baseline = _graph("t2v_5s_sage_sparge_4step_baseline_api.json")
    trt = _graph("t2v_5s_sage_sparge_4step_trt_api.json")
    baseline["63"]["inputs"]["filename_prefix"] = "exp_h3_vae_trt/t2v5s_native"
    trt["63"]["inputs"]["filename_prefix"] = "exp_h3_vae_trt/t2v5s_trt"

    warmup_g = json.loads(json.dumps(baseline))
    warmup_g["63"]["inputs"]["filename_prefix"] = "exp_h3_vae_trt/t2v5s_warmup"

    print("=== warmup native VAE ===", flush=True)
    warm = run_one("warmup_native", warmup_g)
    print(json.dumps({k: warm.get(k) for k in ("status", "elapsed_sec", "comfy_executed_sec", "vae_decode_sec", "error")}, ensure_ascii=False), flush=True)
    if warm.get("status") != "success":
        (OUT / "e2e_t2v5s_vae.json").write_text(json.dumps({"warmup": warm}, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(f"warmup failed: {warm.get('error')}")

    print("=== timed native VAE ===", flush=True)
    native = run_one("native_vae", baseline)
    print(json.dumps({k: native.get(k) for k in ("status", "elapsed_sec", "comfy_executed_sec", "vae_decode_sec", "error")}, ensure_ascii=False), flush=True)

    print("=== timed TRT VAE ===", flush=True)
    trt_r = run_one("trt_vae", trt)
    print(json.dumps({k: trt_r.get(k) for k in ("status", "elapsed_sec", "comfy_executed_sec", "vae_decode_sec", "error")}, ensure_ascii=False), flush=True)

    compare = None
    nv = (native.get("videos") or [{}])[0].get("probe", {}).get("path")
    tv = (trt_r.get("videos") or [{}])[0].get("probe", {}).get("path")
    if nv and tv:
        compare = psnr_pair(Path(nv), Path(tv))
        print("psnr", compare.get("psnr_avg"), flush=True)

    payload = {
        "stack": "sparge0.5 + sage2_memeff_dense + 4step",
        "size": "1280x704 length=124",
        "warmup": warm,
        "native": native,
        "trt": trt_r,
        "psnr": compare,
    }
    out = OUT / "e2e_t2v5s_vae.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
