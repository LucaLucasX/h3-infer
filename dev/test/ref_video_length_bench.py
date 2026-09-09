#!/usr/bin/env python3
"""Measure H3 ref2v behavior for reference-video length and multiplicity."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "dev" / "sage_lora_sparge"
INPUT = ROOT / "ComfyUI-master_cp" / "input"
sys.path[:0] = [str(ROOT), str(DEV), str(ROOT / "report")]

import _bootstrap_h3_graph  # noqa: F401
from comfy_client import extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from loras import LORA_R2V

WIDTH, HEIGHT, FPS, STEPS = 1376, 768, 24.0, 4
UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
VIDEO_NAMES = {
    3: "h3_test_video_3s.mp4",
    5: "h3_test_video_5s.mp4",
    8: "h3_test_video_8s.mp4",
    10: "h3_test_video_10s.mp4",
    12: "h3_test_video_12s.mp4",
    15: "h3_test_video_15s.mp4",
}
IMAGES = [
    "realref_bust_01.jpg",
    "realref_bust_02.jpg",
    "realref_bust_03.jpg",
    "realref_face_01.jpg",
    "realref_face_02.jpg",
    "realref_face_03.jpg",
    "realref_full_01.png",
    "realref_full_02.png",
    "realref_full_03.png",
]
STEP_RE = re.compile(r"\[H3TURBO step (\d+)\].*?elapsed=([0-9.]+)s")
REF_VIDEO_RE = re.compile(
    r"\[H3_REF_VIDEO\] name=(\S+) input_frames=(\d+) resize=(\d+x\d+) "
    r"target_frames=(\d+) encoded_frames=(\d+) truncated=(True|False) qwen_frames=(\d+)"
)
TE_RE = re.compile(r"\[H3_TE_TIMING\] encode=([0-9.]+)s")
VAE_RE = re.compile(r"\[(VAE_AUDIO_TIMING|VAE_VIDEO_TIMING)\] ([^\n]+)")
PROMPT_RE = re.compile(r"Prompt executed in ([0-9.]+) seconds")
SAVE_RE = re.compile(r"\[SAVE_VIDEO_TIMING\] ([^\n]+)")


def align_length(seconds: int) -> tuple[int, float]:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    length = min(candidates, key=lambda value: abs(value - requested))
    return length, length / FPS


def ffprobe(path: Path) -> dict:
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames,"
         "sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(raw.stdout)
    video = next(s for s in data["streams"] if s.get("codec_type") == "video")
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    return {
        "file": path.name,
        "width": video.get("width"), "height": video.get("height"),
        "fps": video.get("avg_frame_rate"), "frames": video.get("nb_frames"),
        "duration_sec": round(float(data["format"]["duration"]), 3),
        "audio_sample_rate": audio.get("sample_rate") if audio else None,
        "audio_channels": audio.get("channels") if audio else None,
    }


def gpu_mem(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def parse_log(text: str) -> dict:
    refs = [
        {
            "name": m.group(1), "input_frames": int(m.group(2)),
            "resized": m.group(3), "target_frames": int(m.group(4)),
            "encoded_frames": int(m.group(5)), "truncated": m.group(6) == "True",
            "qwen_frames": int(m.group(7)),
        }
        for m in REF_VIDEO_RE.finditer(text)
    ]
    steps = [
        {"step": int(m.group(1)), "duration_sec": float(m.group(2))}
        for m in STEP_RE.finditer(text)
    ]
    te = TE_RE.findall(text)
    return {
        "ref_video_processing": refs,
        "te_encode_sec": float(te[-1]) if te else None,
        "steps": steps,
        "vae_timings": [{"kind": m.group(1), "details": m.group(2)} for m in VAE_RE.finditer(text)],
        "prompt_execution_sec": float(PROMPT_RE.findall(text)[-1]) if PROMPT_RE.findall(text) else None,
        "save_timings": SAVE_RE.findall(text),
    }


def cases(seconds: int) -> list[dict]:
    if seconds == 5:
        return [
            {"label": "5s_1video_3s", "durations": [3]},
            {"label": "5s_1video_5s", "durations": [5]},
            {"label": "5s_2video_3s", "durations": [3, 3]},
            {"label": "5s_2video_5s", "durations": [5, 5]},
            {"label": "5s_3video_3s", "durations": [3, 3, 3]},
            {"label": "5s_3video_5s", "durations": [5, 5, 5]},
        ]
    return [{"label": f"12s_1video_{d}s", "durations": [d]} for d in (3, 5, 8, 10, 12, 15)]


def extreme_cases(seconds: int) -> list[dict]:
    return [
        {
            "label": f"{seconds}s_9images_1video_{d}s",
            "durations": [d],
            "ref_images": IMAGES,
        }
        for d in (3, 5, 8)
    ]


def graph_for(case: dict, length: int, seed: int, prefix: str) -> dict:
    videos = []
    counters = {3: 0, 5: 0}
    for d in case["durations"]:
        filename = VIDEO_NAMES[d]
        if d in counters and counters[d]:
            filename = filename.replace(".mp4", f"_{'bc'[counters[d]-1]}.mp4")
        counters[d] = counters.get(d, 0) + 1
        videos.append(filename)
    return build_graph(GenerateParams(
        mode="r2v",
        prompt="Cinematic photorealistic forest scene with a woman walking naturally, no text or logos.",
        width=WIDTH, height=HEIGHT, length=length, fps=FPS, steps=STEPS,
        seed=seed, turbo=True, turbo_lora=LORA_R2V, turbo_strength=1.0,
        attention_backend="sage_v2_memeff", sparge=True, sparge_topk=0.5,
        sparge_dense_first_steps=0, sparge_num_layers=50, sparge_audio_dense=True,
        unet_name=UNET, ref_images=case.get("ref_images", []), ref_videos=videos, ref_audios=[],
        output_prefix=prefix,
    ))


def run_one(base: str, log_path: Path, gpu: int, case: dict, length: int,
            actual_duration: float, seed: int, warmup: bool) -> dict:
    offset = log_path.stat().st_size if log_path.exists() else 0
    started = time.perf_counter()
    peak = {"mb": gpu_mem(gpu)}
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(1.0):
            peak["mb"] = max(peak["mb"], gpu_mem(gpu))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        prefix = f"ref_video_length_{'warmup' if warmup else 'results'}/{case['label']}"
        pid = submit_prompt(base, graph_for(case, length, seed, prefix))
        entry = wait_prompt(base, pid, timeout=3600, poll_interval=2)
        result = {
            "case": case["label"], "status": (entry.get("status") or {}).get("status_str"),
            "output_requested_duration_sec": case["output_seconds"],
            "output_actual_duration_sec": actual_duration, "output_frames": length,
            "input_videos": [ffprobe(INPUT / VIDEO_NAMES[d]) for d in case["durations"]],
            "input_duration_spec_sec": case["durations"],
            "warmup": warmup, "seed": seed,
            "end_to_end_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"], "prompt_id": pid,
            "outputs": extract_outputs(entry),
        }
        result.update(parse_log(read_log(log_path, offset)))
        return result
    except Exception as exc:
        return {"case": case["label"], "status": "error", "error": str(exc)[:4000]}
    finally:
        stop.set()
        watcher.join(timeout=2)


def read_log(path: Path, offset: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, choices=(5, 12), required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--extreme", action="store_true",
                    help="run 9 reference images plus one reference video")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    health(base)
    length, actual = align_length(args.seconds)
    selected = extreme_cases(args.seconds) if args.extreme else cases(args.seconds)
    # Warmup the ref2v model/LoRA once before measuring cases.
    warmup = run_one(
        base, args.log, args.gpu,
        {
            "label": f"{args.seconds}s_extreme_warmup" if args.extreme else f"{args.seconds}s_warmup",
            "durations": [3],
            "ref_images": IMAGES if args.extreme else [],
            "output_seconds": args.seconds,
        },
        length, actual, 990000, True,
    )
    results = []
    for i, case in enumerate(selected):
        case["output_seconds"] = args.seconds
        row = run_one(base, args.log, args.gpu, case, length, actual, 991000 + i, False)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        results.append(row)
    suffix = "_extreme_9images" if args.extreme else ""
    output = DEV / "logs" / f"ref_video_length_{args.seconds}s{suffix}.json"
    output.write_text(json.dumps({
        "config": {
            "gpu": args.gpu, "port": args.port, "resolution": [WIDTH, HEIGHT],
            "steps": STEPS, "attention": "sage_v2_memeff", "sparge_topk": 0.5,
            "unet": UNET, "lora": LORA_R2V, "warmup_once": True,
            "output_requested_seconds": args.seconds, "output_frames": length,
            "output_actual_seconds": actual, "video_max_count": 3,
            "extreme_case": args.extreme, "reference_image_count": 9 if args.extreme else 0,
        },
        "warmup": warmup, "results": results,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output}", flush=True)
    return 0 if all(r.get("status") == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
