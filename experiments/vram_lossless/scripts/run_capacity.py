#!/usr/bin/env python3
"""Capacity probe for lossless token-chunking on the production H3 stack.

Uses the same sage + Sparge topk=0.5 + 4-step ref2v turbo LoRA graph as the
benchmark, plus opt-in MiniMaxH3LosslessMemPatch. Does not modify h3_graph.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "vram_lossless"
DEV = ROOT / "dev" / "sage_lora_sparge"
sys.path[:0] = [str(ROOT), str(DEV), str(ROOT / "report"), str(EXP / "scripts")]

import _bootstrap_h3_graph  # noqa: F401
from comfy_client import extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from inject_mem_patch import inject_lossless_mem
from loras import LORA_R2V

WIDTH, HEIGHT, FPS, STEPS = 1376, 768, 24.0, 4
UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
VIDEO_SUFFIX = ("", "_b", "_c")
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


def video_name(duration: int, slot: int = 0) -> str:
    suffix = VIDEO_SUFFIX[slot] if slot < len(VIDEO_SUFFIX) else ""
    preferred = f"h3_test_video_{duration}s{suffix}.mp4"
    fallback = f"h3_test_video_{duration}s.mp4"
    latest = ROOT / "ComfyUI-master_cp" / "input"
    if (latest / preferred).exists():
        return preferred
    return fallback


def align_length(seconds: int) -> tuple[int, float]:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    length = min(candidates, key=lambda value: abs(value - requested))
    return length, length / FPS


def gpu_mem(gpu: int) -> int:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def resolve_videos(case: dict) -> list[str]:
    if case.get("videos"):
        return list(case["videos"])
    return [video_name(d, i) for i, d in enumerate(case["durations"])]


def graph_for(case: dict, length: int, seed: int, prefix: str, chunk_rows: int | None) -> dict:
    videos = resolve_videos(case)
    g = build_graph(GenerateParams(
        mode="r2v",
        prompt="Cinematic photorealistic forest scene with a woman walking naturally, no text or logos.",
        width=WIDTH, height=HEIGHT, length=length, fps=FPS, steps=STEPS,
        seed=seed, turbo=True, turbo_lora=LORA_R2V, turbo_strength=1.0,
        attention_backend="sage_v2_memeff", sparge=True, sparge_topk=0.5,
        sparge_dense_first_steps=0, sparge_num_layers=50, sparge_audio_dense=True,
        unet_name=UNET, ref_images=case.get("ref_images", []), ref_videos=videos,
        ref_audios=[], output_prefix=prefix,
    ))
    if chunk_rows:
        g = inject_lossless_mem(g, chunk_rows=chunk_rows)
    return g


def run_one(base: str, gpu: int, case: dict, length: int, seed: int,
            chunk_rows: int | None, warmup: bool) -> dict:
    peak = {"mb": gpu_mem(gpu)}
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(1.0):
            peak["mb"] = max(peak["mb"], gpu_mem(gpu))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    started = time.perf_counter()
    tag = "warmup" if warmup else "results"
    prefix = f"exp_vram_lossless/{tag}/{case['label']}"
    try:
        g = graph_for(case, length, seed, prefix, chunk_rows)
        pid = submit_prompt(base, g)
        entry = wait_prompt(base, pid, timeout=3600, poll_interval=2)
        status = (entry.get("status") or {}).get("status_str")
        return {
            "case": case["label"],
            "status": status,
            "warmup": warmup,
            "chunk_rows": chunk_rows,
            "videos": resolve_videos(case),
            "n_images": len(case.get("ref_images") or []),
            "seed": seed,
            "end_to_end_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"],
            "prompt_id": pid,
            "outputs": extract_outputs(entry),
            "messages": ((entry.get("status") or {}).get("messages") or [])[-8:],
        }
    except Exception as exc:
        return {
            "case": case["label"], "status": "error", "warmup": warmup,
            "chunk_rows": chunk_rows, "error": str(exc)[:4000],
            "end_to_end_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"],
        }
    finally:
        stop.set()
        watcher.join(timeout=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8193")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--no-mem-patch", action="store_true",
                    help="run the production graph without the lossless patch (control)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these case labels (default: the full capacity set)")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--out", default=None, help="write JSON here (default: logs/capacity_gpu1.json)")
    args = ap.parse_args()
    chunk_rows = None if args.no_mem_patch else int(args.chunk_rows)
    health(args.comfy)
    length, actual = align_length(12)

    cases = [
        {"label": "12s_1video_3s", "durations": [3], "ref_images": []},
        {"label": "12s_1video_8s", "durations": [8], "ref_images": []},
        {"label": "12s_1video_8s_b", "durations": [8], "ref_images": [],
         "videos": ["h3_test_video_8s_b.mp4"]},
        {"label": "12s_1video_10s", "durations": [10], "ref_images": []},
        {"label": "12s_1video_12s", "durations": [12], "ref_images": []},
        {"label": "12s_9images_1video_8s", "durations": [8], "ref_images": IMAGES},
        {"label": "12s_1video_15s", "durations": [15], "ref_images": []},
        {"label": "12s_9img_1video_15s", "durations": [15], "ref_images": IMAGES},
        {"label": "12s_9img_3video_3s", "durations": [3, 3, 3], "ref_images": IMAGES},
        {"label": "12s_9img_3video_5s", "durations": [5, 5, 5], "ref_images": IMAGES},
        {"label": "12s_9img_2video_8s", "durations": [8, 8], "ref_images": IMAGES},
        {"label": "12s_9img_2video_10s", "durations": [10, 10], "ref_images": IMAGES},
        {"label": "12s_9img_2video_12s", "durations": [12, 12], "ref_images": IMAGES},
        {"label": "12s_9img_2video_15s", "durations": [15, 15], "ref_images": IMAGES},
        {"label": "12s_9img_3video_8s", "durations": [8, 8, 8], "ref_images": IMAGES},
        {"label": "12s_9img_3video_10s", "durations": [10, 10, 10], "ref_images": IMAGES},
        {"label": "12s_9img_3video_12s", "durations": [12, 12, 12], "ref_images": IMAGES},
        {"label": "12s_9img_3video_15s", "durations": [15, 15, 15], "ref_images": IMAGES},
    ]
    if args.only:
        want = set(args.only)
        cases = [c for c in cases if c["label"] in want]
        if not cases:
            raise SystemExit(f"no cases matched --only {sorted(want)}")

    warmup = None
    if not args.skip_warmup:
        warmup = run_one(
            args.comfy, args.gpu,
            {"label": "12s_warmup_1video_3s", "durations": [3], "ref_images": []},
            length, 990000, chunk_rows, True,
        )
        print(json.dumps(warmup, ensure_ascii=False), flush=True)

    results = []
    for i, case in enumerate(cases):
        row = run_one(args.comfy, args.gpu, case, length, 991000 + i, chunk_rows, False)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        results.append(row)
        if args.stop_on_fail and row.get("status") != "success":
            break

    out = Path(args.out) if args.out else EXP / "logs" / "capacity_gpu1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "gpu": args.gpu, "comfy": args.comfy, "resolution": [WIDTH, HEIGHT],
            "steps": STEPS, "attention": "sage_v2_memeff", "sparge_topk": 0.5,
            "unet": UNET, "lora": LORA_R2V, "chunk_rows": chunk_rows,
            "output_frames": length, "output_actual_seconds": actual,
            "note": "production sage+sparge+4step lora; mem patch is opt-in node 22",
        },
        "warmup": warmup,
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    warmup_ok = warmup is None or warmup.get("status") == "success"
    ok = warmup_ok and all(r.get("status") == "success" for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
