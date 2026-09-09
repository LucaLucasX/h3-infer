#!/usr/bin/env python3
"""Benchmark pruned vs. non-pruned H3 int8 diffusion weights."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "dev" / "sage_lora_sparge"
sys.path[:0] = [str(ROOT), str(DEV), str(ROOT / "report")]
import _bootstrap_h3_graph  # noqa: F401  # type: ignore[reportMissingImports]

from comfy_client import extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from loras import LORA_R2V, LORA_T2V  # type: ignore[reportMissingImports]

COMFY = "http://127.0.0.1:8190"
WIDTH, HEIGHT, LENGTH, STEPS, SEED = 1376, 768, 124, 4, 42
REF_IMAGES = [
    "realref_full_01.png",
    "realref_full_02.png",
    "realref_full_03.png",
]
MODELS = {
    "t2v_pruned": ("t2v", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", LORA_T2V),
    "t2v_unpruned": ("t2v", "minimax_h3_fl2va_int8_convrot.safetensors", LORA_T2V),
    "r2v_pruned": ("r2v", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", LORA_R2V),
    "r2v_unpruned": ("r2v", "minimax_h3_ref2va_int8_convrot.safetensors", LORA_R2V),
}


def gpu_mem_mb() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def graph_for(label: str, seed: int) -> dict:
    mode, unet_name, lora = MODELS[label]
    params = GenerateParams(
        mode=mode,
        prompt=(
            "Cinematic photorealistic golden-hour forest, a woman walks along a "
            "mossy path and turns toward camera, natural motion, no text or logos."
        ),
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=24.0,
        steps=STEPS,
        seed=seed,
        turbo=True,
        turbo_lora=lora,
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=True,
        sparge_topk=0.5,
        sparge_dense_first_steps=0,
        sparge_num_layers=50,
        sparge_audio_dense=True,
        unet_name=unet_name,
        ref_images=REF_IMAGES if mode == "r2v" else [],
        ref_videos=[],
        output_prefix=f"bench_pruned_vs_unpruned/{label}",
    )
    return build_graph(params)


def error_text(entry: dict) -> str:
    messages = (entry.get("status") or {}).get("messages") or []
    return str(messages)[-4000:]


def run_one(label: str, seed: int) -> dict:
    started = time.perf_counter()
    peak = {"mb": gpu_mem_mb() or 0}
    stop = threading.Event()

    def watch_gpu() -> None:
        while not stop.wait(1.0):
            current = gpu_mem_mb()
            if current is not None:
                peak["mb"] = max(peak["mb"], current)

    watcher = threading.Thread(target=watch_gpu, daemon=True)
    watcher.start()
    try:
        prompt_id = submit_prompt(COMFY, graph_for(label, seed))
        entry = wait_prompt(COMFY, prompt_id, timeout=1800, poll_interval=2.0)
        status = (entry.get("status") or {}).get("status_str")
        row = {
            "case": label,
            "status": "success" if status == "success" else f"comfy_{status}",
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"],
            "prompt_id": prompt_id,
            "outputs": extract_outputs(entry),
        }
        if status != "success":
            row["error"] = error_text(entry)
        return row
    except Exception as exc:
        return {
            "case": label,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"],
            "error": str(exc)[:4000],
        }
    finally:
        stop.set()
        watcher.join(timeout=2)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    health(COMFY)
    results = []
    order = list(MODELS)
    if args.reverse:
        order = [
            "t2v_unpruned",
            "t2v_pruned",
            "r2v_unpruned",
            "r2v_pruned",
        ]
    warmups = []
    for label in order:
        print(f"warmup {label}", flush=True)
        warmup = run_one(label, SEED)
        warmups.append(warmup)
        print(json.dumps(warmup, ensure_ascii=False), flush=True)
        print(f"running {label}", flush=True)
        # Use a different seed so ComfyUI classic cache cannot return warmup.
        result = run_one(label, SEED + 1000)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    output_name = (
        "bench_unpruned_first.json" if args.reverse
        else "bench_pruned_vs_unpruned.json"
    )
    output = DEV / "logs" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "config": {
            "canvas": [WIDTH, HEIGHT],
            "frames_requested": LENGTH,
            "steps": STEPS,
            "attention": "sage_v2_memeff",
            "sparge_topk": 0.5,
            "ref_images": 3,
            "warmup_per_case": True,
        },
        "warmups": warmups,
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output}", flush=True)
    return 0 if all(row["status"] == "success" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
