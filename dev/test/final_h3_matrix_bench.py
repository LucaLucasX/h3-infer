#!/usr/bin/env python3
"""Final H3 t2v/fl2v/ref2v matrix benchmark with per-node timings."""
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
sys.path[:0] = [str(ROOT), str(DEV), str(ROOT / "report")]

import _bootstrap_h3_graph  # noqa: F401
from comfy_client import extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from loras import LORA_R2V, LORA_T2V

WIDTH, HEIGHT, FPS, STEPS = 1376, 768, 24.0, 4
UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
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
AUDIO = [
    "h3rb_0344004e3b92_speech_r2va_5s.mp3",
    "h3rb_46969c83c9b7_beach_ambience_10s.mp3",
    "h3rb_72b088b9d44e_music_3s.mp3",
]
STEP_RE = re.compile(r"\[H3TURBO step (\d+)\].*?elapsed=([0-9.]+)s")
NODE_RE = re.compile(
    r"\[KikoStats\] Completed tracking: (.*?) \(([-0-9.]+)ms, "
    r"CPU: ([-0-9.]+)%, GPU: ([-0-9.]+)%\)"
)
PROMPT_RE = re.compile(r"Prompt executed in ([0-9.]+) seconds")
TIMING_RE = re.compile(r"\[([A-Z0-9_]+TIMING)\] (.*)")
TE_RE = re.compile(r"\[H3_TE_TIMING\] encode=([0-9.]+)s")
KV_RE = re.compile(r"([a-z_]+)=([0-9.]+)s")


def duration_config(seconds: int) -> tuple[int, float]:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    length = min(candidates, key=lambda value: abs(value - requested))
    return length, length / FPS


def gpu_mem_mb(gpu: int) -> int | None:
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, check=False,
    )
    try:
        return int(p.stdout.strip().splitlines()[0])
    except Exception:
        return None


def read_log(path: Path, offset: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()
    except FileNotFoundError:
        return ""


def parse_log(text: str) -> dict:
    nodes = []
    for m in NODE_RE.finditer(text):
        nodes.append({
            "node": m.group(1),
            "duration_ms": float(m.group(2)),
            "cpu_percent": float(m.group(3)),
            "gpu_percent": float(m.group(4)),
        })
    steps = [
        {"step": int(m.group(1)), "duration_sec": float(m.group(2))}
        for m in STEP_RE.finditer(text)
    ]
    timings = []
    for m in TIMING_RE.finditer(text):
        timings.append({"kind": m.group(1), "details": m.group(2)})
    prompt_times = [float(x) for x in PROMPT_RE.findall(text)]
    return {
        "nodes": nodes,
        "steps": steps,
        "prompt_execution_sec": prompt_times[-1] if prompt_times else None,
        "timing_lines": timings,
        "te_encode_sec": float(te_matches[-1]) if (te_matches := TE_RE.findall(text)) else None,
    }


def graph_for(case: dict, length: int, seed: int, prefix: str) -> dict:
    mode = case["mode"]
    is_ref = mode == "r2v"
    params = GenerateParams(
        mode=mode,
        prompt=(
            "Cinematic photorealistic golden-hour forest, a woman walks along a "
            "mossy path and turns toward camera, natural motion, no text or logos."
        ),
        width=WIDTH, height=HEIGHT, length=length, fps=FPS, steps=STEPS, seed=seed,
        turbo=True,
        turbo_lora=LORA_R2V if is_ref else LORA_T2V,
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=True, sparge_topk=0.5, sparge_dense_first_steps=0,
        sparge_num_layers=50, sparge_audio_dense=True,
        unet_name=UNET_REF2VA if is_ref else UNET_FL2VA,
        image=case.get("image", ""),
        last_frame=case.get("last_frame"),
        ref_images=case.get("ref_images", []),
        ref_videos=[],
        ref_audios=AUDIO if case.get("audio") else [],
        output_prefix=prefix,
    )
    return build_graph(params)


def run_one(base: str, log_path: Path, gpu: int, case: dict, length: int,
            actual_duration: float, seed: int, warmup: bool = False) -> dict:
    label = case["label"]
    prefix = f"final_h3_matrix_{'warmup' if warmup else 'results'}/{label}"
    try:
        offset = log_path.stat().st_size if log_path.exists() else 0
        peak = {"mb": gpu_mem_mb(gpu) or 0}
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(1.0):
                current = gpu_mem_mb(gpu)
                if current is not None:
                    peak["mb"] = max(peak["mb"], current)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        started = time.perf_counter()
        prompt_id = submit_prompt(base, graph_for(case, length, seed, prefix))
        entry = wait_prompt(base, prompt_id, timeout=3600, poll_interval=2.0)
        elapsed = time.perf_counter() - started
        stop.set()
        watcher.join(timeout=2)
        parsed = parse_log(read_log(log_path, offset))
        status = (entry.get("status") or {}).get("status_str")
        result = {
            "case": label,
            "mode": case["mode"],
            "images": len(case.get("ref_images", [])) + (2 if case["mode"] == "i2v" and case.get("last_frame") else (1 if case["mode"] == "i2v" else 0)),
            "audio_count": 3 if case.get("audio") else 0,
            "requested_duration_sec": case["requested_duration"],
            "actual_duration_sec": actual_duration,
            "frames": length,
            "seed": seed,
            "status": status,
            "end_to_end_sec": round(elapsed, 3),
            "peak_gpu_mem_mb": peak["mb"],
            "prompt_execution_sec": parsed["prompt_execution_sec"],
            "te_encode_sec": parsed["te_encode_sec"],
            "te_cached": parsed["te_encode_sec"] is None,
            "steps": parsed["steps"],
            "node_timings": parsed["nodes"],
            "explicit_timings": parsed["timing_lines"],
            "outputs": extract_outputs(entry),
            "prompt_id": prompt_id,
        }
        return result
    except Exception as exc:
        return {
            "case": label, "mode": case["mode"], "status": "error",
            "requested_duration_sec": case["requested_duration"],
            "actual_duration_sec": actual_duration, "frames": length,
            "error": str(exc)[:4000],
        }


def cases_for(seconds: int) -> list[dict]:
    cases = [{
        "label": f"{seconds}s_t2v",
        "mode": "t2v",
        "requested_duration": seconds,
    }]
    cases += [
        {
            "label": f"{seconds}s_fl2v_first",
            "mode": "i2v",
            "image": IMAGES[6],
            "requested_duration": seconds,
        },
        {
            "label": f"{seconds}s_fl2v_first_last",
            "mode": "i2v",
            "image": IMAGES[6],
            "last_frame": IMAGES[8],
            "requested_duration": seconds,
        },
    ]
    for count in range(1, 10):
        refs = IMAGES[:count]
        cases.append({
            "label": f"{seconds}s_ref2v_i{count}",
            "mode": "r2v", "ref_images": refs, "requested_duration": seconds,
        })
        cases.append({
            "label": f"{seconds}s_ref2v_i{count}a3",
            "mode": "r2v", "ref_images": refs, "audio": True,
            "requested_duration": seconds,
        })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, choices=(5, 12), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    health(base)
    length, actual_duration = duration_config(args.seconds)
    cases = cases_for(args.seconds)
    results = []
    warmups = []
    # One warmup per model/LoRA pair: fl2va first, then ref2va.
    warmup_cases = [cases[0], next(case for case in cases if case["mode"] == "r2v")]
    for index, case in enumerate(warmup_cases):
        warmups.append(run_one(
            base, args.log, args.gpu, case, length, actual_duration,
            900000 + index, warmup=True,
        ))
    for index, case in enumerate(cases):
        result = run_one(
            base, args.log, args.gpu, case, length, actual_duration,
            910000 + index,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    report = {
        "config": {
            "gpu": args.gpu, "port": args.port, "requested_duration_sec": args.seconds,
            "actual_duration_sec": actual_duration, "frames": length,
            "resolution": [WIDTH, HEIGHT], "fps": FPS, "steps": STEPS,
            "attention": "sage_v2_memeff", "sparge_topk": 0.5,
            "unet": "pruned_int8_convrot", "warmup_policy": "one_per_model_lora",
            "audio_files": AUDIO,
        },
        "warmups": warmups,
        "results": results,
    }
    output = DEV / "logs" / f"final_h3_matrix_{args.seconds}s.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output}", flush=True)
    return 0 if all(r.get("status") == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
