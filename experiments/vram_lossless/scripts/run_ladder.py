#!/usr/bin/env python3
"""Ladder benchmark: L0 → L1-token → L1 → L2 with monotonic-chain early stop.

Cases within each difficulty chain stop after the first OOM (harder cases in the
same chain are skipped). seed=42 for all prompts. Writes per-tier JSON under
experiments/vram_lossless/logs/ladder_gpu0_seed42/.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "vram_lossless"
DEV = ROOT / "dev" / "sage_lora_sparge"
SCRIPTS = EXP / "scripts"
sys.path[:0] = [str(ROOT), str(DEV), str(ROOT / "report"), str(SCRIPTS)]

import _bootstrap_h3_graph  # noqa: F401
from comfy_client import extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from inject_mem_patch import inject_lossless_mem
from loras import LORA_R2V

from run_capacity import (  # noqa: E402
    IMAGES,
    STEPS,
    UNET,
    WIDTH,
    HEIGHT,
    FPS,
    align_length,
    gpu_mem,
    resolve_videos,
    video_name,
)

LOG_DIR = EXP / "logs" / "ladder_gpu0_seed42"
COMFY_LOG = ROOT / "experiments" / "pdd" / "logs" / "comfyui_8193.log"
SEED = 42

# Monotonic chains: first OOM skips the rest of the chain for this tier.
CASE_CHAINS: list[list[dict]] = [
    [
        {"label": "12s_1video_3s", "durations": [3], "ref_images": []},
        {"label": "12s_1video_8s", "durations": [8], "ref_images": []},
        {"label": "12s_1video_8s_b", "durations": [8], "ref_images": [],
         "videos": ["h3_test_video_8s_b.mp4"]},
        {"label": "12s_1video_10s", "durations": [10], "ref_images": []},
        {"label": "12s_1video_12s", "durations": [12], "ref_images": []},
        {"label": "12s_1video_15s", "durations": [15], "ref_images": []},
    ],
    [
        {"label": "12s_9images_1video_8s", "durations": [8], "ref_images": IMAGES},
        {"label": "12s_9img_1video_15s", "durations": [15], "ref_images": IMAGES},
    ],
    [
        {"label": "12s_9img_3video_3s", "durations": [3, 3, 3], "ref_images": IMAGES},
        {"label": "12s_9img_3video_5s", "durations": [5, 5, 5], "ref_images": IMAGES},
    ],
    [
        {"label": "12s_9img_2video_8s", "durations": [8, 8], "ref_images": IMAGES},
        {"label": "12s_9img_2video_10s", "durations": [10, 10], "ref_images": IMAGES},
        {"label": "12s_9img_2video_12s", "durations": [12, 12], "ref_images": IMAGES},
        {"label": "12s_9img_2video_15s", "durations": [15, 15], "ref_images": IMAGES},
    ],
    [
        {"label": "12s_9img_3video_8s", "durations": [8, 8, 8], "ref_images": IMAGES},
        {"label": "12s_9img_3video_10s", "durations": [10, 10, 10], "ref_images": IMAGES},
        {"label": "12s_9img_3video_12s", "durations": [12, 12, 12], "ref_images": IMAGES},
        {"label": "12s_9img_3video_15s", "durations": [15, 15, 15], "ref_images": IMAGES},
    ],
]

P1_CASE = {
    "label": "P1_9img_3video_15s",
    "durations": [15, 15, 15],
    "ref_images": IMAGES,
}
P1_MATRIX_LABEL = "12s_9img_3video_15s"

TIERS: dict[str, dict] = {
    "L0": {"patch": False},
    "L1-token": {"patch": True, "mode": "step1", "head_chunks": 56},
    "L1": {"patch": True, "mode": "step1", "head_chunks": 4},
    "L2": {"patch": True, "mode": "step2", "head_chunks": 4},
}

_STEP_RE = re.compile(r"\[H3TURBO step (\d+)\].*elapsed=([\d.]+)s")


def tier_config(name: str) -> dict:
    if name not in TIERS:
        raise SystemExit(f"unknown tier {name!r}; choose from {sorted(TIERS)}")
    return {"name": name, **TIERS[name]}


def graph_for(case: dict, length: int, seed: int, prefix: str, tier: dict) -> dict:
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
    if tier.get("patch"):
        g = inject_lossless_mem(
            g,
            chunk_rows=int(tier.get("chunk_rows", 4096)),
            head_chunks=int(tier["head_chunks"]),
            mode=str(tier.get("mode", "step1")),
        )
    return g


def parse_steps_from_log(prompt_id: str, log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    marker = f'prompt_id": "{prompt_id}"'
    idx = text.rfind(marker)
    if idx < 0:
        chunk = text[-500_000:]
    else:
        chunk = text[idx: idx + 400_000]
    steps: dict[str, float] = {}
    for m in _STEP_RE.finditer(chunk):
        steps[f"step_{int(m.group(1))}_sec"] = float(m.group(2))
    return steps


def is_oom(row: dict) -> bool:
    if row.get("status") == "success":
        return False
    blob = json.dumps(row, ensure_ascii=False).lower()
    return "out of memory" in blob or "oom" in blob or row.get("status") != "success"


def run_one(
    base: str,
    gpu: int,
    case: dict,
    length: int,
    seed: int,
    tier: dict,
    *,
    warmup: bool,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> dict:
    if skipped:
        return {
            "case": case["label"],
            "status": "skipped",
            "skip_reason": skip_reason,
            "warmup": warmup,
            "tier": tier["name"],
            "seed": seed,
        }
    peak = {"mb": gpu_mem(gpu)}
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(1.0):
            peak["mb"] = max(peak["mb"], gpu_mem(gpu))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    started = time.perf_counter()
    tag = f"ladder/{tier['name']}/{'warmup' if warmup else 'results'}"
    prefix = f"exp_vram_lossless/{tag}/{case['label']}"
    try:
        g = graph_for(case, length, seed, prefix, tier)
        pid = submit_prompt(base, g)
        entry = wait_prompt(base, pid, timeout=7200, poll_interval=2)
        status = (entry.get("status") or {}).get("status_str")
        row = {
            "case": case["label"],
            "status": status,
            "warmup": warmup,
            "tier": tier["name"],
            "seed": seed,
            "videos": resolve_videos(case),
            "n_images": len(case.get("ref_images") or []),
            "end_to_end_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"],
            "prompt_id": pid,
            "outputs": extract_outputs(entry),
            "messages": ((entry.get("status") or {}).get("messages") or [])[-4:],
        }
        row.update(parse_steps_from_log(pid, COMFY_LOG))
        if status != "success":
            row["oom"] = is_oom(row)
        return row
    except Exception as exc:
        err = str(exc)[:4000]
        return {
            "case": case["label"],
            "status": "error",
            "warmup": warmup,
            "tier": tier["name"],
            "seed": seed,
            "error": err,
            "oom": "out of memory" in err.lower() or "oom" in err.lower(),
            "end_to_end_sec": round(time.perf_counter() - started, 3),
            "peak_gpu_mem_mb": peak["mb"],
        }
    finally:
        stop.set()
        watcher.join(timeout=2)


def run_tier(base: str, gpu: int, tier_name: str, *, skip_warmup: bool) -> dict:
    tier = tier_config(tier_name)
    length, actual = align_length(12)
    results: list[dict] = []
    warmup = None
    if not skip_warmup:
        warmup = run_one(
            base, gpu,
            {"label": "12s_warmup_1video_3s", "durations": [3], "ref_images": []},
            length, SEED, tier, warmup=True,
        )
        print(json.dumps(warmup, ensure_ascii=False), flush=True)

    for chain in CASE_CHAINS:
        chain_failed = False
        for case in chain:
            if chain_failed:
                row = run_one(
                    base, gpu, case, length, SEED, tier,
                    warmup=False, skipped=True,
                    skip_reason="earlier case in chain OOM/skipped",
                )
            else:
                row = run_one(base, gpu, case, length, SEED, tier, warmup=False)
                if is_oom(row):
                    chain_failed = True
            print(json.dumps(row, ensure_ascii=False), flush=True)
            results.append(row)

    payload = {
        "config": {
            "tier": tier_name,
            "gpu": gpu,
            "comfy": base,
            "seed": SEED,
            "resolution": [WIDTH, HEIGHT],
            "steps": STEPS,
            "chunk_rows": 4096 if tier.get("patch") else None,
            "head_chunks": tier.get("head_chunks"),
            "mode": tier.get("mode"),
            "output_frames": length,
            "output_actual_seconds": actual,
            "early_stop": "per-chain on first OOM",
        },
        "warmup": warmup,
        "results": results,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{tier_name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return payload


def run_l2_p1(base: str, gpu: int, l2_payload: dict) -> dict | None:
    for row in l2_payload.get("results") or []:
        if row.get("case") == P1_MATRIX_LABEL and row.get("status") == "success":
            print(f"P1 skipped: {P1_MATRIX_LABEL} already success in L2 matrix", flush=True)
            return None
    tier = tier_config("L2")
    length, actual = align_length(12)
    row = run_one(base, gpu, P1_CASE, length, SEED, tier, warmup=False)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "config": {
            "tier": "L2-P1",
            "note": "9img+3x15s ceiling probe; skip if matrix already passed",
            "seed": SEED,
            "output_actual_seconds": actual,
        },
        "results": [row],
    }
    out = LOG_DIR / "L2_P1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8193")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--tier",
        choices=["L0", "L1-token", "L1", "L2", "all"],
        default="all",
    )
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--p1-only", action="store_true",
                    help="run L2 P1 probe using existing L2.json")
    args = ap.parse_args()
    health(args.comfy)

    if args.p1_only:
        l2_path = LOG_DIR / "L2.json"
        if not l2_path.exists():
            raise SystemExit(f"missing {l2_path}")
        run_l2_p1(args.comfy, args.gpu, json.loads(l2_path.read_text()))
        return 0

    order = ["L0", "L1-token", "L1", "L2"] if args.tier == "all" else [args.tier]
    l2_payload = None
    for name in order:
        print(f"\n=== tier {name} ===", flush=True)
        payload = run_tier(args.comfy, args.gpu, name, skip_warmup=args.skip_warmup)
        if name == "L2":
            l2_payload = payload
    if l2_payload is not None:
        run_l2_p1(args.comfy, args.gpu, l2_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
