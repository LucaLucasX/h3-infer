#!/usr/bin/env python3
"""Ablation: Sage2 baseline vs +fp16_accum vs +EasyCache vs both.

Fixed: t2v direct 1344×768, 20 steps, seed=42, Sage2 MemEff.
Timing: wall-clock submit → history done (TE/VAE/IO included).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from benchmark_official_1344x768 import COMPLEX_TEXT, HEIGHT, LENGTH, SEED, WIDTH
from h3_graph import GenerateParams, build_graph

OUT_ROOT = "h3_accel_ablation"

CASES: list[tuple[str, dict]] = [
    ("01_baseline_sage2", {}),
    ("02_fp16_accum", {"fp16_accumulation": True}),
    ("03_easycache", {"easycache": True}),
    ("04_fp16_easycache", {"fp16_accumulation": True, "easycache": True}),
]


def run_case(label: str, overrides: dict, base_url: str) -> dict:
    prefix = f"{OUT_ROOT}/{label}"
    p = GenerateParams(
        mode="t2v",
        prompt=COMPLEX_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=20,
        seed=SEED,
        output_prefix=prefix,
        attention_backend="sage_v2_memeff",
        **overrides,
    )
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, build_graph(p))
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        return {
            "label": label,
            "status": "success",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "fp16_accumulation": p.fp16_accumulation,
            "easycache": p.easycache,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e),
            **overrides,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_accel_ablation.json")
    ap.add_argument("--only", nargs="*", help="Run subset labels, e.g. 02_fp16_accum")
    args = ap.parse_args()

    if not health(args.comfy):
        raise SystemExit(f"ComfyUI not reachable at {args.comfy}")

    cases = CASES
    if args.only:
        only = set(args.only)
        cases = [(l, o) for l, o in CASES if l in only]

    results: list[dict] = []
    baseline: float | None = None
    for label, overrides in cases:
        print(f"\n=== {label} ===", flush=True)
        row = run_case(label, overrides, args.comfy)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        if row["status"] == "success":
            if baseline is None:
                baseline = row["elapsed_sec"]
            elif baseline:
                speedup = round(baseline / row["elapsed_sec"], 3)
                row["speedup_vs_baseline"] = speedup
                print(f"  vs baseline: {speedup:.3f}x", flush=True)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": 20,
            "seed": SEED,
            "mode": "t2v",
            "attention": "sage_v2_memeff",
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
