#!/usr/bin/env python3
"""Single-knob acceleration ablation @ Turbo-8 Sage2 baseline.

One cache/accel at a time vs baseline vs full stack (reference).
Same prompt/seed/size as h3_direct_accel_ab.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from benchmark_official_1344x768 import COMPLEX_TEXT, HEIGHT, LENGTH, SEED, WIDTH
from h3_graph import GenerateParams, build_graph

OUT_ROOT = "h3_accel_single_ab"
STEPS = 8

COMMON = dict(
    mode="t2v",
    steps=STEPS,
    turbo=True,
    attention_backend="sage_v2_memeff",
    ref_images=[],
    ref_videos=[],
    image="",
)

CASES: list[tuple[str, dict]] = [
    ("01_baseline", {}),
    ("02_fp16_only", dict(fp16_accumulation=True)),
    ("03_easycache_only", dict(easycache=True)),
    ("04_teacache_only", dict(teacache=True)),
    ("05_tespeed_only", dict(tespeed=True)),
    ("07_spectrum_only", dict(spectrum=True)),
    (
        "06_max_stack",
        dict(
            fp16_accumulation=True,
            easycache=True,
            teacache=True,
            tespeed=True,
        ),
    ),
]


def run_case(label: str, overrides: dict, base_url: str) -> dict:
    kw: dict = {
        "prompt": COMPLEX_TEXT,
        "width": WIDTH,
        "height": HEIGHT,
        "length": LENGTH,
        "seed": SEED,
        "output_prefix": f"{OUT_ROOT}/{label}",
    }
    kw.update(COMMON)
    kw.update(overrides)
    p = GenerateParams(**kw)
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, build_graph(p))
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        return {
            "label": label,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "fp16_accumulation": p.fp16_accumulation,
            "easycache": p.easycache,
            "teacache": p.teacache,
            "tespeed": p.tespeed,
            "spectrum": p.spectrum,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:3000],
            **overrides,
        }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_accel_single_ab.json")
    ap.add_argument("--only", nargs="*", help="Run subset labels, e.g. 04_teacache_only")
    args = ap.parse_args()

    if not health(args.comfy):
        raise SystemExit(f"ComfyUI not reachable at {args.comfy}")

    cases = CASES
    if args.only:
        allow = set(args.only)
        cases = [(l, o) for l, o in CASES if l in allow]

    print(f"Accel single ablation: {WIDTH}x{HEIGHT} turbo={STEPS} seed={SEED}", flush=True)
    results: list[dict] = []

    if not args.only:
        print("\n=== warmup_baseline (discard) ===", flush=True)
        w = run_case("warmup_baseline", {}, args.comfy)
        print(json.dumps({k: w[k] for k in w if k != "outputs"}, ensure_ascii=False), flush=True)
        if w["status"] != "success":
            raise SystemExit(f"warmup failed: {w.get('error')}")

    for label, overrides in cases:
        print(f"\n=== {label} ===", flush=True)
        row = run_case(label, overrides, args.comfy)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        if row["status"] != "success":
            raise SystemExit(f"failed: {row.get('error')}")

    base_t = next((r["elapsed_sec"] for r in results if r["label"] == "01_baseline"), None)
    summary = []
    for r in results:
        t = r["elapsed_sec"]
        speedup = round(base_t / t, 3) if base_t and t and r["status"] == "success" else None
        summary.append({"label": r["label"], "elapsed_sec": t, "speedup_vs_baseline": speedup})

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "steps": STEPS,
            "baseline": "pruned_int8 + sage_v2_memeff + turbo_lora",
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
        },
        "summary": summary,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
