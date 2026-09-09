#!/usr/bin/env python3
"""Direct t2v baseline vs max-acceleration stack @ Turbo-8.

Baseline: pruned int8 UNET + Sage2 MemEff + Turbo LoRA (distill), no caches.
Max stack: baseline + TE-Speed + TeaCache + fp16_accum + EasyCache.
(MagCache has no MiniMax H3 support; Spectrum skipped — mutually exclusive with EasyCache.)
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

OUT_ROOT = "h3_direct_accel_ab"
STEPS = 8

BASELINE = dict(
    mode="t2v",
    steps=STEPS,
    turbo=True,
    attention_backend="sage_v2_memeff",
    ref_images=[],
    ref_videos=[],
    image="",
)

MAX_ACCEL = dict(
    **BASELINE,
    tespeed=True,
    teacache=True,
    fp16_accumulation=True,
    easycache=True,
    spectrum=False,
)

CASES: list[tuple[str, dict]] = [
    ("01_baseline_turbo8", BASELINE),
    ("02_max_accel_turbo8", MAX_ACCEL),
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
            "steps": p.steps,
            "turbo": p.turbo,
            "attention": p.resolved_attention_backend(),
            "tespeed": p.tespeed,
            "teacache": p.teacache,
            "fp16_accumulation": p.fp16_accumulation,
            "easycache": p.easycache,
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
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_direct_accel_ab.json")
    args = ap.parse_args()

    if not health(args.comfy):
        raise SystemExit(f"ComfyUI not reachable at {args.comfy}")

    print(
        f"Direct accel A/B: {WIDTH}x{HEIGHT} turbo={STEPS} steps seed={SEED}",
        flush=True,
    )
    results: list[dict] = []

    print("\n=== warmup (discard) ===", flush=True)
    w = run_case("warmup", MAX_ACCEL, args.comfy)
    print(json.dumps({k: w[k] for k in w if k != "outputs"}, ensure_ascii=False), flush=True)
    if w["status"] != "success":
        raise SystemExit(f"warmup failed: {w.get('error')}")

    for label, overrides in CASES:
        print(f"\n=== {label} ===", flush=True)
        row = run_case(label, overrides, args.comfy)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        if row["status"] != "success":
            raise SystemExit(f"failed: {row.get('error')}")

    base_t = next(r["elapsed_sec"] for r in results if r["label"].startswith("01_"))
    max_t = next(r["elapsed_sec"] for r in results if r["label"].startswith("02_"))
    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "steps": STEPS,
            "baseline": "pruned_int8 + sage_v2_memeff + turbo_lora",
            "max_stack": "baseline + tespeed + teacache + fp16_accum + easycache",
            "skipped": ["MagCache (no H3 support)", "Spectrum (exclusive with EasyCache)"],
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
        },
        "speedup_baseline_over_max": round(base_t / max_t, 4) if max_t else None,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)
    print(f"Speedup (baseline/max): {payload['speedup_baseline_over_max']}x", flush=True)


if __name__ == "__main__":
    main()
