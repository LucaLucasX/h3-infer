#!/usr/bin/env python3
"""Twostep v2 quality retake: direct vs fixed twostep @ 20 and Turbo-8.

Uses new defaults: pass2 full-res re-encode, bicubic upscale, split=steps*2//5.
Runs twostep with audio_denoise=0.25 (official) and 0.0 (stable audio).
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

OUT_ROOT = "h3_twostep_v2_ab"
SEED = 42

CASES: list[tuple[str, dict]] = [
    ("01_direct_20", dict(mode="t2v", steps=20, turbo=False)),
    ("02_twostep_v2_ad025_20", dict(mode="twostep", steps=20, turbo=False, twostep_audio_denoise=0.25)),
    ("03_twostep_v2_ad0_20", dict(mode="twostep", steps=20, turbo=False, twostep_audio_denoise=0.0)),
    ("04_direct_lora_8", dict(mode="t2v", steps=8, turbo=True)),
    ("05_twostep_v2_ad025_lora_8", dict(mode="twostep", steps=8, turbo=True, twostep_audio_denoise=0.25)),
    ("06_twostep_v2_ad0_lora_8", dict(mode="twostep", steps=8, turbo=True, twostep_audio_denoise=0.0)),
]


def run_case(label: str, overrides: dict, base_url: str) -> dict:
    kw: dict = {
        "prompt": COMPLEX_TEXT,
        "width": WIDTH,
        "height": HEIGHT,
        "length": LENGTH,
        "seed": SEED,
        "output_prefix": f"{OUT_ROOT}/{label}",
        "attention_backend": "sage_v2_memeff",
        "twostep_full_pass1": True,
        "twostep_reencode_pass2_cond": True,
        "twostep_upscale_method": "bicubic",
        "ref_images": [],
        "ref_videos": [],
        "image": "",
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
            "mode": p.mode,
            "steps": p.steps,
            "turbo": p.turbo,
            "twostep_audio_denoise": p.twostep_audio_denoise if p.mode == "twostep" else None,
            "twostep_reencode_pass2_cond": p.twostep_reencode_pass2_cond,
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
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_twostep_v2_ab.json")
    args = ap.parse_args()

    if not health(args.comfy):
        raise SystemExit(f"ComfyUI not reachable at {args.comfy}")

    print(f"Twostep v2 A/B: {WIDTH}x{HEIGHT} seed={SEED} sage2", flush=True)
    results: list[dict] = []

    # Warmup twostep path (discard)
    print("\n=== warmup_twostep (discard) ===", flush=True)
    w = run_case("warmup_twostep", dict(mode="twostep", steps=8, turbo=True, twostep_audio_denoise=0.0), args.comfy)
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

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "attention": "sage_v2_memeff",
            "twostep_fixes": {
                "full_pass1": True,
                "reencode_pass2_cond": True,
                "upscale_method": "bicubic",
                "split_default": "steps*2//5",
            },
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
