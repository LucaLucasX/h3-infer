#!/usr/bin/env python3
"""12s voice: rolled-back twostep (reencode+bicubic) + Sage2 + Turbo.

Warmup runs once then its outputs are deleted. Formal clip lands in h3_accel_single_ab_12s.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from benchmark_accel_single_ab_12s import COMPLEX_VOICE_TEXT, FPS, LENGTH, OUT_ROOT, STEPS
from benchmark_official_1344x768 import HEIGHT, SEED, WIDTH
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph

LABEL = "07_twostep_v2_sage2_turbo"
WARMUP_LABEL = "_warmup_twostep_discard"
OUT_DIR = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT


def _params(label: str) -> GenerateParams:
    return GenerateParams(
        mode="twostep",
        prompt=COMPLEX_VOICE_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        attention_backend="sage_v2_memeff",
        image="",
        ref_images=[],
        ref_videos=[],
        output_prefix=f"{OUT_ROOT}/{label}",
    )


def _run(label: str, base_url: str) -> dict:
    p = _params(label)
    p1w, p1h, p2w, p2h, split = p.resolve_twostep()
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, build_graph(p))
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        return {
            "label": label,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "mode": "twostep",
            "attention_backend": "sage_v2_memeff",
            "turbo": True,
            "steps": STEPS,
            "pass1": [p1w, p1h],
            "pass2_approx": [p2w, p2h],
            "twostep_split": split,
            "twostep_upscale_method": p.twostep_upscale_method,
            "twostep_audio_denoise": p.twostep_audio_denoise,
            "twostep_full_pass1": p.twostep_full_pass1,
            "twostep_reencode_pass2_cond": p.twostep_reencode_pass2_cond,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:3000],
        }


def _delete_warmup() -> None:
    if not OUT_DIR.is_dir():
        return
    for path in OUT_DIR.glob(f"{WARMUP_LABEL}*"):
        path.unlink(missing_ok=True)
        print(f"deleted warmup {path.name}", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_twostep_12s.json")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    health(args.comfy)
    p0 = _params(LABEL)
    p1w, p1h, _, _, split = p0.resolve_twostep()
    print(
        f"twostep v2 12s: {WIDTH}x{HEIGHT} length={LENGTH} turbo={STEPS} "
        f"pass1={p1w}x{p1h} split={split} method={p0.twostep_upscale_method} "
        f"ad={p0.twostep_audio_denoise} full_pass1={p0.twostep_full_pass1} "
        f"reencode={p0.twostep_reencode_pass2_cond}",
        flush=True,
    )

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = _run(WARMUP_LABEL, args.comfy)
        print(json.dumps({k: w[k] for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        _delete_warmup()

    print(f"=== {LABEL} ===", flush=True)
    result = _run(LABEL, args.comfy)
    if result.get("elapsed_sec"):
        result["baseline_12s_elapsed_sec"] = 290.11
        result["speedup_vs_baseline_12s"] = round(290.11 / result["elapsed_sec"], 3)
    result["compare_dir"] = f"ComfyUI-master_cp/output/{OUT_ROOT}"
    # ensure no warmup leftovers
    _delete_warmup()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
