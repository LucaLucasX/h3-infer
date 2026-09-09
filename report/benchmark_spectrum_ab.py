#!/usr/bin/env python3
"""Rigorous Spectrum A/B: direct t2v Sage2 vs Sage2+Spectrum.

Protocol:
  1) Warmup once per config (discarded; loads weights / CUDA kernels / Spectrum path)
  2) Timed: interleave baseline / spectrum × 3 (reduces thermal order bias)
  3) Same prompt/seed/size/steps; only spectrum flag differs
  4) Keep every timed video under output/h3_spectrum_ab/

Timing: wall-clock submit → history done (TE/VAE/IO included).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from benchmark_official_1344x768 import COMPLEX_TEXT, HEIGHT, LENGTH, SEED, WIDTH
from h3_graph import GenerateParams, build_graph

OUT_ROOT = "h3_spectrum_ab"
STEPS = 20
REPEATS = 3


def make_params(*, spectrum: bool, prefix: str, debug: bool = False) -> GenerateParams:
    return GenerateParams(
        mode="t2v",
        prompt=COMPLEX_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=STEPS,
        seed=SEED,
        output_prefix=prefix,
        attention_backend="sage_v2_memeff",
        turbo=False,
        fp16_accumulation=False,
        easycache=False,
        spectrum=spectrum,
        spectrum_debug=debug,
        # README default audio-safe path
        spectrum_offline_smoothing_replay=True,
        spectrum_audio_blend_weight=0.0,
        spectrum_blend_weight=0.50,
    )


def run_once(label: str, spectrum: bool, base_url: str, *, timed: bool, debug: bool = False) -> dict:
    prefix = f"{OUT_ROOT}/{label}"
    p = make_params(spectrum=spectrum, prefix=prefix, debug=debug)
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, build_graph(p))
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 3)
        outs = extract_outputs(hist)
        st = (hist.get("status") or {}).get("status_str")
        row = {
            "label": label,
            "timed": timed,
            "spectrum": spectrum,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": outs,
        }
        if st != "success":
            row["error"] = str((hist.get("status") or {}).get("messages") or "")[:2000]
        return row
    except ComfyUIError as e:
        return {
            "label": label,
            "timed": timed,
            "spectrum": spectrum,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "error": str(e)[:3000],
        }


def summarize(rows: list[dict], key: str) -> dict:
    ok = [r for r in rows if r.get("timed") and r.get("spectrum") == (key == "spectrum") and r.get("status") == "success"]
    times = [r["elapsed_sec"] for r in ok]
    if not times:
        return {"n": 0, "times": [], "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(times),
        "times": times,
        "mean": round(statistics.mean(times), 3),
        "stdev": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "min": round(min(times), 3),
        "max": round(max(times), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_spectrum_ab.json")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    if not health(args.comfy):
        raise SystemExit(f"ComfyUI not reachable at {args.comfy}")

    all_rows: list[dict] = []
    print(
        f"Spectrum A/B: {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} seed={SEED} "
        f"sage2 repeats={args.repeats}",
        flush=True,
    )

    if not args.skip_warmup:
        for spectrum, tag in ((False, "warmup_baseline"), (True, "warmup_spectrum")):
            print(f"\n=== {tag} (discard) ===", flush=True)
            # debug on spectrum warmup to confirm forecasts in Comfy log
            row = run_once(tag, spectrum, args.comfy, timed=False, debug=spectrum)
            all_rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            if row["status"] != "success":
                raise SystemExit(f"warmup failed: {row.get('error')}")

    # Interleaved timed runs
    for i in range(1, args.repeats + 1):
        for spectrum, name in ((False, "baseline"), (True, "spectrum")):
            label = f"{name}_r{i}"
            print(f"\n=== TIMED {label} ===", flush=True)
            row = run_once(label, spectrum, args.comfy, timed=True, debug=False)
            all_rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            if row["status"] != "success":
                raise SystemExit(f"timed run failed: {row.get('error')}")

    base = summarize(all_rows, "baseline")
    spec = summarize(all_rows, "spectrum")
    speedup = None
    if base["mean"] and spec["mean"]:
        speedup = round(base["mean"] / spec["mean"], 4)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "seed": SEED,
            "mode": "t2v",
            "attention": "sage_v2_memeff",
            "sampler": "res_multistep",
            "scheduler": "simple",
            "spectrum_defaults": {
                "offline_smoothing_replay": True,
                "blend_weight": 0.50,
                "audio_blend_weight": 0.0,
                "degree": 1,
            },
            "protocol": "warmup_each_then_interleaved_3x",
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
        },
        "summary": {
            "baseline": base,
            "spectrum": spec,
            "speedup_baseline_over_spectrum": speedup,
            "note": "speedup > 1 means Spectrum is faster end-to-end",
        },
        "runs": all_rows,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
