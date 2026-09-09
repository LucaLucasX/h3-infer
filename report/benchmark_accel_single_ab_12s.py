#!/usr/bin/env python3
"""Single-knob accel ablation @ 12s / voice-heavy complex scene.

Baseline: Sage2 + Turbo-8 + pruned int8. One accel at a time vs baseline.
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
from benchmark_official_1344x768 import HEIGHT, SEED, WIDTH
from h3_graph import GenerateParams, build_graph

# ~12s @ 24fps (snaps to 17k+5 grid → 294 frames ≈ 12.25s)
LENGTH = 288
FPS = 24.0
STEPS = 8
OUT_ROOT = "h3_accel_single_ab_12s"

COMPLEX_VOICE_TEXT = """Cinematic photorealistic rainy-night Tokyo drama, 24fps, shallow depth of field, film grain.

Cast:
- Woman, 28, East Asian, short black bob, red trench coat, black umbrella
- Man, 35, navy wool coat, glasses, paper coffee cup
- Crowd extras with neon reflections

Timeline with spoken dialogue (stereo, clear lip-sync, no voiceover narration):
[0.0s-2.5s] Close-up on the woman under neon kanji; soft jazz piano; rain on umbrella.
  Woman (calm, slightly breathless): "对不起，我迟到了。"
[2.5s-5.0s] Medium shot crosswalk; she looks up; man notices from opposite curb; footsteps in puddles.
  Man (warm, surprised): "没关系，我也刚到。"
[5.0s-8.0s] Tracking shot as they walk toward each other under warm streetlights; bicycle bell once at 6.2s.
  Woman (soft smile): "这雨好像不会停。"
  Man (gentle): "那正好，咖啡还热着。"
[8.0s-12.0s] Over-shoulder meet under convenience-store awning; he offers coffee; rain continues; jazz swells then softens.

Camera: naturalistic handheld, no jump cuts, no text overlays, no logos.
Audio: stereo rain bed, soft jazz trio, footsteps, one bicycle bell, **clear dialogue as above**, no narrator.
Lighting: night rain, cyan/magenta neon + warm shop tungsten.
""".strip()

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
    ("06_spectrum_only", dict(spectrum=True)),
]


def run_case(label: str, overrides: dict, base_url: str) -> dict:
    kw: dict = {
        "prompt": COMPLEX_VOICE_TEXT,
        "width": WIDTH,
        "height": HEIGHT,
        "length": LENGTH,
        "fps": FPS,
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
            "length": p.length,
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
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_accel_single_ab_12s.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    if not health(args.comfy):
        raise SystemExit(f"ComfyUI not reachable at {args.comfy}")

    cases = CASES
    if args.only:
        cases = [(l, o) for l, o in CASES if l in set(args.only)]

    print(
        f"12s voice ablation: {WIDTH}x{HEIGHT} length={LENGTH} turbo={STEPS} seed={SEED}",
        flush=True,
    )
    results: list[dict] = []

    if not args.skip_warmup:
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
    summary = [
        {
            "label": r["label"],
            "elapsed_sec": r["elapsed_sec"],
            "speedup_vs_baseline": round(base_t / r["elapsed_sec"], 3)
            if base_t and r["elapsed_sec"] and r["status"] == "success"
            else None,
        }
        for r in results
    ]

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "target_duration_sec": 12,
            "seed": SEED,
            "steps": STEPS,
            "prompt": "voice-heavy complex Tokyo rain dialogue scene",
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
