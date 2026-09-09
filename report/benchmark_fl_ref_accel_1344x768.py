#!/usr/bin/env python3
"""Accel matrix for fl2va first+last and ref2va (was 14–19 direct-only).

1344×768, ~5s, complex prompts. Covers:
  - direct + Sage2/Sage3 + Turbo LoRA
  - twostep + Sage2/Sage3 + Turbo LoRA
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
from h3_graph import TURBO_LORA_V4, GenerateParams, build_graph
from benchmark_official_1344x768 import (
    COMPLEX_FL2VA,
    COMPLEX_REF_IMG,
    COMPLEX_REF_VID,
    FIRST_FRAME,
    LAST_FRAME,
    REF_IMAGES,
    REF_VIDEO,
    WIDTH,
    HEIGHT,
    LENGTH,
    SEED,
)

OUT_ROOT = "h3_official_1344x768_accel_fl_ref"

# Focused accel cases (8-step primary; 20-step for best-of each kind)
CASES: list[tuple[str, dict]] = [
    # --- fl2va first+last ---
    ("A01_fl_direct_sage2_8", dict(kind="firstlast", mode="i2v", steps=8, attention="sage_v2_memeff", turbo=False)),
    ("A02_fl_direct_sage3_8", dict(kind="firstlast", mode="i2v", steps=8, attention="sage_v3", turbo=False)),
    ("A03_fl_direct_lora_sage2_8", dict(kind="firstlast", mode="i2v", steps=8, attention="sage_v2_memeff", turbo=True)),
    ("A04_fl_direct_lora_sage3_8", dict(kind="firstlast", mode="i2v", steps=8, attention="sage_v3", turbo=True)),
    ("A05_fl_twostep_8", dict(kind="firstlast", mode="twostep", steps=8, attention="pytorch", turbo=False)),
    ("A06_fl_twostep_sage2_8", dict(kind="firstlast", mode="twostep", steps=8, attention="sage_v2_memeff", turbo=False)),
    ("A07_fl_twostep_sage3_8", dict(kind="firstlast", mode="twostep", steps=8, attention="sage_v3", turbo=False)),
    ("A08_fl_twostep_lora_sage2_8", dict(kind="firstlast", mode="twostep", steps=8, attention="sage_v2_memeff", turbo=True)),
    ("A09_fl_twostep_lora_sage3_8", dict(kind="firstlast", mode="twostep", steps=8, attention="sage_v3", turbo=True)),
    ("A10_fl_twostep_lora_sage3_20", dict(kind="firstlast", mode="twostep", steps=20, attention="sage_v3", turbo=True)),
    # --- ref2va images ---
    ("B01_refimg_direct_sage2_8", dict(kind="ref_images", mode="r2v", steps=8, attention="sage_v2_memeff", turbo=False)),
    ("B02_refimg_direct_sage3_8", dict(kind="ref_images", mode="r2v", steps=8, attention="sage_v3", turbo=False)),
    ("B03_refimg_direct_lora_sage2_8", dict(kind="ref_images", mode="r2v", steps=8, attention="sage_v2_memeff", turbo=True)),
    ("B04_refimg_direct_lora_sage3_8", dict(kind="ref_images", mode="r2v", steps=8, attention="sage_v3", turbo=True)),
    ("B05_refimg_twostep_sage3_8", dict(kind="ref_images", mode="twostep", steps=8, attention="sage_v3", turbo=False)),
    ("B06_refimg_twostep_lora_sage3_8", dict(kind="ref_images", mode="twostep", steps=8, attention="sage_v3", turbo=True)),
    ("B07_refimg_twostep_lora_sage3_20", dict(kind="ref_images", mode="twostep", steps=20, attention="sage_v3", turbo=True)),
    # --- ref2va video ---
    ("C01_refvid_direct_lora_sage3_8", dict(kind="ref_video", mode="r2v", steps=8, attention="sage_v3", turbo=True)),
    ("C02_refvid_twostep_lora_sage3_8", dict(kind="ref_video", mode="twostep", steps=8, attention="sage_v3", turbo=True)),
    ("C03_refvid_twostep_lora_sage3_20", dict(kind="ref_video", mode="twostep", steps=20, attention="sage_v3", turbo=True)),
]


def make_params(label: str, ov: dict) -> GenerateParams:
    kind = ov["kind"]
    if kind == "firstlast":
        prompt, image, last, refs, vids = COMPLEX_FL2VA, FIRST_FRAME, LAST_FRAME, [], []
    elif kind == "ref_images":
        prompt, image, last, refs, vids = COMPLEX_REF_IMG, "", None, list(REF_IMAGES), []
    elif kind == "ref_video":
        prompt, image, last, refs, vids = COMPLEX_REF_VID, "", None, [], [REF_VIDEO]
    else:
        raise ValueError(kind)

    return GenerateParams(
        mode=ov["mode"],
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=ov["steps"],
        seed=SEED,
        image=image,
        last_frame=last,
        ref_images=refs,
        ref_videos=vids,
        output_prefix=f"{OUT_ROOT}/{label}",
        sage_attention="disabled",
        attention_backend=ov["attention"],
        turbo=ov["turbo"],
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
        sampler="res_multistep",
        scheduler="simple",
    )


def run_one(base_url: str, label: str, ov: dict) -> dict:
    params = make_params(label, ov)
    params.validate()
    t0 = time.time()
    pid = submit_prompt(base_url, build_graph(params))
    entry = wait_prompt(base_url, pid, poll_interval=3.0, timeout=4 * 3600)
    elapsed = round(time.time() - t0, 2)
    st = entry.get("status") or {}
    outs = extract_outputs(entry)
    if st.get("status_str") == "error":
        raise ComfyUIError(f"{label}: {(st.get('messages') or [])[:3]}")
    return {
        "label": label,
        "kind": ov["kind"],
        "mode": params.mode,
        "steps": params.steps,
        "attention": ov["attention"],
        "turbo": params.turbo,
        "elapsed_sec": elapsed,
        "prompt_id": pid,
        "outputs": outs,
        "status": st.get("status_str"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_fl_ref_accel_1344x768.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    cases = CASES
    if args.only:
        want = set(args.only)
        cases = [(l, o) for l, o in CASES if l in want]

    print(f"Accel fl/ref {WIDTH}x{HEIGHT}: {len(cases)} cases → {OUT_ROOT}/", flush=True)
    results = []
    for label, ov in cases:
        print(f"\n=== {label} ===", flush=True)
        try:
            # dry validate graph
            build_graph(make_params(label, ov))
            r = run_one(base_url, label, ov)
            print(f"{label}: {r['elapsed_sec']}s", flush=True)
            results.append(r)
        except Exception as e:
            print(f"{label}: FAILED {e}", flush=True)
            results.append({"label": label, "status": "error", "error": str(e)})
        args.out.write_text(
            json.dumps(
                {
                    "config": {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "length": LENGTH,
                        "baseline_direct_pytorch": {
                            "14_fl2va_firstlast_8": 165.3,
                            "15_fl2va_firstlast_20": 337.77,
                            "16_ref2va_images_8": 180.77,
                            "17_ref2va_images_20": 330.12,
                            "18_ref2va_video_8": 486.2,
                            "19_ref2va_video_20": 1092.47,
                        },
                        "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
                    },
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    ok = sum(1 for r in results if r.get("status") == "success")
    print(f"\nWrote {args.out}  ({ok}/{len(results)} ok)")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
