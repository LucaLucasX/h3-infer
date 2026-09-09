#!/usr/bin/env python3
"""fl2va (first+last frame) + ref2va complex-scene 720P 5s timing (8 / 20 steps)."""

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
from h3_graph import GenerateParams, build_graph
from benchmark_compare_720p5s_people import COMPLEX_PEOPLE_PROMPT, WIDTH, HEIGHT, LENGTH, SEED

OUT_ROOT = "h3_compare_720p5s_fl2va_ref2va"

# first / last for fl2va; refs for ref2va (under ComfyUI input/)
FIRST_FRAME = "example.png"
LAST_FRAME = "i1.png"
REF_IMAGES = ["example.png", "i1.png"]

CASES: list[tuple[str, dict]] = [
    (
        "01_fl2va_firstlast_8",
        dict(mode="i2v", steps=8, image=FIRST_FRAME, last_frame=LAST_FRAME, ref_images=[]),
    ),
    (
        "02_fl2va_firstlast_20",
        dict(mode="i2v", steps=20, image=FIRST_FRAME, last_frame=LAST_FRAME, ref_images=[]),
    ),
    (
        "03_ref2va_8",
        dict(mode="r2v", steps=8, image="", last_frame=None, ref_images=REF_IMAGES),
    ),
    (
        "04_ref2va_20",
        dict(mode="r2v", steps=20, image="", last_frame=None, ref_images=REF_IMAGES),
    ),
]


def make_params(label: str, overrides: dict) -> GenerateParams:
    return GenerateParams(
        mode=overrides["mode"],
        prompt=COMPLEX_PEOPLE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=overrides["steps"],
        seed=SEED,
        image=overrides.get("image") or "",
        last_frame=overrides.get("last_frame"),
        ref_images=list(overrides.get("ref_images") or []),
        output_prefix=f"{OUT_ROOT}/{label}",
        sage_attention="disabled",
        attention_backend="pytorch",
        turbo=False,
        sampler="res_multistep",
        scheduler="simple",
    )


def run_one(base_url: str, label: str, overrides: dict) -> dict:
    params = make_params(label, overrides)
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
        "mode": params.mode,
        "steps": params.steps,
        "first_frame": params.image or None,
        "last_frame": params.last_frame,
        "ref_images": params.ref_images,
        "attention": "pytorch",
        "turbo": False,
        "elapsed_sec": elapsed,
        "prompt_id": pid,
        "outputs": outs,
        "output_prefix": params.output_prefix,
        "status": st.get("status_str"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_fl2va_ref2va_720p5s.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    cases = CASES
    if args.only:
        want = set(args.only)
        cases = [(l, o) for l, o in CASES if l in want]
        if not cases:
            raise SystemExit(f"no cases match --only {args.only}")

    print(f"Running {len(cases)} cases → {OUT_ROOT}/ (720P {LENGTH}f ≈5s, complex people prompt)")
    results = []
    for label, ov in cases:
        print(f"\n=== {label} ===", flush=True)
        r = run_one(base_url, label, ov)
        print(f"{label}: {r['elapsed_sec']}s", flush=True)
        results.append(r)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "prompt_theme": "complex multi-character rainy Tokyo night",
            "fl2va_first": FIRST_FRAME,
            "fl2va_last": LAST_FRAME,
            "ref2va_refs": REF_IMAGES,
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "timing": "wall-clock submit→history done (incl. TE/VAE/IO)",
            "attention": "pytorch",
            "turbo": False,
        },
        "prompt": COMPLEX_PEOPLE_PROMPT,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
