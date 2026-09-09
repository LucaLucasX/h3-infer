#!/usr/bin/env python3
"""A/B wall-clock with COMFY_KITCHEN_FORCE_CUDA (expects ComfyUI already started with that env).

Re-runs a few people-suite cases for comparison against benchmark_compare_720p5s_people.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import TURBO_LORA_V4, GenerateParams, build_graph

# Same prompt as people suite
from benchmark_compare_720p5s_people import COMPLEX_PEOPLE_PROMPT, WIDTH, HEIGHT, LENGTH, SEED

OUT_ROOT = "h3_force_cuda_ab"
CASES: list[tuple[str, dict]] = [
    ("01_direct_8", dict(mode="t2v", steps=8, attention_backend="pytorch", turbo=False)),
    ("12_direct_lora_sage2_8", dict(mode="t2v", steps=8, attention_backend="sage_v2_memeff", turbo=True)),
    ("08_twostep_lora_sage2_8", dict(mode="twostep", steps=8, attention_backend="sage_v2_memeff", turbo=True)),
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
        image="",
        output_prefix=f"{OUT_ROOT}/{label}",
        sage_attention="disabled",
        attention_backend=overrides["attention_backend"],
        turbo=overrides["turbo"],
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
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
        "attention": overrides["attention_backend"],
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
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_force_cuda_ab.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)
    force = os.environ.get("COMFY_KITCHEN_FORCE_CUDA", "(client-side unknown; check ComfyUI process)")
    print(f"COMFY_KITCHEN_FORCE_CUDA (client env)={force}")
    print(f"Running {len(CASES)} cases → {OUT_ROOT}/")

    results = []
    for label, ov in CASES:
        print(f"\n=== {label} ===", flush=True)
        r = run_one(base_url, label, ov)
        print(f"{label}: {r['elapsed_sec']}s", flush=True)
        results.append(r)

    payload = {
        "config": {
            "note": "ComfyUI must be started with COMFY_KITCHEN_FORCE_CUDA=1",
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "timing": "wall-clock submit→history done (incl. TE/VAE/IO)",
            "baseline_json": "benchmark_compare_720p5s_people.json",
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
