#!/usr/bin/env python3
"""Isolated SpargeAttn / LoSA-approx A/B vs Sage2 MemEff (short clip).

Uses experimental ComfyUI :8190. Does not change h3_graph defaults.
Outputs under experiments/sparse_attn/output/ (and Comfy output subfolder).
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
from h3_graph import GenerateParams, build_graph, node

EXP_OUT = "exp_sparse_attn"
WIDTH, HEIGHT = 864, 480
LENGTH = 124  # ~5s
STEPS = 8
SEED = 42
PROMPT = (
    "Cinematic rainy night street, woman with umbrella speaking clearly in Chinese: "
    '"对不起，我迟到了。" soft rain and jazz, photorealistic, 24fps.'
)


def _base_params(label: str) -> GenerateParams:
    return GenerateParams(
        mode="t2v",
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        image="",
        output_prefix=f"{EXP_OUT}/{label}",
    )


def _graph_sage(label: str):
    return build_graph(_base_params(label))


def _graph_sparge(label: str, topk: float):
    g = build_graph(_base_params(label))
    # Replace Sage2 MemEff node (id 5) with EXP Sparge patch; keep MODEL wire from UNET.
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {"model": ["1", 0], "topk": float(topk)},
    )
    return g


def _graph_losa(label: str, mass: float, profile_steps: int):
    g = build_graph(_base_params(label))
    g["5"] = node(
        "MiniMaxH3LoSAApproxPatchExp",
        {
            "model": ["1", 0],
            "mass_thresh": float(mass),
            "profile_steps": int(profile_steps),
        },
    )
    return g


def _run(label: str, graph: dict, base_url: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        return {
            "label": label,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:4000],
        }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190", help="EXP ComfyUI (default :8190)")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--topk", type=float, default=0.5)
    ap.add_argument("--losa-mass", type=float, default=0.99)
    args = ap.parse_args()

    health(args.comfy)
    results = []

    if not args.skip_warmup:
        print("=== warmup sage2 (discard timing) ===", flush=True)
        w = _run("_warmup_sage2", _graph_sage("_warmup_sage2"), args.comfy)
        print(json.dumps({k: w[k] for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        # delete warmup files
        out_dir = ROOT / "ComfyUI-master_cp" / "output" / EXP_OUT
        if out_dir.is_dir():
            for p in out_dir.glob("_warmup_sage2*"):
                p.unlink(missing_ok=True)
                print(f"deleted {p.name}", flush=True)

    cases = [
        ("01_sage2_memeff", lambda: _graph_sage("01_sage2_memeff")),
        (f"02_sparge_topk{args.topk:g}", lambda: _graph_sparge(f"02_sparge_topk{args.topk:g}", args.topk)),
        (
            f"03_losa_approx_m{args.losa_mass:g}",
            lambda: _graph_losa(f"03_losa_approx_m{args.losa_mass:g}", args.losa_mass, 1),
        ),
    ]
    for label, make in cases:
        print(f"=== {label} ===", flush=True)
        r = _run(label, make(), args.comfy)
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))

    baseline = next((r["elapsed_sec"] for r in results if r["label"].startswith("01_") and r.get("elapsed_sec")), None)
    summary = []
    for r in results:
        row = {"label": r["label"], "elapsed_sec": r.get("elapsed_sec"), "status": r.get("status")}
        if baseline and r.get("elapsed_sec"):
            row["speedup_vs_sage2"] = round(baseline / r["elapsed_sec"], 3)
        summary.append(row)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "seed": SEED,
            "turbo": True,
            "comfy": args.comfy,
            "sparge_topk": args.topk,
            "losa_mass": args.losa_mass,
            "note": "EXP only; h3_graph defaults unchanged",
        },
        "summary": summary,
        "results": results,
    }
    out = ROOT / "experiments" / "sparse_attn" / "output" / "ab_sparge_losa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
