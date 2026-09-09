#!/usr/bin/env python3
"""12s SpargeAttn / LoSA-frozen-mask A/B vs Sage2 MemEff (EXP :8190)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from benchmark_accel_single_ab_12s import COMPLEX_VOICE_TEXT, FPS, LENGTH, STEPS
from benchmark_official_1344x768 import HEIGHT, SEED, WIDTH
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node

EXP_OUT = "exp_sparse_attn_12s"
WARMUP_LABEL = "_warmup_sage2_12s_discard"


def _base(label: str) -> GenerateParams:
    return GenerateParams(
        mode="t2v",
        prompt=COMPLEX_VOICE_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
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
    return build_graph(_base(label))


def _graph_sparge(label: str, topk: float):
    g = build_graph(_base(label))
    g["5"] = node("MiniMaxH3SpargeAttnPatchExp", {"model": ["1", 0], "topk": float(topk)})
    return g


def _graph_losa(label: str, mass: float, profile_steps: int, num_layers: int):
    g = build_graph(_base(label))
    g["5"] = node(
        "MiniMaxH3LoSAApproxPatchExp",
        {
            "model": ["1", 0],
            "mass_thresh": float(mass),
            "profile_steps": int(profile_steps),
            "num_layers": int(num_layers),
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


def _delete_prefix(prefix: str) -> None:
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / EXP_OUT
    if not out_dir.is_dir():
        return
    for p in out_dir.glob(f"{prefix}*"):
        p.unlink(missing_ok=True)
        print(f"deleted {p.name}", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--topk", type=float, default=0.5)
    ap.add_argument("--losa-mass", type=float, default=0.99)
    ap.add_argument("--num-layers", type=int, default=50)
    args = ap.parse_args()

    health(args.comfy)
    print(
        f"12s sparse A/B: {WIDTH}x{HEIGHT} length={LENGTH} turbo={STEPS} "
        f"sparge_topk={args.topk} losa_mass={args.losa_mass}",
        flush=True,
    )

    if not args.skip_warmup:
        print("=== warmup sage2 (discard) ===", flush=True)
        w = _run(WARMUP_LABEL, _graph_sage(WARMUP_LABEL), args.comfy)
        print(json.dumps({k: w[k] for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        _delete_prefix(WARMUP_LABEL)

    cases = [
        ("01_sage2_memeff", lambda: _graph_sage("01_sage2_memeff")),
        (f"02_sparge_topk{args.topk:g}", lambda: _graph_sparge(f"02_sparge_topk{args.topk:g}", args.topk)),
        (
            f"03_losa_frozen_m{args.losa_mass:g}",
            lambda: _graph_losa(
                f"03_losa_frozen_m{args.losa_mass:g}", args.losa_mass, 1, args.num_layers
            ),
        ),
    ]
    results = []
    for label, make in cases:
        print(f"=== {label} ===", flush=True)
        r = _run(label, make(), args.comfy)
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))

    baseline = next(
        (r["elapsed_sec"] for r in results if r["label"].startswith("01_") and r.get("elapsed_sec")),
        None,
    )
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
            "losa_num_layers": args.num_layers,
            "losa": "frozen block_map via get_block_map_meansim(cdfthreshd=mass) + block_sparse_sage2",
            "compare_dir": f"ComfyUI-master_cp/output/{EXP_OUT}",
        },
        "summary": summary,
        "results": results,
    }
    out = ROOT / "experiments" / "sparse_attn" / "output" / "ab_sparge_losa_12s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
