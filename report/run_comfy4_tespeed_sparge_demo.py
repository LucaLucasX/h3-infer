#!/usr/bin/env python3
"""One-shot demo: 4-step TurboLoRA + TE-Speed + Sparge (no TeaCache)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from benchmark_cn_accel_12s import CN_VOICE_TEXT, LENGTH, FPS
from benchmark_official_1344x768 import HEIGHT, SEED, WIDTH
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node

STEPS = 4
OUT_ROOT = "exp_comfy4_tespeed_sparge"
LABEL = "demo_tespeed_sparge"


def build_demo_graph() -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=CN_VOICE_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        tespeed=True,
        teacache=False,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        image="",
        output_prefix=f"{OUT_ROOT}/{LABEL}",
    )
    g = build_graph(p)
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {"model": ["1", 0], "topk": 0.5, "dense_first_steps": 0, "num_layers": 50},
    )
    return g


def run_one(base_url: str, label: str, graph: dict) -> dict:
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
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    health(args.comfy)
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"comfy4 tespeed+sparge demo: {WIDTH}x{HEIGHT} length={LENGTH} steps={STEPS}",
        flush=True,
    )

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = run_one(args.comfy, "_warmup", build_demo_graph())
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup*"):
            p.unlink(missing_ok=True)

    print(f"=== {LABEL} ===", flush=True)
    r = run_one(args.comfy, LABEL, build_demo_graph())
    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "turbo": True,
            "tespeed": True,
            "teacache": False,
            "sparge_topk": 0.5,
            "attention": "sparge replaces sage node",
            "out": str(out_dir.relative_to(ROOT)),
        },
        "result": r,
    }
    meta = ROOT / "experiments" / "sparse_attn" / "output" / "comfy4_tespeed_sparge_demo.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"wrote {meta}", flush=True)
    if r.get("outputs"):
        for o in r["outputs"]:
            print(f"output: ComfyUI-master_cp/output/{OUT_ROOT}/{o.get('filename')}", flush=True)


if __name__ == "__main__":
    main()
