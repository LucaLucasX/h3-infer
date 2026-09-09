#!/usr/bin/env python3
"""Run high-contrast none vs effect-embedding pairs on selected t2v cases."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
REPO = DEV.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DEV))
sys.path.insert(0, str(REPO / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node
from loras import LORA_T2V

# Import case loader + t2v graph defaults from run_t2v
from run_t2v import WIDTH, HEIGHT, LENGTH, STEPS, SEED, SPARGE_TOPK, AUDIO_DENSE, TURBO_LORA, load_cases, _slug

# Case id -> effect preset (chosen for semantic / motion overlap with prompt)
FX_PAIRS: list[tuple[str, str, str]] = [
    ("o01", "blooming_flowers", "遍地发光雏菊/花瓣消散 — 花开特效"),
    ("o17", "bullet_time", "时间冻结街景 — 子弹时间"),
    ("o19", "bullet_time", "世界武术对决 — 子弹时间"),
    ("o18", "spiral_ascent", "飞越古伦敦 FPV — 螺旋上升运镜"),
    ("h01", "storm_magic", "白发少女召水龙雷暴 — 风暴魔法"),
]


def graph_for(out_root: str, label: str, prompt: str, effect: str) -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
        effect_embedding=effect,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=24.0,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        turbo_lora=TURBO_LORA,
        attention_backend="sage_v2_memeff",
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {
            "model": ["1", 0],
            "topk": float(SPARGE_TOPK),
            "dense_first_steps": 0,
            "num_layers": 50,
            "audio_dense": bool(AUDIO_DENSE),
        },
    )
    return g


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base, pid)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--out-root", default="exp_fx_compare")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    case_by_id = {c["id"]: c for c in load_cases()}
    health(args.comfy)
    out_root = args.out_root
    meta = DEV / "logs" / f"{out_root}.json"
    results: list[dict] = []

    if not args.skip_warmup:
        warm = case_by_id[FX_PAIRS[0][0]]
        print("=== warmup ===", flush=True)
        run_one(args.comfy, graph_for(out_root, "_warmup", warm["prompt"], "none"), "_warmup")

    for case_id, effect, reason in FX_PAIRS:
        c = case_by_id.get(case_id)
        if c is None:
            print(f"skip missing case {case_id}", flush=True)
            continue
        slug = _slug(c["title"])
        print(f"\n=== {case_id} {c['title']} | {reason} ===", flush=True)
        for fx in ("none", effect):
            label = f"{case_id}_{slug}_{fx}"
            print(f"--- effect={fx} ---", flush=True)
            r = run_one(args.comfy, graph_for(out_root, label, c["prompt"], fx), label)
            r.update({
                "case_id": case_id,
                "title": c["title"],
                "effect": fx,
                "paired_effect": effect,
                "reason": reason,
            })
            results.append(r)
            print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False), flush=True)

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success"]
    payload = {"pairs": [{"case": a, "effect": b, "reason": c} for a, b, c in FX_PAIRS], "n": len(results), "n_ok": sum(1 for r in results if r.get("status") == "success"), "mean_sec": round(sum(ok) / len(ok), 2) if ok else None, "results": results}
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "n_ok", "mean_sec")}, ensure_ascii=False))
    print(f"wrote {meta}", flush=True)


if __name__ == "__main__":
    main()
