#!/usr/bin/env python3
"""4-step Turbo LoRA ablation @ 12s: baseline / TeaCache / Sparge / TC+Sparge."""

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
OUT_ROOT = "exp_4step_ablation_12s"

COMMON = dict(
    mode="t2v",
    prompt=CN_VOICE_TEXT,
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
)


def _base(label: str, **overrides) -> GenerateParams:
    kw = dict(COMMON)
    kw["output_prefix"] = f"{OUT_ROOT}/{label}"
    kw.update(overrides)
    return GenerateParams(**kw)


def _graph_plain(label: str, **overrides):
    return build_graph(_base(label, **overrides))


def _graph_sparge(label: str, **overrides):
    g = build_graph(_base(label, **overrides))
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {
            "model": ["1", 0],
            "topk": 0.5,
            "dense_first_steps": 0,
            "num_layers": 50,
        },
    )
    return g


CASES = [
    ("4s_01_baseline", lambda: _graph_plain("4s_01_baseline")),
    ("4s_02_teacache_only", lambda: _graph_plain("4s_02_teacache_only", teacache=True)),
    ("4s_03_sparge_only", lambda: _graph_sparge("4s_03_sparge_only")),
    (
        "4s_04_teacache_sparge",
        lambda: _graph_sparge("4s_04_teacache_sparge", teacache=True),
    ),
]


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
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    health(args.comfy)
    cases = CASES
    if args.only:
        allow = set(args.only)
        cases = [(l, fn) for l, fn in CASES if l in allow]

    print(
        f"4-step ablation 12s: {WIDTH}x{HEIGHT} length={LENGTH} steps={STEPS} cases={len(cases)}",
        flush=True,
    )
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = _run("_warmup", _graph_plain("_warmup"), args.comfy)
        print(json.dumps({k: w[k] for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))

    results = []
    for label, make in cases:
        print(f"=== {label} ===", flush=True)
        r = _run(label, make(), args.comfy)
        results.append(r)
        print(
            json.dumps(
                {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    base_t = next(
        (r["elapsed_sec"] for r in results if r["label"] == "4s_01_baseline" and r.get("elapsed_sec")),
        None,
    )
    summary = []
    for r in results:
        row = {"label": r["label"], "elapsed_sec": r.get("elapsed_sec"), "status": r.get("status")}
        if base_t and r.get("elapsed_sec") and r.get("status") == "success":
            row["speedup_vs_baseline"] = round(base_t / r["elapsed_sec"], 3)
            row["saved_sec_vs_baseline"] = round(base_t - r["elapsed_sec"], 2)
        summary.append(row)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "seed": SEED,
            "prompt": "Shanghai Bund rainy Mandarin dialogue (CN signs)",
            "comfy": args.comfy,
            "compare_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "stack": "TurboLoRA v4-600 + sage_v2_memeff + pruned_int8",
            "sparge": "topk=0.5 dense_first_steps=0",
            "teacache": "default start_step=2 end_step=-2",
        },
        "summary": summary,
        "results": results,
    }
    out = ROOT / "experiments" / "sparse_attn" / "output" / "ab_4step_ablation_12s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
