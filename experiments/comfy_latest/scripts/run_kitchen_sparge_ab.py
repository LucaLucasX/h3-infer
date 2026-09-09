#!/usr/bin/env python3
"""A/B on ComfyUI-latest: 4-step LoRA + Kitchen, then +Sparge.

Kitchen is enabled globally via --use-ck-attention on the server.
Variant kitchen: no attention patch (DiT uses Kitchen).
Variant kitchen_sparge: Sparge patch replaces DiT self-attn (prod topk=0.5).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "sla"))
sys.path.insert(0, str(REPO / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from loras import LORA_PROD_T2V

WIDTH, HEIGHT = 1344, 768
LENGTH = 288
FPS = 24.0
STEPS = 4
SEED = 42
SPARGE_TOPK = 0.5


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def load_cases():
    from run_textcases_comfy4_tc_sparge import load_cases as _load

    return _load()


def graph_kitchen(out_root: str, label: str, prompt: str) -> dict:
    """4-step prod LoRA; attention = global Kitchen (no Sage/Sparge node)."""
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        turbo_lora=LORA_PROD_T2V,
        turbo_strength=1.0,
        attention_backend="pytorch",  # no Sage patch; Kitchen via CLI
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    )
    return build_graph(p)


def graph_kitchen_sparge(out_root: str, label: str, prompt: str) -> dict:
    """4-step prod LoRA + Sparge (topk=0.5). Sparge overrides DiT self-attn."""
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        turbo_lora=LORA_PROD_T2V,
        turbo_strength=1.0,
        attention_backend="pytorch",  # Sparge node owns DiT attn; dense path uses Sage2
        sparge=True,
        sparge_topk=SPARGE_TOPK,
        sparge_dense_first_steps=0,
        sparge_num_layers=50,
        sparge_audio_dense=True,
        output_prefix=f"{out_root}/{label}",
    )
    return build_graph(p)


def _profiling(hist: dict, submitted_at: float) -> dict:
    result = {
        "submitted_at": round(submitted_at, 3),
        "queue_sec": None,
        "generation_sec": None,
        "total_sec": None,
    }
    messages = (hist.get("status") or {}).get("messages", [])
    ts = {}
    for m in messages:
        if isinstance(m, list) and len(m) > 1 and isinstance(m[1], dict):
            if isinstance(m[1].get("timestamp"), (int, float)):
                ts[m[0]] = m[1]["timestamp"] / 1000.0
    started, completed = ts.get("execution_start"), ts.get("execution_success")
    if started is not None:
        result["queue_sec"] = round(max(0.0, started - submitted_at), 3)
    if completed is not None and started is not None:
        result["generation_sec"] = round(max(0.0, completed - started), 3)
        result["total_sec"] = round(max(0.0, completed - submitted_at), 3)
    return result


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        submitted_at = time.time()
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
            "profiling": _profiling(hist, submitted_at),
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
    ap.add_argument("--comfy", default="http://127.0.0.1:8192")
    ap.add_argument("--only", nargs="*", default=["o02", "o03", "o04"])
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["kitchen", "kitchen_sparge"],
        choices=["kitchen", "kitchen_sparge"],
    )
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--out-root", default="exp_kitchen_sparge_ab")
    args = ap.parse_args()

    out_dir = REPO / "ComfyUI-master_cp" / "output" / args.out_root
    # also mirror into master_cp output path? keep under latest
    meta = EXP / "logs" / f"{args.out_root}.json"
    health(args.comfy)

    all_cases = load_cases()
    cases = [c for c in all_cases if c["id"] in set(args.only) or c["title"] in set(args.only)]
    timed = {c["prompt"] for c in cases}
    warmup_prompt = next(
        (c["prompt"] for c in all_cases if c["prompt"] not in timed),
        "warmup only, ignore content.",
    )

    builders = {
        "kitchen": graph_kitchen,
        "kitchen_sparge": graph_kitchen_sparge,
    }
    print(
        f"Kitchen A/B {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} "
        f"variants={args.variants} n={len(cases)} lora={LORA_PROD_T2V}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in args.variants:
        build = builders[variant]
        print(f"\n######## variant={variant} ########", flush=True)
        if not args.skip_warmup:
            print(f"=== warmup {variant} ===", flush=True)
            w = run_one(args.comfy, build(args.out_root, f"_warmup_{variant}", warmup_prompt), f"_warmup_{variant}")
            print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))
            for p in out_dir.glob(f"_warmup_{variant}*"):
                p.unlink(missing_ok=True)

        for c in cases:
            label = f"{variant}_{c['id']}_{_slug(c['title'])}"
            print(f"=== {label} ===", flush=True)
            r = run_one(args.comfy, build(args.out_root, label, c["prompt"]), label)
            r.update({"variant": variant, "case_id": c["id"], "title": c["title"], "lora": LORA_PROD_T2V})
            results.append(r)
            print(
                json.dumps(
                    {
                        "label": r.get("label"),
                        "status": r.get("status"),
                        "elapsed_sec": r.get("elapsed_sec"),
                        "generation_sec": (r.get("profiling") or {}).get("generation_sec"),
                        "error": r.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {}
    for variant in args.variants:
        rows = [r for r in results if r.get("variant") == variant and r.get("status") == "success"]
        gens = [(r.get("profiling") or {}).get("generation_sec") for r in rows]
        gens = [g for g in gens if g is not None]
        summary[variant] = {
            "n_ok": len(rows),
            "mean_sec": round(sum(r["elapsed_sec"] for r in rows) / len(rows), 2) if rows else None,
            "mean_generation_sec": round(sum(gens) / len(gens), 3) if gens else None,
            "generation_secs": gens,
        }
    payload = {"summary": summary, "results": results, "config": vars(args)}
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {meta}")
    print(f"videos: {out_dir}")


if __name__ == "__main__":
    main()
