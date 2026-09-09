#!/usr/bin/env python3
"""Run baked Turbo LoRA via MiniMaxH3TurboLoRA (+ optional SLA attn A/B)."""
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
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node
from loras import LORA_MERGE_PROD07_SLA03, LORA_OPTIMIZER_BAKED, LORA_PROD_T2V, LORA_SLA_T2V

WIDTH, HEIGHT = 1344, 768
LENGTH = 288
FPS = 24.0
STEPS = 4
SEED = 42
SLA_SPARSITY = 0.85


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def load_cases() -> list[dict]:
    from run_textcases_comfy4_tc_sparge import load_cases as _load

    return _load()


def graph_turbo_sla(
    out_root: str,
    label: str,
    prompt: str,
    turbo_lora: str,
    *,
    sparsity: float = SLA_SPARSITY,
    strength: float = 1.0,
) -> dict:
    """Sage2 MemEff + SLA attn + single TurboLoRA file (offline merge or baseline)."""
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
        turbo_lora=turbo_lora,
        turbo_strength=strength,
        attention_backend="sage_v2_memeff",
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    g["5"] = node(
        "MiniMaxH3SLAAttnPatchExp",
        {
            "model": ["1", 0],
            "sparsity_ratio": float(sparsity),
            "dense_first_steps": 0,
            "num_layers": 50,
            "audio_dense": True,
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
    ap.add_argument("--only", nargs="*", default=["o02", "o03", "o04"])
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["optimizer_baked"],
        choices=["optimizer_baked", "fixed_merge", "prod", "sla"],
    )
    ap.add_argument("--sla-sparsity", type=float, default=SLA_SPARSITY)
    ap.add_argument("--out-root", default="exp_lora_optimizer_baked")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    builders = {
        "optimizer_baked": (LORA_OPTIMIZER_BAKED, "optimizer_baked"),
        "fixed_merge": (LORA_MERGE_PROD07_SLA03, "fixed_merge"),
        "prod": (LORA_PROD_T2V, "prod"),
        "sla": (LORA_SLA_T2V, "sla"),
    }

    health(args.comfy)
    all_cases = load_cases()
    cases = all_cases
    if args.only:
        allow = set(args.only)
        cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]

    out_root = args.out_root
    out_dir = REPO / "ComfyUI-master_cp" / "output" / out_root
    meta = EXP / "logs" / f"{out_root}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"offline-merge t2v {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} "
        f"variants={args.variants} sla_sparsity={args.sla_sparsity} n={len(cases)}",
        flush=True,
    )

    results = []
    for variant in args.variants:
        lora, tag = builders[variant]
        print(f"\n######## variant={variant} lora={lora} ########", flush=True)

        if not args.skip_warmup:
            warmup_prompt = next(
                (c["prompt"] for c in all_cases if c["prompt"] not in {x["prompt"] for x in cases}),
                "warmup only, ignore content.",
            )
            wlabel = f"_warmup_{tag}"
            print(f"=== warmup {tag} (discard) ===", flush=True)
            w = run_one(
                args.comfy,
                graph_turbo_sla(out_root, wlabel, warmup_prompt, lora, sparsity=args.sla_sparsity),
                wlabel,
            )
            print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))
            for p in out_dir.glob(f"_warmup_{tag}*"):
                p.unlink(missing_ok=True)

        for c in cases:
            label = f"{tag}_{c['id']}_{_slug(c['title'])}"
            print(f"=== {label} ===", flush=True)
            r = run_one(
                args.comfy,
                graph_turbo_sla(out_root, label, c["prompt"], lora, sparsity=args.sla_sparsity),
                label,
            )
            r.update({
                "variant": variant,
                "case_id": c["id"],
                "title": c["title"],
                "lora": lora,
            })
            results.append(r)
            print(
                json.dumps(
                    {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {}
    for variant in args.variants:
        ok = [
            r["elapsed_sec"]
            for r in results
            if r.get("variant") == variant and r.get("status") == "success" and r.get("elapsed_sec")
        ]
        summary[variant] = {
            "n_ok": len(ok),
            "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
            "secs": ok,
        }

    payload = {
        "config": {
            "comfy": args.comfy,
            "variants": args.variants,
            "sla_sparsity": args.sla_sparsity,
            "optimizer_baked_lora": LORA_OPTIMIZER_BAKED,
            "resolution": [WIDTH, HEIGHT],
            "length": LENGTH,
            "steps": STEPS,
        },
        "summary": summary,
        "results": results,
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {meta}", flush=True)
    print(f"videos: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
