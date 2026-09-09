#!/usr/bin/env python3
"""A/B: LoRAOptimizer node (merged) vs baked TurboLoRA on the same Comfy instance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import health
from loras import LORA_OPTIMIZER_BAKED

from run_lora_opt_sla_test import (
    SLA_SPARSITY,
    STEPS,
    _slug,
    graph_sla_lora_opt_merged,
    load_cases,
    run_one,
)
from run_offline_merge_test import LENGTH, WIDTH, HEIGHT, graph_turbo_sla


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8196")
    ap.add_argument("--only", nargs="*", default=["o02"])
    ap.add_argument("--out-root", default="exp_lora_opt_ab_gpu1")
    ap.add_argument("--sla-sparsity", type=float, default=SLA_SPARSITY)
    args = ap.parse_args()

    health(args.comfy)
    allow = set(args.only)
    cases = [c for c in load_cases() if c["id"] in allow or c["title"] in allow]
    if not cases:
        raise SystemExit("no cases matched")

    out_root = args.out_root
    out_dir = REPO / "ComfyUI-master_cp" / "output" / out_root
    meta = EXP / "logs" / f"{out_root}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"A/B GPU1 t2v {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} "
        f"variants=['merged_node','optimizer_baked'] n={len(cases)}",
        flush=True,
    )

    results = []
    for c in cases:
        slug = _slug(c["title"])
        for variant, builder in (
            ("merged_node", lambda label, prompt: graph_sla_lora_opt_merged(out_root, label, prompt)),
            (
                "optimizer_baked",
                lambda label, prompt: graph_turbo_sla(
                    out_root,
                    label,
                    prompt,
                    LORA_OPTIMIZER_BAKED,
                    sparsity=args.sla_sparsity,
                ),
            ),
        ):
            label = f"{variant}_{c['id']}_{slug}"
            print(f"=== {label} ===", flush=True)
            r = run_one(args.comfy, builder(label, c["prompt"]), label)
            r.update({"variant": variant, "case_id": c["id"], "title": c["title"]})
            results.append(r)
            print(
                json.dumps(
                    {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {}
    for variant in ("merged_node", "optimizer_baked"):
        secs = [
            r["elapsed_sec"]
            for r in results
            if r.get("variant") == variant and r.get("status") == "success" and r.get("elapsed_sec")
        ]
        summary[variant] = {
            "n_ok": sum(1 for r in results if r.get("variant") == variant and r.get("status") == "success"),
            "mean_sec": round(sum(secs) / len(secs), 2) if secs else None,
            "secs": secs,
        }

    payload = {
        "config": {
            "comfy": args.comfy,
            "sla_sparsity": args.sla_sparsity,
            "baked_lora": LORA_OPTIMIZER_BAKED,
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
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
