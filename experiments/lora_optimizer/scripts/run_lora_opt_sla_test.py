#!/usr/bin/env python3
"""SLA attention + Turbo LoRA merge via ComfyUI-LoRA-Optimizer."""
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
sys.path.insert(0, str(REPO / "experiments" / "sla"))
sys.path.insert(0, str(REPO / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node
from loras import LORA_PROD_T2V, LORA_SLA_T2V

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


def _rewire_model(
    g: dict,
    old_ref: list,
    new_ref: list,
    *,
    skip_nodes: set[str] | None = None,
) -> None:
    """Rewire model consumers; skip_nodes keeps a node's own inputs unchanged."""
    skip = skip_nodes or set()
    for nid, nd in g.items():
        if nid in skip:
            continue
        inputs = nd.get("inputs") or {}
        for k, v in list(inputs.items()):
            if v == old_ref:
                inputs[k] = new_ref


def _patch_sla_after_build(g: dict) -> dict:
    g["5"] = node(
        "MiniMaxH3SLAAttnPatchExp",
        {
            "model": ["1", 0],
            "sparsity_ratio": float(SLA_SPARSITY),
            "dense_first_steps": 0,
            "num_layers": 50,
            "audio_dense": True,
        },
    )
    _rewire_model(g, ["1", 0], ["5", 0], skip_nodes={"5"})
    return g


def graph_sla_single_lora(out_root: str, label: str, prompt: str, lora: str) -> dict:
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
        turbo_lora=lora,
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    return _patch_sla_after_build(g)


def graph_sla_lora_opt_merged(out_root: str, label: str, prompt: str) -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=False,
        attention_backend="sage_v2_memeff",
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    _patch_sla_after_build(g)

    g["22"] = node(
        "LoRAStack",
        {
            "lora_name": LORA_PROD_T2V,
            "strength": 1.0,
            "conflict_mode": "all",
            "key_filter": "all",
            "preserve": True,
        },
    )
    g["23"] = node(
        "LoRAStack",
        {
            "lora_name": LORA_SLA_T2V,
            "strength": 1.0,
            "conflict_mode": "all",
            "key_filter": "all",
            "preserve": True,
            "lora_stack": ["22", 0],
        },
    )
    g["24"] = node(
        "LoRAOptimizer",
        {
            "model": ["5", 0],
            "lora_stack": ["23", 0],
            "output_strength": 1.0,
            "clip_strength_multiplier": 1.0,
            "auto_strength": "enabled",
            "auto_strength_floor": -1.0,
            "free_vram_between_passes": "disabled",
            "vram_budget": 0.0,
            "optimization_mode": "additive",
            "cache_patches": "enabled",
            "patch_compression": "smart",
            "svd_device": "gpu",
            "normalize_keys": "enabled",
            "sparsification": "disabled",
            "sparsification_density": 0.7,
            "dare_dampening": 0.0,
            "merge_refinement": "none",
            "strategy_set": "full",
            "architecture_preset": "auto",
            "merge_strategy_override": "",
            "settings_source": "manual",
        },
    )
    _rewire_model(g, ["5", 0], ["24", 0], skip_nodes={"5", "24"})
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8196")
    ap.add_argument("--only", nargs="*", default=["o02", "o03"])
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["merged", "prod", "sla"],
        choices=["merged", "prod", "sla"],
    )
    ap.add_argument("--out-root", default="exp_lora_opt_sla")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    health(args.comfy)
    all_cases = load_cases()
    allow = set(args.only)
    cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]
    if not cases:
        raise SystemExit("no cases matched")

    out_root = args.out_root
    log_path = EXP / "logs" / f"{out_root}.json"
    out_dir = REPO / "ComfyUI-master_cp" / "output" / out_root

    print(
        f"LoRA-opt SLA t2v {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} "
        f"variants={args.variants} n={len(cases)}",
        flush=True,
    )

    builders = {
        "merged": lambda oroot, label, prompt: graph_sla_lora_opt_merged(oroot, label, prompt),
        "prod": lambda oroot, label, prompt: graph_sla_single_lora(
            oroot, label, prompt, LORA_PROD_T2V
        ),
        "sla": lambda oroot, label, prompt: graph_sla_single_lora(
            oroot, label, prompt, LORA_SLA_T2V
        ),
    }

    results = []
    for variant in args.variants:
        build = builders[variant]
        print(f"\n######## variant={variant} ########", flush=True)
        if not args.skip_warmup:
            wlabel = f"_warmup_{variant}"
            print(f"=== warmup {variant} ===", flush=True)
            w = run_one(
                args.comfy,
                build(out_root, wlabel, "warmup prompt, ignore content."),
                wlabel,
            )
            print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))

        for c in cases:
            label = f"{variant}_{c['id']}_{_slug(c['title'])}"
            print(f"=== {label} ===", flush=True)
            r = run_one(args.comfy, build(out_root, label, c["prompt"]), label)
            r.update({"variant": variant, "case_id": c["id"], "title": c["title"]})
            results.append(r)
            print(
                json.dumps(
                    {
                        "label": r.get("label"),
                        "status": r.get("status"),
                        "elapsed_sec": r.get("elapsed_sec"),
                        "error": r.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {}
    for variant in args.variants:
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
            "variants": args.variants,
            "sla_sparsity": SLA_SPARSITY,
            "prod_lora": LORA_PROD_T2V,
            "sla_lora": LORA_SLA_T2V,
            "resolution": [WIDTH, HEIGHT],
            "length": LENGTH,
            "steps": STEPS,
        },
        "summary": summary,
        "results": results,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {log_path}", flush=True)
    print(f"videos: {out_dir}", flush=True)
    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
