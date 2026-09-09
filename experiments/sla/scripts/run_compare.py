#!/usr/bin/env python3
"""A/B: production Sparge+v1.0 LoRA vs SLA+SLA LoRA (same canvas/steps/sampler)."""
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
from loras import LORA_PROD_T2V, LORA_SLA_T2V

WIDTH, HEIGHT = 1344, 768
LENGTH = 288
FPS = 24.0
STEPS = 4
SEED = 42
SPARGE_TOPK = 0.5
SLA_SPARSITY = 0.85  # overridden by --sla-sparsity


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def load_cases() -> list[dict]:
    from run_textcases_comfy4_tc_sparge import load_cases as _load

    return _load()


def graph_sparge(out_root: str, label: str, prompt: str) -> dict:
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
        attention_backend="sage_v2_memeff",
        sparge=True,
        sparge_topk=SPARGE_TOPK,
        sparge_dense_first_steps=0,
        sparge_num_layers=50,
        sparge_audio_dense=True,
        output_prefix=f"{out_root}/{label}",
    )
    return build_graph(p)


def graph_sla(
    out_root: str,
    label: str,
    prompt: str,
    sparsity: float = SLA_SPARSITY,
    turbo_lora: str = LORA_SLA_T2V,
) -> dict:
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
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    # Overwrite Sage node 5 with SLA (LoRA already consumes [5,0]).
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


def graph_sol(
    out_root: str,
    label: str,
    prompt: str,
    turbo_lora: str = LORA_PROD_T2V,
    tau: float = 1.3,
) -> dict:
    """Sol-Attn on top of Sage2 MemEff (prod LoRA). No Sparge/SLA."""
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
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    # Tip after build is usually LoRA node 21 consuming Sage node 5.
    # Node ids 50+ are used by noise/sampler — pick a free id for Sol.
    tip = "21" if "21" in g else "5"
    sol_id = "70"
    g[sol_id] = node(
        "SolAttnPatch",
        {
            "model": [tip, 0],
            "tau": float(tau),
            "start_percent": 0.2,
            "end_percent": 0.9,
            "min_tokens": 4096,
            "int8_qk": True,
            "sink_conditioning": "exact_kv_and_rows",
            "morton": False,
            "morton_curve": "2d_frame",
            "verbose": True,
            "use_tma": False,
        },
    )
    for nid, nd in g.items():
        if nid == sol_id:
            continue
        inputs = nd.get("inputs") or {}
        for k, v in list(inputs.items()):
            if v == [tip, 0]:
                inputs[k] = [sol_id, 0]
    return g


def _profiling_timing(hist: dict, submitted_at: float) -> dict:
    """Extract queue/gen/total timing from Comfy history messages."""
    result = {
        "submitted_at": round(submitted_at, 3),
        "started_at": None,
        "completed_at": None,
        "queue_sec": None,
        "generation_sec": None,
        "total_sec": None,
    }
    messages = (hist.get("status") or {}).get("messages", [])
    ts = {}
    for m in messages:
        if (
            isinstance(m, list)
            and len(m) > 1
            and isinstance(m[1], dict)
            and isinstance(m[1].get("timestamp"), (int, float))
        ):
            ts[m[0]] = m[1]["timestamp"] / 1000.0
    started = ts.get("execution_start")
    completed = ts.get("execution_success")
    if started is not None:
        result["started_at"] = round(started, 3)
        result["queue_sec"] = round(max(0.0, started - submitted_at), 3)
    if completed is not None:
        result["completed_at"] = round(completed, 3)
        if started is not None:
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
        timing = _profiling_timing(hist, submitted_at)
        return {
            "label": label,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "profiling": timing,
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
    ap.add_argument("--comfy", default="http://127.0.0.1:8191")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*", default=["o02", "o03", "o04"])
    ap.add_argument("--variants", nargs="*", default=["sparge", "sla"],
                    choices=["sparge", "sla", "sol"])
    ap.add_argument("--sla-sparsity", type=float, default=SLA_SPARSITY)
    ap.add_argument(
        "--sla-lora",
        choices=["sla", "prod"],
        default="sla",
        help="sla=Turbo-SLA LoRA; prod=production v1.0 h3keys LoRA",
    )
    ap.add_argument("--sol-tau", type=float, default=1.3,
                    help="Sol-Attn tau (higher = sparser/faster)")
    ap.add_argument("--out-root", default="exp_sla_ab")
    args = ap.parse_args()

    out_root = args.out_root
    out_dir = REPO / "ComfyUI-master_cp" / "output" / out_root
    meta = EXP / "logs" / f"{out_root}.json"
    health(args.comfy)
    all_cases = load_cases()
    cases = all_cases
    if args.only:
        allow = set(args.only)
        cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]
    # Warmup prompt must not match any timed case (avoid --cache-classic hits).
    timed_prompts = {c["prompt"] for c in cases}
    warmup_prompt = next(
        (c["prompt"] for c in all_cases if c["prompt"] not in timed_prompts),
        "warmup only, ignore content.",
    )
    sla_sparsity = float(args.sla_sparsity)
    sla_lora = LORA_PROD_T2V if args.sla_lora == "prod" else LORA_SLA_T2V
    sol_tau = float(args.sol_tau)
    print(
        f"A/B t2v {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} "
        f"variants={args.variants} sla_sparsity={sla_sparsity} "
        f"sla_lora={args.sla_lora} sol_tau={sol_tau} n={len(cases)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def build_sla(oroot: str, label: str, prompt: str) -> dict:
        return graph_sla(
            oroot, label, prompt, sparsity=sla_sparsity, turbo_lora=sla_lora
        )

    def build_sol(oroot: str, label: str, prompt: str) -> dict:
        return graph_sol(oroot, label, prompt, turbo_lora=LORA_PROD_T2V, tau=sol_tau)

    builders = {
        "sparge": (graph_sparge, LORA_PROD_T2V),
        "sla": (build_sla, sla_lora),
        "sol": (build_sol, LORA_PROD_T2V),
    }

    results = []
    for variant in args.variants:
        build, lora = builders[variant]
        print(f"\n######## variant={variant} lora={lora} ########", flush=True)
        if not args.skip_warmup:
            print(f"=== warmup {variant} (discard) ===", flush=True)
            w = run_one(
                args.comfy,
                build(out_root, f"_warmup_{variant}", warmup_prompt),
                f"_warmup_{variant}",
            )
            print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))
            for p in out_dir.glob(f"_warmup_{variant}*"):
                p.unlink(missing_ok=True)

        for c in cases:
            if variant == "sla":
                tag = f"sla{sla_sparsity:g}_{args.sla_lora}lora"
            elif variant == "sol":
                tag = f"sol_tau{sol_tau:g}"
            else:
                tag = variant
            label = f"{tag}_{c['id']}_{_slug(c['title'])}"
            print(f"=== {label} ===", flush=True)
            r = run_one(args.comfy, build(out_root, label, c["prompt"]), label)
            r.update({
                "variant": variant,
                "case_id": c["id"],
                "group": c["group"],
                "title": c["title"],
                "lora": lora,
                "sla_sparsity": sla_sparsity if variant == "sla" else None,
                "sla_lora_kind": args.sla_lora if variant == "sla" else None,
                "sol_tau": sol_tau if variant == "sol" else None,
            })
            results.append(r)
            print(
                json.dumps(
                    {
                        "label": r.get("label"),
                        "status": r.get("status"),
                        "elapsed_sec": r.get("elapsed_sec"),
                        "generation_sec": (r.get("profiling") or {}).get("generation_sec"),
                        "queue_sec": (r.get("profiling") or {}).get("queue_sec"),
                        "error": r.get("error"),
                    },
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
        gen = [
            (r.get("profiling") or {}).get("generation_sec")
            for r in results
            if r.get("variant") == variant and r.get("status") == "success"
            and (r.get("profiling") or {}).get("generation_sec") is not None
        ]
        summary[variant] = {
            "n_ok": len(ok),
            "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
            "secs": ok,
            "mean_generation_sec": round(sum(gen) / len(gen), 3) if gen else None,
            "generation_secs": gen,
        }
    payload = {"summary": summary, "results": results, "config": {
        "width": WIDTH, "height": HEIGHT, "length": LENGTH, "steps": STEPS,
        "seed": SEED, "sparge_topk": SPARGE_TOPK, "sla_sparsity": sla_sparsity,
        "sla_lora": args.sla_lora, "sla_lora_file": sla_lora,
    }}
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {meta}", flush=True)
    print(f"videos: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
