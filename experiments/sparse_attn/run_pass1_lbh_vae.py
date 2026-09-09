#!/usr/bin/env python3
"""Benchmark: 4-step + Sparge → LBH latent upscale → direct VAE decode (no pass2 refine).

Pipeline:
  pass1 (4 steps, full sigmas)  →  LBH 3D learned upscaler  →  VAEDecode / VAEDecodeAudio
No pass2 refine, no first-frame reference (pure t2v via twostep scaffold).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from report.run_textcases_comfy4_tc_sparge import load_cases

WIDTH, HEIGHT = 1344, 768
LENGTH = 288
FPS = 24.0
SEED = 42

DEFAULT_LORA = "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"
DEFAULT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def build_pass1_lbh_graph(
    *,
    out_root: str,
    label: str,
    prompt: str,
    steps: int,
    turbo_lora: str,
    use_sparge: bool,
    sparge_topk: float,
    audio_dense: bool,
    upscaler_model_name: str,
    upscaler_precision: str,
    pass1_scale: float,
    lbh_scale: float,
) -> dict:
    # Build twostep scaffold (no image = no first-frame reference)
    p = GenerateParams(
        mode="twostep",
        prompt=prompt,
        image=None,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=steps,
        seed=SEED,
        turbo=True,
        turbo_lora=turbo_lora,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        twostep_scale=pass1_scale,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)

    # --- Sparge patch (replaces node "5" MiniMaxH3MemoryEfficientSageAttentionPatch) ---
    if use_sparge:
        g["5"] = {
            "class_type": "MiniMaxH3SpargeAttnPatchExp",
            "inputs": {
                "model": ["1", 0],
                "topk": float(sparge_topk),
                "dense_first_steps": 0,
                "num_layers": 50,
                "audio_dense": bool(audio_dense),
            },
        }

    # Use full sigmas (not split) for pass1 — use node "10" output directly
    # node "10" = BasicScheduler -> outputs [sigmas]
    # node "11" = SplitSigmas -> ["10",0] split at split_step
    # pass1 sampler "16" currently uses ["10", 0] (full sigmas) — already correct
    # (SplitSigmas "11" output ["11",0] = pass1 sigmas, ["11",1] = pass2 sigmas)
    # Actually node 16 inputs sigmas=["10",0] (full schedule), so pass1 already runs all steps.
    # For pure pass1 we keep it as-is (all `steps` steps on pass1).

    # --- Replace node "17" with LBH learned upscaler ---
    # Original node 17 = MiniMaxH3LatentUpscaleCombined, inputs samples=["16",1]
    # LBH upscaler takes latent dict; wrap in {"samples": ...} via the node interface
    g["17"] = {
        "class_type": "MinimaxH3LatentUpscalerNode3D",
        "inputs": {
            "latent": ["16", 0],
            "model_name": upscaler_model_name,
            "scale": float(lbh_scale),
            "device": "cuda",
            "precision": upscaler_precision,
        },
    }

    # --- Rewire VAEDecode / VAEDecodeAudio to read from LBH upscaler output ---
    # node "20" = VAEDecode, node "21" = VAEDecodeAudio
    # Original: samples=["19",0] (pass2 sampler) -> change to ["17",0] (LBH output)
    g["20"]["inputs"]["samples"] = ["17", 0]
    g["21"]["inputs"]["samples"] = ["17", 0]

    # --- Remove pass2 nodes (no longer needed) ---
    g.pop("18", None)  # BasicGuider pass2
    g.pop("19", None)  # SamplerCustomAdvanced pass2

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
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--pass1-scale", type=float, default=2.0, help="twostep pass1 scale in h3_graph")
    ap.add_argument("--lbh-scale", type=float, default=2.0, help="LBH latent upscaler scale")
    ap.add_argument("--turbo-lora", default=DEFAULT_LORA)
    ap.add_argument("--no-sparge", action="store_true")
    ap.add_argument("--topk", type=float, default=0.5)
    ap.add_argument("--audio-dense", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--upscaler-model", default=DEFAULT_UPSCALER)
    ap.add_argument("--upscaler-precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--out-root", default="exp_pass1_lbh_vae")
    args = ap.parse_args()

    out_root = args.out_root
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / out_root
    meta = ROOT / "experiments" / "sparse_attn" / "output" / f"{out_root}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    health(args.comfy)
    cases = load_cases()
    if args.only:
        allow = set(args.only)
        cases = [c for c in cases if c["id"] in allow or c["title"] in allow]

    use_sparge = not args.no_sparge
    print(
        f"pass1+lbh+vae: {WIDTH}x{HEIGHT} len={LENGTH} steps={args.steps} "
        f"n={len(cases)} lora={args.turbo_lora} upscaler={args.upscaler_model} "
        f"sparge={use_sparge} topk={args.topk} audio_dense={args.audio_dense} "
        f"pass1_scale={args.pass1_scale} lbh_scale={args.lbh_scale} out={out_root}",
        flush=True,
    )

    if not args.skip_warmup:
        wlabel = "_warmup"
        wg = build_pass1_lbh_graph(
            out_root=out_root,
            label=wlabel,
            prompt=cases[0]["prompt"],
            steps=args.steps,
            turbo_lora=args.turbo_lora,
            use_sparge=use_sparge,
            sparge_topk=args.topk,
            audio_dense=args.audio_dense,
            upscaler_model_name=args.upscaler_model,
            upscaler_precision=args.upscaler_precision,
            pass1_scale=args.pass1_scale,
            lbh_scale=args.lbh_scale,
        )
        print("=== warmup (discard) ===", flush=True)
        w = run_one(args.comfy, wg, wlabel)
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup*"):
            p.unlink(missing_ok=True)

    results = []
    for c in cases:
        label = f"{c['id']}_{_slug(c['title'])}"
        g = build_pass1_lbh_graph(
            out_root=out_root,
            label=label,
            prompt=c["prompt"],
            steps=args.steps,
            turbo_lora=args.turbo_lora,
            use_sparge=use_sparge,
            sparge_topk=args.topk,
            audio_dense=args.audio_dense,
            upscaler_model_name=args.upscaler_model,
            upscaler_precision=args.upscaler_precision,
            pass1_scale=args.pass1_scale,
            lbh_scale=args.lbh_scale,
        )
        print(f"=== {label} ({c['group']}) ===", flush=True)
        r = run_one(args.comfy, g, label)
        r.update({"case_id": c["id"], "group": c["group"], "title": c["title"]})
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False), flush=True)

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success" and r.get("elapsed_sec")]
    payload = {
        "config": {
            "width": WIDTH, "height": HEIGHT, "length": LENGTH,
            "steps": int(args.steps),
            "seed": SEED,
            "turbo_lora": args.turbo_lora,
            "pass1_scale": float(args.pass1_scale),
            "lbh_scale": float(args.lbh_scale),
            "upscaler_model": args.upscaler_model,
            "upscaler_precision": args.upscaler_precision,
            "sparge": use_sparge,
            "sparge_topk": float(args.topk) if use_sparge else None,
            "audio_dense": bool(args.audio_dense) if use_sparge else None,
            "attention": "sage_v2_memeff",
            "pass2": False,
            "out": str(out_dir.relative_to(ROOT)),
        },
        "n": len(results),
        "n_ok": len(ok),
        "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
        "results": results,
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "n_ok", "mean_sec")}, ensure_ascii=False))
    print(f"wrote {meta}", flush=True)


if __name__ == "__main__":
    main()
