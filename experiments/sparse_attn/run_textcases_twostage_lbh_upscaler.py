#!/usr/bin/env python3
"""53-case benchmark: two-stage MiniMax H3 + LBH latent upscaler.

Pipeline:
1) pass-1 at low latent canvas (twostep mode inside h3_graph)
2) LBH learned latent upscaler node (2D/3D) replaces default bicubic combiner
3) pass-2 refinement at target 1344x768, 12s
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
OUT_ROOT = "exp_textcases_12s_twostage_lbh3d"

DEFAULT_LORA = "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"
DEFAULT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def build_twostage_graph(
    *,
    out_root: str,
    label: str,
    prompt: str,
    image: str,
    steps: int,
    split_step: int,
    twostep_scale: float,
    turbo_lora: str,
    use_sparge: bool,
    sparge_topk: float,
    audio_dense: bool,
    upscaler_class: str,
    upscaler_model_name: str,
    upscaler_precision: str,
) -> dict:
    p = GenerateParams(
        mode="twostep",
        prompt=prompt,
        image=image,
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
        twostep_scale=twostep_scale,
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)

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

    # keep turbo LoRA configurable
    g["8"]["inputs"]["lora_name"] = turbo_lora
    # global scheduler + split point between pass1/pass2
    g["10"]["inputs"]["steps"] = int(steps)
    g["11"]["inputs"]["step"] = int(split_step)

    # replace default bicubic combiner with LBH learned upscaler
    g["17"] = {
        "class_type": upscaler_class,
        "inputs": {
            "latent": ["16", 1],
            "model_name": upscaler_model_name,
            "scale": float(twostep_scale),
            "device": "cuda",
            "precision": upscaler_precision,
        },
    }
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
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--split-step", type=int, default=3)
    ap.add_argument("--twostep-scale", type=float, default=2.0)
    ap.add_argument("--image", default="example.png")
    ap.add_argument("--turbo-lora", default=DEFAULT_LORA)
    ap.add_argument("--no-sparge", action="store_true")
    ap.add_argument("--topk", type=float, default=0.5)
    ap.add_argument("--audio-dense", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--upscaler-model", default=DEFAULT_UPSCALER)
    ap.add_argument("--upscaler-node", choices=("3d", "2d"), default="3d")
    ap.add_argument("--upscaler-precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--out-root", default=OUT_ROOT)
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

    upscaler_class = "MinimaxH3LatentUpscalerNode3D" if args.upscaler_node == "3d" else "MinimaxH3LatentUpscalerNode2D"
    use_sparge = not args.no_sparge
    print(
        f"textcases twostage+lbh: {WIDTH}x{HEIGHT} len={LENGTH} steps={args.steps} split={args.split_step} "
        f"n={len(cases)} lora={args.turbo_lora} upscaler={args.upscaler_model}/{args.upscaler_node} "
        f"sparge={use_sparge} topk={args.topk} audio_dense={args.audio_dense} out={out_root}",
        flush=True,
    )

    if not args.skip_warmup:
        wlabel = "_warmup"
        wg = build_twostage_graph(
            out_root=out_root,
            label=wlabel,
            prompt=cases[0]["prompt"],
            image=args.image,
            steps=args.steps,
            split_step=args.split_step,
            twostep_scale=args.twostep_scale,
            turbo_lora=args.turbo_lora,
            use_sparge=use_sparge,
            sparge_topk=args.topk,
            audio_dense=args.audio_dense,
            upscaler_class=upscaler_class,
            upscaler_model_name=args.upscaler_model,
            upscaler_precision=args.upscaler_precision,
        )
        print("=== warmup (discard) ===", flush=True)
        w = run_one(args.comfy, wg, wlabel)
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup*"):
            p.unlink(missing_ok=True)

    results = []
    for c in cases:
        label = f"{c['id']}_{_slug(c['title'])}"
        g = build_twostage_graph(
            out_root=out_root,
            label=label,
            prompt=c["prompt"],
            image=args.image,
            steps=args.steps,
            split_step=args.split_step,
            twostep_scale=args.twostep_scale,
            turbo_lora=args.turbo_lora,
            use_sparge=use_sparge,
            sparge_topk=args.topk,
            audio_dense=args.audio_dense,
            upscaler_class=upscaler_class,
            upscaler_model_name=args.upscaler_model,
            upscaler_precision=args.upscaler_precision,
        )
        print(f"=== {label} ({c['group']}) ===", flush=True)
        r = run_one(args.comfy, g, label)
        r.update({"case_id": c["id"], "group": c["group"], "title": c["title"]})
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False), flush=True)

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success" and r.get("elapsed_sec")]
    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": int(args.steps),
            "split_step": int(args.split_step),
            "seed": SEED,
            "image": args.image,
            "turbo_lora": args.turbo_lora,
            "twostep_scale": float(args.twostep_scale),
            "upscaler_class": upscaler_class,
            "upscaler_model": args.upscaler_model,
            "upscaler_precision": args.upscaler_precision,
            "sparge": use_sparge,
            "sparge_topk": float(args.topk) if use_sparge else None,
            "audio_dense": bool(args.audio_dense) if use_sparge else None,
            "attention": "sage_v2_memeff",
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

