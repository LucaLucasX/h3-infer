#!/usr/bin/env python3
"""Run a single t2v prompt on a running Sage2+TurboLoRA+Sparge ComfyUI instance.

Usage (after starting the server):
  python3 dev/sage_lora_sparge/scripts/run_custom.py --prompt "A cat on the moon"
  python3 dev/sage_lora_sparge/scripts/run_custom.py --case o01
  python3 dev/sage_lora_sparge/scripts/run_custom.py --prompt "..." --width 864 --height 480 --length 121 --steps 4 --topk 0.5

If --prompt is omitted, an interactive prompt is shown.
"""
from __future__ import annotations

import argparse
import json
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


def load_case(case_id: str) -> str:
    sys.path.insert(0, str(REPO / "report"))
    from run_textcases_comfy4_tc_sparge import load_cases

    cases = load_cases()
    for c in cases:
        if c["id"] == case_id or c["title"] == case_id:
            return c["prompt"]
    raise SystemExit(f"case '{case_id}' not found (available ids like o01..o27, h01..h17, e007..)")


def graph_for(out_root: str, label: str, prompt: str, args: argparse.Namespace) -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        steps=args.steps,
        seed=args.seed,
        turbo=True,
        turbo_lora=LORA_T2V,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        image="",
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    if not args.no_sparge:
        g["5"] = node(
            "MiniMaxH3SpargeAttnPatchExp",
            {
                "model": ["1", 0],
                "topk": float(args.topk),
                "dense_first_steps": 0,
                "num_layers": 50,
                "audio_dense": True,
            },
        )
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--prompt", default=None, help="t2v prompt text")
    ap.add_argument("--case", default=None, help="load prompt from a text-case id/title, e.g. o01")
    ap.add_argument("--width", type=int, default=1376)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=288, help="frames (288 @24fps = 12s)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=float, default=0.5)
    ap.add_argument("--no-sparge", action="store_true")
    ap.add_argument("--out-root", default="dev_custom_t2v")
    ap.add_argument("--label", default="custom")
    args = ap.parse_args()

    if args.prompt and args.case:
        raise SystemExit("use either --prompt or --case, not both")
    if args.case:
        prompt = load_case(args.case)
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = input("prompt> ").strip()
    if not prompt:
        raise SystemExit("empty prompt")

    health(args.comfy)
    g = graph_for(args.out_root, args.label, prompt, args)
    print(
        f"t2v {args.width}x{args.height} len={args.length} fps={args.fps} steps={args.steps} "
        f"seed={args.seed} topk={args.topk} sparge={not args.no_sparge} "
        f"lora={LORA_T2V}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(args.comfy, g)
        print(f"prompt_id={pid}", flush=True)
        hist = wait_prompt(args.comfy, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        ok = st == "success"
        print(json.dumps(
            {"status": "success" if ok else f"comfy_{st}", "elapsed_sec": elapsed, "outputs": extract_outputs(hist)},
            ensure_ascii=False,
        ), flush=True)
        if not ok:
            msgs = (hist.get("status") or {}).get("messages", [])
            for m in msgs:
                if m and m[0] == "execution_error":
                    print("ERROR:", m[1].get("exception_message", ""), file=sys.stderr)
                    break
    except ComfyUIError as e:
        print(json.dumps({"status": "error", "elapsed_sec": round(time.perf_counter() - t0, 2), "error": str(e)[:4000]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
