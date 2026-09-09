#!/usr/bin/env python3
"""Benchmark twostep + Turbo LoRA v4 (8 steps) with optional Sage attention."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, TURBO_LORA_V4, build_graph, twostep_pass1_size, twostep_pass2_size

CASES = [
    ("twostep_pytorch", {"attention_backend": "pytorch", "turbo": False}),
    ("twostep_sage_memeff", {"attention_backend": "sage_v2_memeff", "turbo": False}),
    ("twostep_turbo", {"attention_backend": "pytorch", "turbo": True}),
    ("twostep_sage_memeff_turbo", {"attention_backend": "sage_v2_memeff", "turbo": True}),
]


def run_case(base_url: str, label: str, base: GenerateParams, overrides: dict, seed: int) -> dict:
    params = GenerateParams(
        mode=base.mode,
        prompt=base.prompt,
        width=base.width,
        height=base.height,
        length=base.length,
        steps=base.steps,
        seed=seed,
        image=base.image,
        output_prefix=f"h3_bench_twostep/{label}_{seed}",
        turbo_lora=base.turbo_lora,
        turbo_strength=base.turbo_strength,
        **overrides,
    )
    p1w, p1h = twostep_pass1_size(params.width, params.height)
    p2w, p2h = twostep_pass2_size(p1w, p1h)
    _, _, _, _, split = params.resolve_twostep()
    t0 = time.time()
    pid = submit_prompt(base_url, build_graph(params))
    entry = wait_prompt(base_url, pid, poll_interval=3.0)
    elapsed = round(time.time() - t0, 2)
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        raise ComfyUIError(f"{label} failed: {(st.get('messages') or [])[:2]}")
    return {
        "label": label,
        "elapsed_sec": elapsed,
        "prompt_id": pid,
        "pass1": f"{p1w}x{p1h}",
        "pass2_approx": f"{p2w}x{p2h}",
        "split": f"{split}+{params.steps - split}",
        "turbo": params.turbo,
        "attention": overrides.get("attention_backend") or params.sage_attention,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image", default="example.png")
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_twostep_turbo.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    base = GenerateParams(
        mode="twostep",
        prompt="A woman walking in a sunny park, cinematic",
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        image=args.image,
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
    )
    base.validate()

    print(
        f"twostep benchmark: {args.width}x{args.height} len={args.length} "
        f"steps={args.steps} lora={TURBO_LORA_V4}"
    )
    results: list[dict] = []
    for i, (label, overrides) in enumerate(CASES):
        print(f"\n=== {label} ===")
        try:
            row = run_case(base_url, label, base, overrides, seed=args.seed + i)
            results.append(row)
            print(f"  {row['elapsed_sec']}s  pass1={row['pass1']} split={row['split']}")
        except Exception as e:
            results.append({"label": label, "error": str(e)})
            print(f"  FAILED: {e}")

    payload = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "results": results}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}")

    ok = [r for r in results if "elapsed_sec" in r]
    if ok:
        fastest = min(ok, key=lambda r: r["elapsed_sec"])
        print("\nSummary:")
        for r in ok:
            rel = r["elapsed_sec"] / fastest["elapsed_sec"]
            print(f"  {r['label']:28s} {r['elapsed_sec']:7.1f}s  ({rel:.2f}x)")
    return 0 if len(ok) == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
