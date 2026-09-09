#!/usr/bin/env python3
"""Fair A/B: same pipeline, 20-step baseline vs 8-step Turbo LoRA."""

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
from h3_graph import GenerateParams, TURBO_LORA_V4, build_graph

CASES = [
    # Direct t2v — apples to apples with earlier 480p/20step bench
    ("t2v_20_sage", dict(mode="t2v", steps=20, turbo=False, attention_backend="sage_v2_memeff")),
    ("t2v_8_sage_turbo", dict(mode="t2v", steps=8, turbo=True, attention_backend="sage_v2_memeff")),
    # Twostep — same res/length, only steps+turbo differ
    ("twostep_20_sage", dict(mode="twostep", steps=20, turbo=False, attention_backend="sage_v2_memeff")),
    ("twostep_8_sage_turbo", dict(mode="twostep", steps=8, turbo=True, attention_backend="sage_v2_memeff")),
]


def run_one(base_url: str, label: str, overrides: dict, seed: int, width: int, height: int, length: int, image: str) -> dict:
    params = GenerateParams(
        mode=overrides["mode"],
        prompt="A woman walking in a sunny park, cinematic",
        width=width,
        height=height,
        length=length,
        steps=overrides["steps"],
        seed=seed,
        image=image,
        output_prefix=f"h3_bench_fair/{label}",
        attention_backend=overrides["attention_backend"],
        turbo=overrides["turbo"],
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
    )
    t0 = time.time()
    pid = submit_prompt(base_url, build_graph(params))
    entry = wait_prompt(base_url, pid, poll_interval=2.0)
    elapsed = round(time.time() - t0, 2)
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        raise ComfyUIError(f"{label}: {(st.get('messages') or [])[:2]}")
    return {
        "label": label,
        "mode": params.mode,
        "steps": params.steps,
        "turbo": params.turbo,
        "elapsed_sec": elapsed,
        "prompt_id": pid,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--image", default="example.png")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_fair_8_vs_20.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    if args.warmup:
        print("Warmup (t2v 4-step turbo, not timed)...")
        run_one(
            base_url,
            "warmup",
            dict(mode="t2v", steps=4, turbo=True, attention_backend="sage_v2_memeff"),
            seed=args.seed + 999,
            width=args.width,
            height=args.height,
            length=args.length,
            image=args.image,
        )

    results = []
    print(f"Fair A/B @ {args.width}x{args.height} len={args.length}")
    for i, (label, ov) in enumerate(CASES):
        print(f"\n=== {label} ===")
        try:
            row = run_one(base_url, label, ov, args.seed + i, args.width, args.height, args.length, args.image)
            results.append(row)
            print(f"  {row['elapsed_sec']}s")
        except Exception as e:
            results.append({"label": label, "error": str(e)})
            print(f"  FAILED: {e}")

    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}")

    ok = [r for r in results if "elapsed_sec" in r]
    t2v = {r["label"]: r for r in ok if r["label"].startswith("t2v_")}
    ts = {r["label"]: r for r in ok if r["label"].startswith("twostep_")}
    if "t2v_20_sage" in t2v and "t2v_8_sage_turbo" in t2v:
        a, b = t2v["t2v_20_sage"]["elapsed_sec"], t2v["t2v_8_sage_turbo"]["elapsed_sec"]
        print(f"\nt2v: 20-step sage {a}s → 8-step sage+turbo {b}s  ({a/b:.2f}x faster)")
    if "twostep_20_sage" in ts and "twostep_8_sage_turbo" in ts:
        a, b = ts["twostep_20_sage"]["elapsed_sec"], ts["twostep_8_sage_turbo"]["elapsed_sec"]
        print(f"twostep: 20-step sage {a}s → 8-step sage+turbo {b}s  ({a/b:.2f}x faster)")
    return 0 if len(ok) == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
