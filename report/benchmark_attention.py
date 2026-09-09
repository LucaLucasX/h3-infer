#!/usr/bin/env python3
"""Benchmark H3 t2v wall-clock time across attention backends."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, health, submit_prompt, wait_prompt
from h3_graph import AttentionBackend, GenerateParams, build_graph

BACKENDS: list[tuple[str, AttentionBackend]] = [
    ("pytorch", "pytorch"),
    ("sage_v1_triton", "sage_v1"),
    ("sage_v2_memeff", "sage_v2_memeff"),
    ("sage_v3", "sage_v3"),
]


def wait_comfy(base_url: str, timeout: float = 600.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            health(base_url)
            return
        except Exception:
            time.sleep(3.0)
    raise ComfyUIError(f"ComfyUI not ready at {base_url}")


def run_one(base_url: str, backend: AttentionBackend, p: GenerateParams, *, seed: int) -> dict:
    params = GenerateParams(
        mode=p.mode,
        prompt=p.prompt,
        width=p.width,
        height=p.height,
        length=p.length,
        steps=p.steps,
        seed=seed,
        output_prefix=f"h3_bench/{backend}_{seed}",
        attention_backend=backend,
    )
    t0 = time.time()
    pid = submit_prompt(base_url, build_graph(params))
    entry = wait_prompt(base_url, pid, poll_interval=2.0)
    elapsed = time.time() - t0
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        msgs = st.get("messages") or []
        raise ComfyUIError(f"{backend} failed: {msgs[:3]}")
    return {
        "backend": backend,
        "prompt_id": pid,
        "elapsed_sec": round(elapsed, 2),
        "status": st.get("status_str", "unknown"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--width", type=int, default=608)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--length", type=int, default=56)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="A woman walking in a sunny park, cinematic")
    ap.add_argument("--start-comfy", action="store_true", help="start start_sage_bench_comfyui.sh first")
    ap.add_argument("--warmup", action="store_true", default=True, help="one warmup t2v before timed runs")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_attention_results.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    if args.start_comfy:
        subprocess.run(["bash", str(ROOT / "start_sage_bench_comfyui.sh")], check=True)
        time.sleep(5.0)

    wait_comfy(base_url)
    base = GenerateParams(
        mode="t2v",
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
    )
    base.validate()

    if args.warmup:
        print("Warmup run (pytorch, not timed)...")
        run_one(base_url, "pytorch", base, seed=args.seed + 1000)

    results: list[dict] = []
    print(f"Benchmark t2v {args.width}x{args.height} len={args.length} steps={args.steps}")
    for i, (label, backend) in enumerate(BACKENDS):
        print(f"\n=== {label} ===")
        try:
            row = run_one(base_url, backend, base, seed=args.seed + i)
            row["label"] = label
            results.append(row)
            print(f"  {row['elapsed_sec']}s  ({row['status']})")
        except Exception as e:
            results.append({"label": label, "backend": backend, "error": str(e)})
            print(f"  FAILED: {e}")

    payload = {
        "config": {
            "width": args.width,
            "height": args.height,
            "length": args.length,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "base_url": base_url,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}")

    ok = [r for r in results if "elapsed_sec" in r]
    if ok:
        fastest = min(ok, key=lambda r: r["elapsed_sec"])
        print("\nSummary (wall-clock):")
        for r in ok:
            rel = r["elapsed_sec"] / fastest["elapsed_sec"]
            print(f"  {r['label']:16s} {r['elapsed_sec']:7.1f}s  ({rel:.2f}x vs fastest)")
    return 0 if len(ok) == len(BACKENDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
