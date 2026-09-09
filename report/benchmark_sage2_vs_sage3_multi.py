#!/usr/bin/env python3
"""Multi-repeat interleaved Sage2 vs Sage3 validation at official 1344×768.

Design:
  - Same complex text prompt / twostep (and optional direct+lora) configs
  - Warmup once (discarded)
  - Interleaved order each round: sage2 then sage3 (reduces cold-start bias)
  - Multiple repeats; report mean / median / min / max / std
  - Wall-clock submit→history; also scrape ComfyUI "Prompt executed in Xs" from log
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import TURBO_LORA_V4, GenerateParams, build_graph
from benchmark_official_1344x768 import COMPLEX_TEXT, WIDTH, HEIGHT, LENGTH, SEED

OUT_ROOT = "h3_sage2_vs_sage3_multi"
COMFY_LOG = ROOT / "ComfyUI-master_cp" / "logs" / "comfyui_sage_bench_8188.log"
EXEC_RE = re.compile(r"Prompt executed in ([0-9.]+) seconds")

# Focused pairs where prior claim was "sage3 much faster"
PAIRS: list[dict] = [
    dict(
        name="twostep_8",
        mode="twostep",
        steps=8,
        turbo=False,
        prior={"sage2": 81.06, "sage3": 60.18},
    ),
    dict(
        name="twostep_20",
        mode="twostep",
        steps=20,
        turbo=False,
        prior={"sage2": 120.06, "sage3": 102.05},
    ),
    dict(
        name="twostep_lora_8",
        mode="twostep",
        steps=8,
        turbo=True,
        prior={"sage2": 75.05, "sage3": 63.03},
    ),
    dict(
        name="direct_lora_8",
        mode="t2v",
        steps=8,
        turbo=True,
        prior={"sage2": 90.04, "sage3": 84.06},
    ),
]

ATTN = {
    "sage2": "sage_v2_memeff",
    "sage3": "sage_v3",
}


def make_params(pair: dict, backend_key: str, rep: int) -> GenerateParams:
    label = f"{pair['name']}_{backend_key}_r{rep}"
    return GenerateParams(
        mode=pair["mode"],
        prompt=COMPLEX_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=pair["steps"],
        seed=SEED,
        image="",
        output_prefix=f"{OUT_ROOT}/{label}",
        sage_attention="disabled",
        attention_backend=ATTN[backend_key],
        turbo=pair["turbo"],
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
        sampler="res_multistep",
        scheduler="simple",
    )


def scrape_last_exec_time(log_path: Path, after_pos: int) -> tuple[float | None, int]:
    """Return (seconds, new_file_pos) for the first 'Prompt executed' after after_pos."""
    if not log_path.is_file():
        return None, after_pos
    data = log_path.read_bytes()
    chunk = data[after_pos:]
    m = EXEC_RE.search(chunk.decode("utf-8", errors="replace"))
    new_pos = after_pos + (m.end() if m else len(chunk))
    if not m:
        return None, new_pos
    return float(m.group(1)), new_pos


def run_one(base_url: str, pair: dict, backend_key: str, rep: int, log_pos: int) -> tuple[dict, int]:
    params = make_params(pair, backend_key, rep)
    params.validate()
    t0 = time.time()
    pid = submit_prompt(base_url, build_graph(params))
    entry = wait_prompt(base_url, pid, poll_interval=2.0, timeout=4 * 3600)
    wall = round(time.time() - t0, 2)
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        raise ComfyUIError(f"{params.output_prefix}: {(st.get('messages') or [])[:3]}")
    # small settle so log flush catches Prompt executed line
    time.sleep(0.5)
    exec_s, log_pos = scrape_last_exec_time(COMFY_LOG, log_pos)
    return {
        "pair": pair["name"],
        "backend": backend_key,
        "attention": ATTN[backend_key],
        "mode": pair["mode"],
        "steps": pair["steps"],
        "turbo": pair["turbo"],
        "rep": rep,
        "wall_sec": wall,
        "comfy_exec_sec": exec_s,
        "prompt_id": pid,
        "outputs": extract_outputs(entry),
        "status": st.get("status_str"),
    }, log_pos


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        "values": vals,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats per backend (after warmup)")
    ap.add_argument("--pairs", nargs="*", default=None, help="subset of pair names")
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_sage2_vs_sage3_multi.json")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    pairs = PAIRS
    if args.pairs:
        want = set(args.pairs)
        pairs = [p for p in PAIRS if p["name"] in want]
        if not pairs:
            raise SystemExit(f"no pairs match {args.pairs}")

    log_pos = COMFY_LOG.stat().st_size if COMFY_LOG.is_file() else 0
    runs: list[dict] = []

    # Warmup: one sage2 + one sage3 on first pair (discard)
    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        for bk in ("sage2", "sage3"):
            r, log_pos = run_one(base_url, pairs[0], bk, rep=0, log_pos=log_pos)
            print(
                f"warmup {bk}: wall={r['wall_sec']}s comfy={r['comfy_exec_sec']}",
                flush=True,
            )

    print(
        f"\n=== timed: {len(pairs)} pairs × 2 backends × {args.repeats} reps (interleaved) ===",
        flush=True,
    )
    for rep in range(1, args.repeats + 1):
        for pair in pairs:
            for bk in ("sage2", "sage3"):  # interleaved within pair
                print(f"\n--- {pair['name']} {bk} r{rep} ---", flush=True)
                r, log_pos = run_one(base_url, pair, bk, rep=rep, log_pos=log_pos)
                print(
                    f"{pair['name']} {bk} r{rep}: wall={r['wall_sec']}s comfy={r['comfy_exec_sec']}",
                    flush=True,
                )
                runs.append(r)
                # incremental save
                args.out.write_text(
                    json.dumps(
                        {
                            "config": {
                                "width": WIDTH,
                                "height": HEIGHT,
                                "length": LENGTH,
                                "seed": SEED,
                                "repeats": args.repeats,
                                "warmup": not args.skip_warmup,
                                "order": "per-rep: for each pair sage2 then sage3",
                                "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
                            },
                            "runs": runs,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Summaries
    summary = []
    for pair in pairs:
        row = {"name": pair["name"], "prior": pair["prior"], "backends": {}}
        for bk in ("sage2", "sage3"):
            walls = [r["wall_sec"] for r in runs if r["pair"] == pair["name"] and r["backend"] == bk]
            comfys = [
                r["comfy_exec_sec"]
                for r in runs
                if r["pair"] == pair["name"] and r["backend"] == bk and r["comfy_exec_sec"] is not None
            ]
            row["backends"][bk] = {
                "wall": summarize(walls),
                "comfy_exec": summarize(comfys),
            }
        s2 = row["backends"]["sage2"]["wall"].get("mean")
        s3 = row["backends"]["sage3"]["wall"].get("mean")
        if s2 and s3:
            delta = round(s3 - s2, 2)
            pct = round(100.0 * (s3 - s2) / s2, 1)
            row["wall_mean_delta_s3_minus_s2"] = delta
            row["wall_mean_pct"] = pct
            row["verdict"] = (
                "sage3_faster"
                if s3 < s2 * 0.95
                else ("sage2_faster" if s2 < s3 * 0.95 else "approx_equal")
            )
        summary.append(row)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "repeats": args.repeats,
            "warmup": not args.skip_warmup,
            "order": "per-rep: for each pair sage2 then sage3",
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "verdict_rule": "faster if mean < other * 0.95 else approx_equal",
        },
        "summary": summary,
        "runs": runs,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print("\n======== SUMMARY (wall mean) ========", flush=True)
    for row in summary:
        s2 = row["backends"]["sage2"]["wall"]
        s3 = row["backends"]["sage3"]["wall"]
        print(
            f"{row['name']}: sage2 mean={s2.get('mean')} "
            f"(n={s2.get('n')} [{s2.get('min')}-{s2.get('max')}] σ={s2.get('stdev')}) | "
            f"sage3 mean={s3.get('mean')} "
            f"(n={s3.get('n')} [{s3.get('min')}-{s3.get('max')}] σ={s3.get('stdev')}) | "
            f"Δ={row.get('wall_mean_delta_s3_minus_s2')}s ({row.get('wall_mean_pct')}%) "
            f"→ {row.get('verdict')}",
            flush=True,
        )
        print(
            f"  prior single-run: sage2={row['prior']['sage2']} sage3={row['prior']['sage3']}",
            flush=True,
        )
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
