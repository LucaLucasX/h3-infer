#!/usr/bin/env python3
"""T2V cases: current 4-step LoRA, Sparge replaced by VSA pure-sparse (no gate)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "report"), str(REPO / "dev" / "sage_lora_sparge")]

import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node
from loras import LORA_T2V

WIDTH, HEIGHT, FPS, STEPS, SEED = 1280, 704, 24.0, 4, 42
KEEP_PERCENT = 10.0


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def align_length(seconds: int) -> int:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    return min(candidates, key=lambda value: abs(value - requested))


def load_cases() -> list[dict]:
    from run_textcases_comfy4_tc_sparge import load_cases as _load

    return _load()


def graph_vsa(out_root: str, label: str, prompt: str, *, length: int, keep_percent: float, seed: int) -> dict:
    g = build_graph(GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=length,
        fps=FPS,
        steps=STEPS,
        seed=int(seed),
        turbo=True,
        turbo_lora=LORA_T2V,
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=False,
        output_prefix=f"{out_root}/{label}",
    ))
    g["5"] = node(
        "MiniMaxH3VSAAttnPatchExp",
        {
            "model": ["1", 0],
            "keep_percent": float(keep_percent),
            "dense_first_steps": 0,
            "num_layers": 50,
        },
    )
    return g


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base, pid, timeout=7200, poll_interval=2)
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
    ap.add_argument("--comfy", default="http://127.0.0.1:8197")
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--keep-percent", type=float, default=KEEP_PERCENT)
    ap.add_argument("--only", nargs="*", default=["o01", "o02", "o03", "o04"])
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--out-root", default="exp_vsa_pure_sparse")
    args = ap.parse_args()

    length = align_length(args.seconds)
    keep = float(args.keep_percent)
    out_root = args.out_root
    out_dir = REPO / "ComfyUI-master_cp" / "output" / out_root
    meta = EXP / "logs" / f"{out_root}.json"

    print(f"health {args.comfy}", flush=True)
    health(args.comfy)
    all_cases = load_cases()
    cases = all_cases
    if args.only:
        allow = set(args.only)
        cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]
    if not cases:
        raise SystemExit("no cases matched --only")

    timed_prompts = {c["prompt"] for c in cases}
    warmup_prompt = next(
        (c["prompt"] for c in all_cases if c["prompt"] not in timed_prompts),
        "warmup only, ignore content.",
    )
    print(
        f"VSA t2v {WIDTH}x{HEIGHT} len={length} steps={STEPS} keep_percent={keep} "
        f"lora={LORA_T2V} n={len(cases)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = run_one(
            args.comfy,
            graph_vsa(out_root, "_warmup_vsa", warmup_prompt, length=length, keep_percent=keep, seed=0),
            "_warmup_vsa",
        )
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup_vsa*"):
            p.unlink(missing_ok=True)

    for c in cases:
        label = f"vsa{keep:g}_{c['id']}_{_slug(c['title'])}"
        print(f"=== {label} ===", flush=True)
        r = run_one(
            args.comfy,
            graph_vsa(out_root, label, c["prompt"], length=length, keep_percent=keep, seed=SEED),
            label,
        )
        r.update({"case_id": c["id"], "group": c["group"], "title": c["title"]})
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False), flush=True)

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success"]
    payload = {
        "stack": "vsa_pure_sparse_no_gate",
        "keep_percent": keep,
        "width": WIDTH,
        "height": HEIGHT,
        "length": length,
        "steps": STEPS,
        "lora": LORA_T2V,
        "n_ok": len(ok),
        "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
        "results": results,
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n_ok": payload["n_ok"], "mean_sec": payload["mean_sec"]}, ensure_ascii=False))
    print(f"wrote {meta}", flush=True)
    print(f"videos: {out_dir}", flush=True)
    return 0 if payload["n_ok"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
