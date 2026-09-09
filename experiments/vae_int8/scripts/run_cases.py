#!/usr/bin/env python3
"""T2V batch with Kijai int8_convrot video VAE on production L0 stack.

L0 = int8_convrot UNET + sage2 + Sparge 0.5 + 4-step turbo LoRA. No lossless-mem.
Default cases: o01, o02, o03 (same trio as nvfp4 / lora_opt smoke).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "vae_int8"
sys.path[:0] = [str(ROOT), str(ROOT / "report"), str(ROOT / "dev" / "sage_lora_sparge")]

import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from loras import LORA_T2V

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VAE_INT8 = "minimax_h3_video_vae_int8_convrot.safetensors"
WIDTH, HEIGHT, FPS, STEPS = 1376, 768, 24.0, 4
SEED = 42
SPARGE_TOPK = 0.5


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


def graph_for(out_root: str, label: str, prompt: str, *, length: int) -> dict:
    g = build_graph(GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=length,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        turbo_lora=LORA_T2V,
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=True,
        sparge_topk=SPARGE_TOPK,
        sparge_dense_first_steps=0,
        sparge_num_layers=50,
        sparge_audio_dense=True,
        unet_name=UNET,
        output_prefix=f"{out_root}/{label}",
    ))
    g["3"]["inputs"]["vae_name"] = VAE_INT8
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8194")  # vae_int8/run_comfy.sh on master_cp
    ap.add_argument("--only", nargs="*", default=["o01", "o02", "o03"])
    ap.add_argument("--duration", type=int, default=12)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--out-root", default="exp_vae_int8")
    args = ap.parse_args()

    length = align_length(args.duration)
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / args.out_root
    meta = EXP / "logs" / "cases.json"
    health(args.comfy)

    all_cases = load_cases()
    allow = set(args.only)
    cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]
    if not cases:
        raise SystemExit(f"no cases matched --only {args.only}")

    print(
        f"vae_int8 t2v: vae={VAE_INT8} unet={UNET} {WIDTH}x{HEIGHT} "
        f"len={length} (~{args.duration}s) steps={STEPS} lora={LORA_T2V} "
        f"stack=L0 n={len(cases)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_warmup:
        print("=== warmup ===", flush=True)
        w = run_one(
            args.comfy,
            graph_for(args.out_root, "_warmup", cases[0]["prompt"], length=length),
            "_warmup",
        )
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup*"):
            p.unlink(missing_ok=True)

    results = []
    for c in cases:
        label = f"{c['id']}_{_slug(c['title'])}"
        print(f"=== {label} ===", flush=True)
        r = run_one(
            args.comfy,
            graph_for(args.out_root, label, c["prompt"], length=length),
            label,
        )
        r.update({"case_id": c["id"], "title": c["title"], "group": c.get("group")})
        results.append(r)
        print(
            json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False),
            flush=True,
        )

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success"]
    payload = {
        "vae": VAE_INT8,
        "unet": UNET,
        "stack": "L0_sage2_sparge0.5_4step",
        "width": WIDTH,
        "height": HEIGHT,
        "length": length,
        "duration_sec": args.duration,
        "steps": STEPS,
        "n": len(results),
        "n_ok": len(ok),
        "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
        "results": results,
    }
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "n_ok", "mean_sec", "vae")}, ensure_ascii=False))
    print(f"wrote {meta}")
    print(f"videos: {out_dir}")


if __name__ == "__main__":
    main()
