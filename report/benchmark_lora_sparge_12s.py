#!/usr/bin/env python3
"""4-step LoRA shootout @ 12s: Sage2 + Sparge0.5, one LoRA per run."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from benchmark_cn_accel_12s import CN_VOICE_TEXT, LENGTH, FPS
from benchmark_official_1344x768 import HEIGHT, SEED, WIDTH
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import TURBO_LORA_V4, GenerateParams, build_graph, node

STEPS = 4
OUT_ROOT = "exp_lora_sparge_12s"

# Larryvrh-style LoRAs only (skip lightx2v merged DiT checkpoints in loras/).
LORAS: list[tuple[str, str]] = [
    ("l01_v4_600_ema", TURBO_LORA_V4),
    ("l02_v1_850_ema", "full_or_original/minimax_h3_turbo_4step_ema_ckpt850.safetensors"),
    ("l03_pruned_v4_600_ema", "pruned_comfyui/RECOMMENDED_pruned_V4_ckpt600_EMA.safetensors"),
    ("l04_v1_500", "pruned_comfyui/minimax_h3_turbo_4step_ckpt500_V1.safetensors"),
    ("l05_v1_850_pruned", "pruned_comfyui/minimax_h3_turbo_4step_ckpt850_V1.safetensors"),
]


def _slug_lora(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", Path(name).stem)[:48]


def graph_for(label: str, lora_name: str) -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=CN_VOICE_TEXT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        turbo_lora=lora_name,
        teacache=False,
        tespeed=False,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        image="",
        output_prefix=f"{OUT_ROOT}/{label}",
    )
    g = build_graph(p)
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {"model": ["1", 0], "topk": 0.5, "dense_first_steps": 0, "num_layers": 50},
    )
    return g


def run_one(base_url: str, label: str, lora_name: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, graph_for(label, lora_name))
        print(f"[{label}] lora={lora_name} prompt_id={pid}", flush=True)
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        return {
            "label": label,
            "lora": lora_name,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "lora": lora_name,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:4000],
        }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*", help="Subset labels, e.g. l02_v1_850_ema")
    args = ap.parse_args()

    health(args.comfy)
    loras = LORAS
    if args.only:
        allow = set(args.only)
        loras = [(l, n) for l, n in LORAS if l in allow]

    out_dir = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"lora sparge shootout: {WIDTH}x{HEIGHT} length={LENGTH} steps={STEPS} n={len(loras)}",
        flush=True,
    )

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = run_one(args.comfy, "_warmup", TURBO_LORA_V4)
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup*"):
            p.unlink(missing_ok=True)

    results = []
    for label, lora_name in loras:
        print(f"=== {label} ===", flush=True)
        r = run_one(args.comfy, label, lora_name)
        results.append(r)
        print(
            json.dumps(
                {k: r.get(k) for k in ("label", "lora", "status", "elapsed_sec", "error")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    base_t = next(
        (r["elapsed_sec"] for r in results if r["label"] == "l01_v4_600_ema" and r.get("elapsed_sec")),
        None,
    )
    summary = []
    for r in results:
        row = {
            "label": r["label"],
            "lora": r["lora"],
            "elapsed_sec": r.get("elapsed_sec"),
            "status": r.get("status"),
        }
        if base_t and r.get("elapsed_sec") and r.get("status") == "success":
            row["speedup_vs_v4_600"] = round(base_t / r["elapsed_sec"], 3)
        summary.append(row)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "seed": SEED,
            "stack": "4-step TurboLoRA + Sage2 + Sparge0.5 (no TeaCache/TE-Speed)",
            "prompt": "Shanghai Bund rainy Mandarin dialogue",
            "out": f"ComfyUI-master_cp/output/{OUT_ROOT}",
        },
        "summary": summary,
        "results": results,
    }
    meta = ROOT / "experiments" / "sparse_attn" / "output" / "ab_lora_sparge_12s.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {meta}", flush=True)


if __name__ == "__main__":
    main()
