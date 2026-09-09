#!/usr/bin/env python3
"""720p t2v: Sage2 + 4-step lightx2v fl2v LoRA + Sparge (audio_dense)."""
from __future__ import annotations

import argparse
import json
import re
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
from h3_graph import GenerateParams, H3_EFFECT_EMBEDDINGS, build_graph, node
from loras import LORA_T2V

WIDTH, HEIGHT = 1376, 768
LENGTH = 288
FPS = 24.0
STEPS = 4
SEED = 42
SPARGE_TOPK = 0.5
AUDIO_DENSE = True
TURBO_LORA = LORA_T2V


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def load_cases() -> list[dict]:
    sys.path.insert(0, str(REPO / "report"))
    from run_textcases_comfy4_tc_sparge import load_cases as _load

    return _load()


def graph_for(out_root: str, label: str, prompt: str, effect: str = "none") -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
        effect_embedding=effect,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=True,
        turbo_lora=TURBO_LORA,
        attention_backend="sage_v2_memeff",
        ref_images=[],
        ref_videos=[],
        image="",
        output_prefix=f"{out_root}/{label}",
    )
    g = build_graph(p)
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {
            "model": ["1", 0],
            "topk": float(SPARGE_TOPK),
            "dense_first_steps": 0,
            "num_layers": 50,
            "audio_dense": bool(AUDIO_DENSE),
        },
    )
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
    ap.add_argument("--out-root", default="dev_sage_lora_sparge")
    ap.add_argument(
        "--effect",
        default="none",
        choices=sorted(H3_EFFECT_EMBEDDINGS.keys()),
        help="H3 effect embedding preset; none keeps the original prompt",
    )
    args = ap.parse_args()

    out_root = args.out_root
    if args.effect != "none":
        out_root = f"{out_root}_fx_{args.effect}"
    out_dir = REPO / "ComfyUI-master_cp" / "output" / out_root
    meta = DEV / "logs" / f"{out_root}.json"
    health(args.comfy)
    cases = load_cases()
    if args.only:
        allow = set(args.only)
        cases = [c for c in cases if c["id"] in allow or c["title"] in allow]
    print(
        f"sage+lora+sparge t2v: {WIDTH}x{HEIGHT} len={LENGTH} steps={STEPS} "
        f"lora={TURBO_LORA} topk={SPARGE_TOPK} audio_dense={AUDIO_DENSE} "
        f"effect={args.effect} n={len(cases)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = run_one(args.comfy, graph_for(out_root, "_warmup", cases[0]["prompt"], args.effect), "_warmup")
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        for p in out_dir.glob("_warmup*"):
            p.unlink(missing_ok=True)

    results = []
    for c in cases:
        label = f"{c['id']}_{_slug(c['title'])}"
        print(f"=== {label} ({c['group']}) ===", flush=True)
        r = run_one(args.comfy, graph_for(out_root, label, c["prompt"], args.effect), label)
        r.update({"case_id": c["id"], "group": c["group"], "title": c["title"], "effect": args.effect})
        results.append(r)
        print(
            json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False),
            flush=True,
        )

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success" and r.get("elapsed_sec")]
    payload = {
        "effect": args.effect,
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
