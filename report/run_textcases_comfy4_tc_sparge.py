#!/usr/bin/env python3
"""Run copied t2v cases on Comfy EXP: 4-step TurboLoRA + Sage2 + Sparge + pruned INT8.

Fixed canvas: 1344x768 (720p 16:9), 12s @ 24fps.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))
import _bootstrap_h3_graph  # noqa: F401 — load h3_graph from bytecode

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import TURBO_LORA_V4, GenerateParams, build_graph, node

WIDTH, HEIGHT = 1344, 768
LENGTH = 288
FPS = 24.0
STEPS = 4
SEED = 42
OUT_ROOT = "exp_textcases_12s_comfy4_sparge_audiodense"
OUT_DIR = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
META = ROOT / "experiments" / "sparse_attn" / "output" / "textcases_comfy4_sparge_audiodense.json"
SPARGE_TOPK = 0.5
AUDIO_DENSE = True
USE_SPARGE = True
TURBO_LORA = TURBO_LORA_V4


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def load_cases() -> list[dict]:
    cases: list[dict] = []
    man = json.loads((ROOT / "text-cases" / "official-27" / "manifest.json").read_text())
    for i, row in enumerate(man, 1):
        p = ROOT / "text-cases" / "official-27" / f"{row['case']}.txt"
        cases.append(
            {
                "id": f"o{i:02d}",
                "group": "official-27",
                "title": row.get("title") or row["case"],
                "prompt": p.read_text(encoding="utf-8").strip(),
            }
        )
    hh = json.loads((ROOT / "text-cases" / "happyhorse" / "cases.json").read_text())
    for i, row in enumerate(hh, 1):
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            pf = row.get("prompt_file") or ""
            local = ROOT / "text-cases" / "happyhorse" / Path(pf).name
            if local.exists():
                prompt = local.read_text(encoding="utf-8").strip()
        if not prompt:
            continue
        cases.append(
            {
                "id": f"h{i:02d}",
                "group": "happyhorse",
                "title": row.get("title") or row.get("case") or f"hh-{i}",
                "prompt": prompt,
            }
        )
    ho = json.loads((ROOT / "text-cases" / "H3-official-cases" / "cases.json").read_text())
    for i, row in enumerate(ho, 1):
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            pf = Path(row.get("prompt_file") or "")
            local = ROOT / "text-cases" / "H3-official-cases" / pf.name
            if local.exists():
                prompt = local.read_text(encoding="utf-8").strip()
        cases.append(
            {
                "id": f"e{row.get('id') or f'{i:02d}'}",
                "group": "h3-official-extra",
                "title": row.get("title") or row.get("case") or f"extra-{i}",
                "prompt": prompt,
            }
        )
    return [c for c in cases if c["prompt"]]


def graph_for(label: str, prompt: str) -> dict:
    p = GenerateParams(
        mode="t2v",
        prompt=prompt,
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
        output_prefix=f"{OUT_ROOT}/{label}",
    )
    g = build_graph(p)
    if USE_SPARGE:
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


def run_one(base: str, label: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph_for(label, prompt))
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
    global OUT_ROOT, OUT_DIR, META, STEPS, SPARGE_TOPK, AUDIO_DENSE, USE_SPARGE, TURBO_LORA

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--topk", type=float, default=SPARGE_TOPK)
    ap.add_argument("--audio-dense", action=argparse.BooleanOptionalAction, default=AUDIO_DENSE)
    ap.add_argument("--no-sparge", action="store_true", help="Sage2 MemEff only (skip Sparge patch)")
    ap.add_argument("--turbo-lora", default=None, help="LoRA filename under Comfy loras/")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    STEPS = int(args.steps)
    SPARGE_TOPK = float(args.topk)
    AUDIO_DENSE = bool(args.audio_dense)
    USE_SPARGE = not args.no_sparge
    if args.turbo_lora:
        TURBO_LORA = args.turbo_lora
    if args.out_root:
        OUT_ROOT = args.out_root
        OUT_DIR = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
        META = ROOT / "experiments" / "sparse_attn" / "output" / f"{OUT_ROOT}.json"

    health(args.comfy)
    cases = load_cases()
    if args.only:
        allow = set(args.only)
        cases = [c for c in cases if c["id"] in allow or c["title"] in allow]
    sparge_desc = f"topk={SPARGE_TOPK} audio_dense={AUDIO_DENSE}" if USE_SPARGE else "no_sparge(sage2)"
    print(
        f"textcases comfy turbo: {WIDTH}x{HEIGHT} length={LENGTH} steps={STEPS} n={len(cases)} "
        f"lora={TURBO_LORA} {sparge_desc} out={OUT_ROOT}",
        flush=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_warmup:
        print("=== warmup (discard) ===", flush=True)
        w = run_one(args.comfy, "_warmup", cases[0]["prompt"])
        print(json.dumps({k: w.get(k) for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        for p in OUT_DIR.glob("_warmup*"):
            p.unlink(missing_ok=True)

    results = []
    for c in cases:
        label = f"{c['id']}_{_slug(c['title'])}"
        print(f"=== {label} ({c['group']}) ===", flush=True)
        r = run_one(args.comfy, label, c["prompt"])
        r.update({"case_id": c["id"], "group": c["group"], "title": c["title"]})
        results.append(r)
        print(
            json.dumps(
                {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success" and r.get("elapsed_sec")]
    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "turbo": True,
            "turbo_lora": TURBO_LORA,
            "teacache": False,
            "sparge": USE_SPARGE,
            "sparge_topk": SPARGE_TOPK if USE_SPARGE else None,
            "audio_dense": AUDIO_DENSE if USE_SPARGE else None,
            "attention": "sage_v2_memeff",
            "unet": "pruned_int8",
            "out": str(OUT_DIR.relative_to(ROOT)),
        },
        "n": len(results),
        "n_ok": len(ok),
        "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
        "results": results,
    }
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "n_ok", "mean_sec")}, ensure_ascii=False))
    print(f"wrote {META}", flush=True)


if __name__ == "__main__":
    main()
