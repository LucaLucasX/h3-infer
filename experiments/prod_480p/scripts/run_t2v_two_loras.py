#!/usr/bin/env python3
"""Production t2v at 480p: Sage2+Sparge+Turbo, two LightX2V LoRAs (768p vs non-768).

Canvas: 864x480 (16:9, multiples of 32), 12s @ 24fps → 294 frames.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "prod_480p"
sys.path[:0] = [str(ROOT), str(ROOT / "report"), str(ROOT / "dev" / "sage_lora_sparge")]

import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from run_textcases_comfy4_tc_sparge import load_cases

WIDTH, HEIGHT, FPS = 864, 480, 24.0
SEED = 42

# 768p = production 4-step distilled for short-edge 768.
# non-768 = same LightX2V family without 768p in the name (8-step v1.0).
LORAS = [
    {
        "tag": "lora768_4step",
        "file": "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors",
        "steps": 4,
    },
    {
        "tag": "lora_no768_8step",
        "file": "lightx2v_fl2v/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "steps": 8,
    },
]


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def align_length(seconds: int) -> int:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    return min(candidates, key=lambda value: abs(value - requested))


def graph_for(out_root: str, label: str, prompt: str, *, length: int,
              lora: str, steps: int) -> dict:
    return build_graph(GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=length,
        fps=FPS,
        steps=steps,
        seed=SEED,
        turbo=True,
        turbo_lora=lora,
        attention_backend="sage_v2_memeff",
        sparge=True,
        sparge_topk=0.5,
        sampler="er_sde",
        scheduler="simple",
        output_prefix=f"{out_root}/{label}",
    ))


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base, pid, timeout=3600, poll_interval=2)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        err = None
        for msg in (hist.get("status") or {}).get("messages") or []:
            if isinstance(msg, (list, tuple)) and msg and msg[0] == "execution_error":
                payload = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
                err = (payload.get("exception_message") or str(payload))[:4000]
        return {
            "label": label,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": extract_outputs(hist),
            "error": err,
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
    ap.add_argument("--only", nargs="*", default=["o01", "o02", "o03"])
    ap.add_argument("--duration", type=int, default=12)
    ap.add_argument("--out-root", default="exp_prod_480p")
    args = ap.parse_args()

    length = align_length(args.duration)
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / args.out_root
    meta = EXP / "logs" / "t2v_480p_o01_o03_two_loras.json"
    health(args.comfy)

    all_cases = load_cases()
    allow = set(args.only)
    cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]
    if not cases:
        raise SystemExit(f"no cases matched --only {args.only}")

    print(
        f"prod 480p t2v: {WIDTH}x{HEIGHT} len={length} (~{args.duration}s) "
        f"sage2+sparge0.5 turbo n_cases={len(cases)} n_loras={len(LORAS)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for lora in LORAS:
        for c in cases:
            label = f"{lora['tag']}_{c['id']}_{_slug(c['title'])}"
            print(f"=== {label} steps={lora['steps']} ===", flush=True)
            r = run_one(
                args.comfy,
                graph_for(
                    args.out_root, label, c["prompt"],
                    length=length, lora=lora["file"], steps=lora["steps"],
                ),
                label,
            )
            r.update({
                "case_id": c["id"],
                "title": c["title"],
                "lora_tag": lora["tag"],
                "lora": lora["file"],
                "steps": lora["steps"],
            })
            results.append(r)
            print(
                json.dumps(
                    {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if r.get("status") != "success":
                break
        else:
            continue
        break

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success"]
    payload = {
        "stack": "prod_sage2_sparge0.5_turbo",
        "width": WIDTH,
        "height": HEIGHT,
        "length": length,
        "duration_sec": args.duration,
        "loras": LORAS,
        "n": len(results),
        "n_ok": len(ok),
        "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
        "results": results,
    }
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "n_ok", "mean_sec")}, ensure_ascii=False))
    print(f"wrote {meta}")
    print(f"videos: {out_dir}")
    if len(ok) != len(cases) * len(LORAS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
