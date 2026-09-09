#!/usr/bin/env python3
"""T2V o01/o02/o03 with ComfyUI-VDN-H3 8-step turbo. No Sparge, no lightx2v.

Default output video: 1280x704 (704p). Override with --width/--height or --size.
Pass a prompt with --prompt, or a file/folder with --case.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "vdn_h3"
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(ROOT),
    str(ROOT / "report"),
    str(ROOT / "dev" / "sage_lora_sparge"),
]

import _bootstrap_h3_graph  # noqa: F401

from case_io import adhoc_prompt, load_case_path, slug
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 704
FPS, STEPS = 24.0, 8
SEED = 42
VDN_CKPT = "stage-dmd-step-250"
SIZE_PRESETS = {"704p": (1280, 704), "768p": (1376, 768)}


def _slug(s: str, n: int = 60) -> str:
    return slug(s, n)


def align32(n: int) -> int:
    return max(32, int(round(n / 32) * 32))


def resolve_video_size(width: int, height: int, size: str | None) -> tuple[int, int]:
    if size:
        key = size.strip().lower()
        if key in SIZE_PRESETS:
            width, height = SIZE_PRESETS[key]
        elif "x" in key:
            w_s, h_s = key.split("x", 1)
            width, height = int(w_s), int(h_s)
        else:
            raise SystemExit(f"unknown --size {size}; use 704p, 768p, or 1280x704")
    tw, th = align32(width), align32(height)
    if (tw, th) != (width, height):
        print(f"output video {width}x{height} -> {tw}x{th} (H3 needs multiples of 32)", flush=True)
    return tw, th


def align_length(seconds: int) -> int:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    return min(candidates, key=lambda value: abs(value - requested))


def load_cases() -> list[dict]:
    from run_textcases_comfy4_tc_sparge import load_cases as _load

    return _load()


def graph_for(out_root: str, label: str, prompt: str, *, length: int,
              width: int, height: int, kj_lowvram: bool = False) -> dict:
    g = build_graph(GenerateParams(
        mode="t2v",
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        fps=FPS,
        steps=STEPS,
        seed=SEED,
        turbo=False,
        sparge=False,
        attention_backend="pytorch",
        sage_attention="disabled",
        sampler="er_sde",
        scheduler="beta",
        unet_name=UNET,
        kj_lowvram=kj_lowvram,
        output_prefix=f"{out_root}/{label}",
    ))
    g["25"] = node("ApplyVDNH3", {
        "model": ["1", 0],
        "vdn_checkpoint": VDN_CKPT,
        "apply_turbo_adapter": True,
        "strength": 1.0,
        "lora_mode": "merge",
        "branch_weights": "stream",
        "verbose": True,
        "attention_backend": "grouped",
    })
    # UNET → VDN → (optional KJ LowVRAM) → sigma shift. VDN owns attn.forward;
    # KJ can compose on top for head/FFN chunks.
    consumer = "23" if "23" in g else "40"
    if consumer not in g or g[consumer]["inputs"].get("model") != ["1", 0]:
        raise RuntimeError(
            f"expected {consumer} to consume UNET [1,0], got "
            f"{g.get(consumer, {}).get('inputs')}"
        )
    g[consumer]["inputs"]["model"] = ["25", 0]
    return g


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base, pid, timeout=7200, poll_interval=2)
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
    ap.add_argument("--prompt", default=None,
                    help="Run this text prompt (does not use the catalog).")
    ap.add_argument("--case", nargs="+", default=None,
                    help="Prompt .txt / .json / folder path. Not a catalog id.")
    ap.add_argument("--name", default="custom",
                    help="Output label when using --prompt.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional catalog ids (o01 o07 …). Ignored if --prompt/--case is set.")
    ap.add_argument("--list", action="store_true", help="Print catalog t2v ids and exit.")
    ap.add_argument("--duration", type=int, default=12)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                    help="Output video width. Default 1280 (704p).")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                    help="Output video height. Default 704 (720 aligned to 32).")
    ap.add_argument("--size", default=None,
                    help="Output video size: 704p, 768p, or WxH like 1376x768. Overrides --width/--height.")
    ap.add_argument("--out-root", default="exp_vdn_h3")
    ap.add_argument("--kj-lowvram", action="store_true",
                    help="UNET→VDN→KJ LowVRAM+ChunkFF.")
    args = ap.parse_args()
    width, height = resolve_video_size(args.width, args.height, args.size)

    all_cases = load_cases()
    if args.list:
        for c in all_cases:
            print(f"{c['id']}\t{c.get('group','')}\t{c['title']}")
        return

    if args.prompt or args.case:
        cases = []
        if args.prompt:
            cases.append(adhoc_prompt(args.prompt, args.name))
        for path in args.case or []:
            cases.extend(load_case_path(path, ROOT))
        meta_tag = cases[0]["id"] if len(cases) == 1 else f"n{len(cases)}"
    else:
        allow = set(args.only or ["o01", "o02", "o03"])
        cases = [c for c in all_cases if c["id"] in allow or c["title"] in allow]
        if not cases:
            known = " ".join(c["id"] for c in all_cases)
            raise SystemExit(f"no catalog ids matched {sorted(allow)}\navailable: {known}")
        meta_tag = "_".join(sorted(allow)[:1] + sorted(allow)[-1:])

    length = align_length(args.duration)
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / args.out_root
    meta = EXP / "logs" / f"t2v_{args.duration}s_{width}x{height}_{meta_tag}.json"
    health(args.comfy)

    print(
        f"vdn t2v: unet={UNET} vdn={VDN_CKPT} turbo_adapter=on "
        f"{width}x{height} len={length} (~{args.duration}s) steps={STEPS} "
        f"sampler=er_sde scheduler=beta merge stream "
        f"kj_lowvram=False no_sparge no_lightx2v n={len(cases)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for c in cases:
        label = f"{c['id']}_{_slug(c['title'])}"
        print(f"=== {label} ===", flush=True)
        r = run_one(
            args.comfy,
            graph_for(args.out_root, label, c["prompt"], length=length,
                      width=width, height=height, kj_lowvram=args.kj_lowvram),
            label,
        )
        r.update({"case_id": c["id"], "title": c["title"], "group": c.get("group")})
        results.append(r)
        print(
            json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")}, ensure_ascii=False),
            flush=True,
        )
        if r.get("status") != "success":
            break

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success"]
    payload = {
        "unet": UNET,
        "vdn": VDN_CKPT,
        "stack": "vdn8_er_sde_beta_merge_stream_no_sparge_no_lightx2v",
        "lora_mode": "merge",
        "kj_lowvram": False,
        "width": width,
        "height": height,
        "length": length,
        "duration_sec": args.duration,
        "steps": STEPS,
        "sampler": "er_sde",
        "scheduler": "beta",
        "n": len(results),
        "n_ok": len(ok),
        "mean_sec": round(sum(ok) / len(ok), 2) if ok else None,
        "results": results,
    }
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n", "n_ok", "mean_sec", "vdn")}, ensure_ascii=False))
    print(f"wrote {meta}")
    print(f"videos: {out_dir}")
    if len(ok) != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
