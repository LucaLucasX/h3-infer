#!/usr/bin/env python3
"""Ref2V with ComfyUI-VDN-H3 8-step turbo. No Sparge, no lightx2v LoRA.

Same stack as run_t2v_cases.py, but loads the ref2va UNET.
ApplyVDNH3 is a MODEL patch — GitHub documents fl2va and ref2va bases both work.
Default output video: 1280x704 (704p). Override with --width/--height or --size.
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

from case_io import load_case_path, slug, stage_media
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node

UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 704
FPS, STEPS = 24.0, 8
SEED = 42
VDN_CKPT = "stage-dmd-step-250"
SIZE_PRESETS = {"704p": (1280, 704), "768p": (1376, 768)}

REF_IMAGES = [
    "01.png",
    "02.png",
    "i1.png",
    "realref_full_01.png",
    "realref_full_02.png",
    "realref_full_03.png",
    "realref_bust_01.jpg",
    "realref_bust_02.jpg",
]
VIDEOS_5S = ["h3_test_video_5s.mp4", "h3_test_video_5s_b.mp4"]
VIDEOS_12S = ["h3_test_video_12s.mp4", "h3_test_video_12s_b.mp4"]


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


def r2v_prompt(n_images: int, n_videos: int) -> str:
    pics = " ".join(f"<Picture {i}>" for i in range(1, n_images + 1))
    vids = " ".join(f"<Video {i}>" for i in range(1, n_videos + 1))
    refs = " ".join(x for x in (pics, vids) if x)
    return (
        f"Cinematic photorealistic scene matching {refs}. "
        "Natural motion, consistent identity and lighting, no text or logos."
    )


def inject_vdn(g: dict) -> dict:
    """UNET → ApplyVDNH3 → (optional KJ LowVRAM) → sigma shift."""
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
    consumer = "23" if "23" in g else "40"
    if consumer not in g or g[consumer]["inputs"].get("model") != ["1", 0]:
        raise RuntimeError(
            f"expected {consumer} to consume UNET [1,0], got "
            f"{g.get(consumer, {}).get('inputs')}"
        )
    g[consumer]["inputs"]["model"] = ["25", 0]
    return g


def graph_for(out_root: str, label: str, prompt: str, *, length: int,
              width: int, height: int,
              ref_images: list[str], ref_videos: list[str],
              kj_lowvram: bool) -> dict:
    g = build_graph(GenerateParams(
        mode="r2v",
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
        ref_images=list(ref_images),
        ref_videos=list(ref_videos),
        ref_image_size="match",
        output_prefix=f"{out_root}/{label}",
    ))
    return inject_vdn(g)


def cases_for(seconds: int) -> list[dict]:
    videos = VIDEOS_5S if seconds <= 5 else VIDEOS_12S
    kj = bool(seconds >= 12)
    return [
        {
            "label": f"ref2va_8img_2vid_{seconds}s",
            "prompt": r2v_prompt(8, 2),
            "ref_images": REF_IMAGES,
            "ref_videos": videos,
            "kj_lowvram": kj,
        },
        {
            "label": f"ref2va_2vid_{seconds}s",
            "prompt": r2v_prompt(0, 2),
            "ref_images": [],
            "ref_videos": videos,
            "kj_lowvram": kj,
        },
    ]


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base, pid, timeout=3 * 3600, poll_interval=2)
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


def _case_aliases(c: dict) -> set[str]:
    label = c["label"]
    stem = label.rsplit("_", 1)[0]  # drop 5s/12s
    return {label, stem, stem.replace("ref2va_", ""), c["label"].split("_", 1)[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--prompt", default=None,
                    help="Text prompt. Use with --images/--videos/--case.")
    ap.add_argument("--case", nargs="+", default=None,
                    help="Folder (images/videos + optional prompt.txt), or a .txt/.json prompt file.")
    ap.add_argument("--name", default="custom",
                    help="Output label for an ad-hoc run.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional built-in smoke labels (8img_2vid, 2vid). Ignored if --prompt/--case/--images/--videos is set.")
    ap.add_argument("--list", action="store_true", help="Print built-in r2v smokes and exit.")
    ap.add_argument("--images", nargs="*", default=None,
                    help="Reference images: Comfy input names or filesystem paths.")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Reference videos: Comfy input names or filesystem paths.")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                    help="Output video width. Default 1280 (704p).")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                    help="Output video height. Default 704 (720 aligned to 32).")
    ap.add_argument("--size", default=None,
                    help="Output video size: 704p, 768p, or WxH like 1376x768. Overrides --width/--height.")
    ap.add_argument("--out-root", default="exp_vdn_h3")
    ap.add_argument("--kj-lowvram", action="store_true",
                    help="Force KJ LowVRAM+ChunkFF (12s with ref video already enables it).")
    args = ap.parse_args()
    width, height = resolve_video_size(args.width, args.height, args.size)

    selected = cases_for(args.duration)
    if args.list:
        for c in selected:
            print(f"{c['label']}\timages={len(c['ref_images'])}\tvideos={len(c['ref_videos'])}")
        return

    input_dir = ROOT / "ComfyUI-master_cp" / "input"
    custom = bool(args.prompt or args.case or args.images is not None or args.videos is not None)
    if custom:
        selected = []
        if args.case:
            for path in args.case:
                selected.extend(load_case_path(path, ROOT))
        if not selected:
            selected = [{
                "id": slug(args.name),
                "title": args.name,
                "prompt": (args.prompt or "").strip(),
                "ref_images": [],
                "ref_videos": [],
                "group": "adhoc",
            }]
        if args.prompt:
            for c in selected:
                c["prompt"] = args.prompt.strip()
                c["title"] = args.name
                c["id"] = slug(args.name)
        if args.images is not None:
            for c in selected:
                c["ref_images"] = list(args.images)
        if args.videos is not None:
            for c in selected:
                c["ref_videos"] = list(args.videos)
        for c in selected:
            c["label"] = c.get("id") or slug(args.name)
            c["kj_lowvram"] = bool(args.duration >= 12)
            c["ref_images"] = stage_media(
                c.get("ref_images") or [], root=ROOT, input_dir=input_dir, subdir=f"vdn_case/{c['label']}"
            )
            c["ref_videos"] = stage_media(
                c.get("ref_videos") or [], root=ROOT, input_dir=input_dir, subdir=f"vdn_case/{c['label']}"
            )
            if not c.get("prompt"):
                c["prompt"] = r2v_prompt(len(c["ref_images"]), len(c["ref_videos"]))
            if not c["ref_images"] and not c["ref_videos"]:
                raise SystemExit("r2v needs images or videos: pass --images/--videos or a folder with media")
    else:
        if args.only:
            want = set(args.only)
            selected = [c for c in selected if want & _case_aliases(c)]
            if not selected:
                known = " ".join(x["label"] for x in cases_for(args.duration))
                raise SystemExit(f"no built-in smokes match {args.only}\navailable: {known}")
    if args.kj_lowvram:
        for c in selected:
            c["kj_lowvram"] = True

    length = align_length(args.duration)
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / args.out_root
    meta = EXP / "logs" / f"r2v_{args.duration}s_{width}x{height}.json"
    health(args.comfy)

    print(
        f"vdn r2v: unet={UNET} vdn={VDN_CKPT} turbo_adapter=on "
        f"{width}x{height} len={length} (~{args.duration}s) steps={STEPS} "
        f"sampler=er_sde scheduler=beta merge stream "
        f"no_sparge no_lightx2v n={len(selected)}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for c in selected:
        print(f"=== {c['label']} kj_lowvram={c['kj_lowvram']} ===", flush=True)
        r = run_one(
            args.comfy,
            graph_for(
                args.out_root, c["label"], c["prompt"], length=length,
                width=width, height=height,
                ref_images=c["ref_images"], ref_videos=c["ref_videos"],
                kj_lowvram=c["kj_lowvram"],
            ),
            c["label"],
        )
        r.update({
            "ref_images": c["ref_images"],
            "ref_videos": c["ref_videos"],
            "kj_lowvram": c["kj_lowvram"],
        })
        results.append(r)
        print(
            json.dumps(
                {k: r.get(k) for k in ("label", "status", "elapsed_sec", "kj_lowvram", "error")},
                ensure_ascii=False,
            ),
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
    if len(ok) != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
