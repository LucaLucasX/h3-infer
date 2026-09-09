#!/usr/bin/env python3
"""Production go-live timings: 1280x704, 4-step LoRA + Sage2 + Sparge, 5s/12s.

Warmup each UNET before timed cases. 12s + reference videos enable
MiniMaxLowVRAMAttention + MiniMaxChunkFeedForward.
Ref videos longer than the output are truncated to the generation frame count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "prod_704p"
sys.path[:0] = [str(ROOT), str(ROOT / "report"), str(ROOT / "dev" / "sage_lora_sparge")]

import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph
from loras import LORA_R2V, LORA_T2V
from run_textcases_comfy4_tc_sparge import load_cases

WIDTH, HEIGHT, FPS, STEPS, SEED = 1280, 704, 24.0, 4, 42
OUT_ROOT = "exp_prod_704p"
# Resize is inside MiniMax nodes (short-edge 720 image / 540 video, long side align 32).

FIRST_FRAME = "01.png"          # 2816x1536
LAST_FRAME = "02.png"           # 2816x1536
REF_IMAGES = [                  # all > 1280x704
    "01.png",
    "02.png",
    "i1.png",
    "realref_full_01.png",
    "realref_full_02.png",
    "realref_full_03.png",
    "realref_bust_01.jpg",
    "realref_bust_02.jpg",
]
# 1080p > 1280x704. Source longer than the output is truncated to generation frames.
VIDEOS_5S = ["h3_test_video_5s.mp4", "h3_test_video_5s_b.mp4"]
VIDEOS_12S = ["h3_test_video_12s.mp4", "h3_test_video_12s_b.mp4"]


def align_length(seconds: int) -> int:
    requested = round(seconds * FPS)
    k = max(0, round((requested - 5) / 17))
    candidates = [max(5, 17 * x + 5) for x in (k - 1, k, k + 1) if x >= 0]
    return min(candidates, key=lambda value: abs(value - requested))


def o01_prompt() -> str:
    for c in load_cases():
        if c["id"] == "o01":
            return c["prompt"]
    raise SystemExit("o01 prompt not found")


def r2v_prompt(n_images: int, n_videos: int) -> str:
    pics = " ".join(f"<Picture {i}>" for i in range(1, n_images + 1))
    vids = " ".join(f"<Video {i}>" for i in range(1, n_videos + 1))
    refs = " ".join(x for x in (pics, vids) if x)
    return (
        f"Cinematic photorealistic scene matching {refs}. "
        "Natural motion, consistent identity and lighting, no text or logos."
    )


def _case(*, label: str, mode: str, seconds: int, prompt: str,
          image: str = "", last_frame: str | None = None,
          ref_images: list[str] | None = None, ref_videos: list[str] | None = None,
          lora: str, seed: int = SEED) -> dict:
    refs_v = list(ref_videos or [])
    return {
        "label": label,
        "mode": mode,
        "seconds": seconds,
        "prompt": prompt,
        "image": image,
        "last_frame": last_frame,
        "ref_images": list(ref_images or []),
        "ref_videos": refs_v,
        "lora": lora,
        "seed": seed,
        # 12s + any ref video: KJ LowVRAM attn + chunked FFN, otherwise 32GB OOMs in sampling
        "kj_lowvram": bool(seconds >= 12 and refs_v),
    }


def timed_cases(prompt_t2v: str) -> list[dict]:
    # Group by UNET so warmup stays hot: all fl2va, then all ref2va.
    rows = []
    for seconds in (5, 12):
        rows.append(_case(
            label=f"fl2va_text_{seconds}s", mode="t2v", seconds=seconds,
            prompt=prompt_t2v, lora=LORA_T2V,
        ))
        rows.append(_case(
            label=f"fl2va_keyframe_{seconds}s", mode="i2v", seconds=seconds,
            prompt=prompt_t2v, image=FIRST_FRAME, last_frame=LAST_FRAME, lora=LORA_T2V,
        ))
    for seconds in (5, 12):
        videos = VIDEOS_5S if seconds == 5 else VIDEOS_12S
        rows.append(_case(
            label=f"ref2va_8img_2vid_{seconds}s", mode="r2v", seconds=seconds,
            prompt=r2v_prompt(8, 2), ref_images=REF_IMAGES, ref_videos=videos, lora=LORA_R2V,
        ))
        rows.append(_case(
            label=f"ref2va_2vid_{seconds}s", mode="r2v", seconds=seconds,
            prompt=r2v_prompt(0, 2), ref_videos=videos, lora=LORA_R2V,
        ))
    return rows


def warmup_for(selected: list[dict], prompt_t2v: str) -> list[dict]:
    need_fl2va = any(c["mode"] in ("t2v", "i2v") for c in selected)
    need_r2v = any(c["mode"] == "r2v" for c in selected)
    out = []
    if need_fl2va:
        w = _case(label="warmup_fl2va_text_5s", mode="t2v", seconds=5,
                  prompt=prompt_t2v, lora=LORA_T2V, seed=0)
        out.append(w)
    if need_r2v:
        w = _case(label="warmup_ref2va_2vid_5s", mode="r2v", seconds=5,
                  prompt=r2v_prompt(0, 2), ref_videos=VIDEOS_5S, lora=LORA_R2V, seed=0)
        out.append(w)
    return out


def graph_for(case: dict, length: int) -> dict:
    g = build_graph(GenerateParams(
        mode=case["mode"],
        prompt=case["prompt"],
        width=WIDTH,
        height=HEIGHT,
        length=length,
        fps=FPS,
        steps=STEPS,
        seed=int(case.get("seed", SEED)),
        turbo=True,
        turbo_lora=case["lora"],
        turbo_strength=1.0,
        attention_backend="sage_v2_memeff",
        sparge=True,
        sparge_topk=0.5,
        sparge_dense_first_steps=0,
        sparge_num_layers=50,
        sparge_audio_dense=True,
        kj_lowvram=bool(case.get("kj_lowvram")),
        image=case["image"],
        last_frame=case["last_frame"],
        ref_images=list(case["ref_images"]),
        ref_videos=list(case["ref_videos"]),
        ref_image_size="match",
        output_prefix=f"{OUT_ROOT}/{case['label']}",
    ))
    return g


def run_one(base: str, case: dict) -> dict:
    length = align_length(case["seconds"])
    label = case["label"]
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base, graph_for(case, length))
        print(
            f"[{label}] prompt_id={pid} {WIDTH}x{HEIGHT} len={length} "
            f"kj_lowvram={bool(case.get('kj_lowvram'))} "
            f"resize=node-internal img720/vid540",
            flush=True,
        )
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
            "mode": case["mode"],
            "seconds": case["seconds"],
            "length": length,
            "lora": case["lora"],
            "kj_lowvram": bool(case.get("kj_lowvram")),
            "image": case["image"] or None,
            "last_frame": case["last_frame"],
            "ref_images": case["ref_images"],
            "ref_videos": case["ref_videos"],
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": extract_outputs(hist),
            "error": err,
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "mode": case["mode"],
            "seconds": case["seconds"],
            "length": length,
            "kj_lowvram": bool(case.get("kj_lowvram")),
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:4000],
        }


def _dump(meta: Path, warmups: list[dict], results: list[dict]) -> None:
    ok = [r for r in results if r.get("status") == "success"]
    payload = {
        "stack": "prod_4step_sage2_sparge0.5",
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "lora_t2v": LORA_T2V,
        "lora_r2v": LORA_R2V,
        "warmup": True,
        "kj_lowvram_when": "12s generation with ref_videos",
        "resize": {
            "mode": "minimax_node_internal",
            "image_short": 720,
            "video_short": 540,
            "align": 32,
            "note": "MiniMaxH3*ToVideo: short-edge 720/540, long side align 32, never upscale; no graph ImageScale",
        },
        "inputs": {
            "first_frame": FIRST_FRAME,
            "last_frame": LAST_FRAME,
            "ref_images": REF_IMAGES,
            "ref_videos_5s": VIDEOS_5S,
            "ref_videos_12s": VIDEOS_12S,
        },
        "n": len(results),
        "n_ok": len(ok),
        "warmups": warmups,
        "results": results,
    }
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    health(args.comfy)
    prompt_t2v = o01_prompt()
    selected = timed_cases(prompt_t2v)
    if args.only:
        want = set(args.only)
        selected = [c for c in selected if c["label"] in want]
        if not selected:
            raise SystemExit(f"no cases match --only {args.only}")

    warm_list = warmup_for(selected, prompt_t2v)
    print(
        f"prod 704p go-live: {WIDTH}x{HEIGHT} 4step+sage2+sparge0.5 "
        f"warmup={len(warm_list)} timed={len(selected)} "
        f"resize=node-internal img720/vid540",
        flush=True,
    )
    meta = EXP / "logs" / "t2v_i2v_r2v_704p_5s_12s.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT).mkdir(parents=True, exist_ok=True)

    warmups: list[dict] = []
    results: list[dict] = []

    def do_warmup(case: dict) -> int:
        print(f"=== WARMUP {case['label']} ===", flush=True)
        row = run_one(args.comfy, case)
        row["warmup"] = True
        warmups.append(row)
        print(
            json.dumps(
                {k: row.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                ensure_ascii=False,
            ),
            flush=True,
        )
        _dump(meta, warmups, results)
        if row.get("status") != "success":
            print(f"wrote {meta} (warmup failed)", flush=True)
            return 1
        return 0

    warmed = set()
    for case in selected:
        family = "r2v" if case["mode"] == "r2v" else "fl2va"
        if family not in warmed:
            for w in warm_list:
                w_family = "r2v" if w["mode"] == "r2v" else "fl2va"
                if w_family == family:
                    if do_warmup(w) != 0:
                        return 1
            warmed.add(family)
        print(
            f"=== {case['label']} kj_lowvram={case['kj_lowvram']} ===",
            flush=True,
        )
        row = run_one(args.comfy, case)
        results.append(row)
        print(
            json.dumps(
                {k: row.get(k) for k in ("label", "status", "elapsed_sec", "kj_lowvram", "error")},
                ensure_ascii=False,
            ),
            flush=True,
        )
        _dump(meta, warmups, results)
        if row.get("status") != "success":
            print(f"wrote {meta} (stopped on failure)", flush=True)
            return 1

    ok = [r["elapsed_sec"] for r in results if r.get("status") == "success"]
    print(json.dumps({"n": len(results), "n_ok": len(ok), "secs": ok}, ensure_ascii=False))
    print(f"wrote {meta}")
    print(f"videos: {ROOT / 'ComfyUI-master_cp' / 'output' / OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
