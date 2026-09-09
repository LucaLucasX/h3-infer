#!/usr/bin/env python3
"""Smoke / demo runner for PDD Acc on ComfyUI-latest (experiment only).

Does NOT modify h3_graph.py or production Turbo/Sparge graphs.
Recipe: FL2VA int8 + MiniMaxH3PDDAccApply (nfe=8) + euler + CFG1 + shift 12/3.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "report"))
import _bootstrap_h3_graph  # noqa: F401

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import node

# Full (non-pruned) FL2VA — matches Acc-FL2VA LoRA.
UNET = "minimax_h3_fl2va_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_V = "minimax_h3_video_vae_fp16.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"
PDD_FILE = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"

WIDTH, HEIGHT = 1344, 768
LENGTH = 124  # shorter smoke default; override via CLI
FPS = 24.0
SEED = 42
NFE = "8"


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s, flags=re.UNICODE).strip("_")
    return (s or "case")[:n]


def build_pdd_t2v(
    *,
    prompt: str,
    out_prefix: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    nfe: str,
    pdd_file: str,
    sparge: bool = False,
    sparge_topk: float = 0.5,
) -> dict:
    """Native H3 T2V + PDD Acc on ComfyUI-latest schema (no Turbo/Kitchen).

    When sparge=True: MiniMaxH3SpargeAttnPatchExp (prod topk=0.5, audio_dense).
    Dense path inside Sparge uses SageAttention 2.x — no separate Sage node.
    """
    g: dict = {}
    g["1"] = node("UNETLoader", {"unet_name": UNET, "weight_dtype": "default"})
    g["2"] = node("CLIPLoader", {"clip_name": CLIP, "type": "minimax", "device": "default"})
    g["3"] = node("VAELoader", {"vae_name": VAE_V})
    g["4"] = node("VAELoader", {"vae_name": VAE_A})

    model_ref: list = ["1", 0]
    if sparge:
        g["5"] = node(
            "MiniMaxH3SpargeAttnPatchExp",
            {
                "model": model_ref,
                "topk": float(sparge_topk),
                "dense_first_steps": 0,
                "num_layers": 50,
                "audio_dense": True,
            },
        )
        model_ref = ["5", 0]

    g["6"] = node(
        "MiniMaxH3SigmaShift",
        {"model": model_ref, "shift_video": 12.0, "shift_audio": 3.0},
    )
    g["7"] = node(
        "MiniMaxH3PDDAccApply",
        {
            "model": ["6", 0],
            "pdd_file": pdd_file,
            "nfe": str(nfe),
            "lora_strength": 1.0,
            "head_strength": 1.0,
            "on_off_grid": "error",
        },
    )
    g["8"] = node(
        "MiniMaxH3ImageToVideo",
        {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
        },
    )
    g["9"] = node("BasicGuider", {"model": ["7", 0], "conditioning": ["8", 0]})
    g["10"] = node("KSamplerSelect", {"sampler_name": "euler"})
    g["11"] = node("RandomNoise", {"noise_seed": int(seed)})
    g["12"] = node(
        "SamplerCustomAdvanced",
        {
            "noise": ["11", 0],
            "guider": ["9", 0],
            "sampler": ["10", 0],
            "sigmas": ["7", 1],
            "latent_image": ["8", 1],
        },
    )
    g["13"] = node("VAEDecode", {"samples": ["12", 0], "vae": ["3", 0]})
    g["14"] = node("VAEDecodeAudio", {"samples": ["12", 0], "vae": ["4", 0]})
    g["15"] = node("CreateVideo", {"images": ["13", 0], "audio": ["14", 0], "fps": FPS})
    g["16"] = node(
        "SaveVideo",
        {
            "video": ["15", 0],
            "filename_prefix": out_prefix,
            "format": "mp4",
            "codec": "h264",
        },
    )
    return g


def _profiling(hist: dict, submitted_at: float) -> dict:
    result = {
        "submitted_at": round(submitted_at, 3),
        "queue_sec": None,
        "generation_sec": None,
        "total_sec": None,
    }
    messages = (hist.get("status") or {}).get("messages", [])
    ts = {}
    for m in messages:
        if isinstance(m, list) and len(m) > 1 and isinstance(m[1], dict):
            if isinstance(m[1].get("timestamp"), (int, float)):
                ts[m[0]] = m[1]["timestamp"] / 1000.0
    started, completed = ts.get("execution_start"), ts.get("execution_success")
    if started is not None:
        result["queue_sec"] = round(max(0.0, started - submitted_at), 3)
    if completed is not None and started is not None:
        result["generation_sec"] = round(max(0.0, completed - started), 3)
        result["total_sec"] = round(max(0.0, completed - submitted_at), 3)
    return result


def run_one(base: str, graph: dict, label: str) -> dict:
    t0 = time.perf_counter()
    try:
        submitted_at = time.time()
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
            "profiling": _profiling(hist, submitted_at),
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
    ap.add_argument("--comfy", default="http://127.0.0.1:8193")
    ap.add_argument("--prompt", default="A tiny mechanical bee crawls across a wooden desk, soft daylight, shallow depth of field.")
    ap.add_argument("--label", default="pdd8_smoke")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--length", type=int, default=LENGTH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--nfe", default=NFE, choices=["8", "4", "6"])
    ap.add_argument("--pdd-file", default=PDD_FILE)
    ap.add_argument("--out-root", default="exp_pdd_acc")
    args = ap.parse_args()

    health(args.comfy)
    out_dir = REPO / "ComfyUI-master_cp" / "output" / args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.label}_{_slug(args.prompt)}"
    prefix = f"{args.out_root}/{label}"
    print(
        f"PDD Acc smoke unet={UNET} pdd={args.pdd_file} nfe={args.nfe} "
        f"{args.width}x{args.height} len={args.length}",
        flush=True,
    )
    g = build_pdd_t2v(
        prompt=args.prompt,
        out_prefix=prefix,
        width=args.width,
        height=args.height,
        length=args.length,
        seed=args.seed,
        nfe=args.nfe,
        pdd_file=args.pdd_file,
    )
    r = run_one(args.comfy, g, label)
    meta = EXP / "logs" / f"{label}.json"
    meta.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: r.get(k) for k in ("label", "status", "elapsed_sec", "profiling", "error")}, ensure_ascii=False, indent=2))
    print(f"wrote {meta}")
    print(f"videos: {out_dir}")


if __name__ == "__main__":
    main()
