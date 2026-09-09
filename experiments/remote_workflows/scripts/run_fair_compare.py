#!/usr/bin/env python3
"""Fair A/B: GPU0 U06 stack vs GPU1 production (L0 / L1-full).

Shared inputs:
  - 9 ref images, short-edge resize 720
  - 2 ref videos 12s, short-edge resize 540
  - 3 ref audios
  - output 1376x768, 12s (294 frames)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "dev" / "sage_lora_sparge"
EXP = ROOT / "experiments" / "vram_lossless"
INPUT = ROOT / "ComfyUI-master_cp" / "input"
sys.path[:0] = [str(ROOT), str(DEV), str(EXP / "scripts"), str(ROOT / "report")]

import _bootstrap_h3_graph  # noqa: F401
from comfy_client import extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node, _align_length
from inject_mem_patch import inject_lossless_mem
from loras import LORA_R2V

IMAGES = [
    "realref_bust_01.jpg", "realref_bust_02.jpg", "realref_bust_03.jpg",
    "realref_face_01.jpg", "realref_face_02.jpg", "realref_face_03.jpg",
    "realref_full_01.png", "realref_full_02.png", "realref_full_03.png",
]
VIDEOS = ["h3_test_video_12s.mp4", "h3_test_video_12s.mp4"]
WIDTH, HEIGHT, FPS = 1376, 768, 24.0
LENGTH = _align_length(int(round(12 * FPS)))
VIDEO_CAP = LENGTH
LORA_8STEP = "lightx2v_fl2v/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
PROMPT = (
    "Cinematic photorealistic forest scene with a woman walking naturally, "
    "strong reference continuity, no text or logos."
)


def pick_audios(n: int = 3) -> list[str]:
    files = sorted(INPUT.glob("api_*_audio_01.mp3"))
    if len(files) < n:
        files = sorted(INPUT.glob("*.mp3"))
    if len(files) < n:
        raise FileNotFoundError(f"need {n} mp3 in {INPUT}")
    return [f.name for f in files[:n]]


def scale_short_edge(w: int, h: int, short: int, multiple: int = 16) -> tuple[int, int]:
    s = min(w, h)
    if s <= short:
        tw, th = w, h
    else:
        scale = short / s
        tw = max(multiple, round(w * scale / multiple) * multiple)
        th = max(multiple, round(h * scale / multiple) * multiple)
    return tw, th


# Precomputed from realref assets (short edge -> 720 / 540)
IMG_SIZE = (1328, 720)   # from ~2816x1536 and ~1640x1500 family
VID_SIZE = (960, 540)    # from 1920x1080


def _resize_image_chain(g: dict, fname: str, nid: str, tw: int, th: int) -> list:
    load = f"{nid}_l"
    sc = f"{nid}_s"
    g[load] = node("LoadImage", {"image": fname})
    g[sc] = node("ImageScale", {
        "image": [load, 0],
        "upscale_method": "lanczos",
        "width": tw,
        "height": th,
        "crop": "disabled",
    })
    return [sc, 0]


def _resize_video_chain(g: dict, fname: str, nid: str, tw: int, th: int) -> list:
    load = f"{nid}_v"
    sc = f"{nid}_s"
    g[load] = node("VHS_LoadVideo", {
        "video": fname,
        "force_rate": FPS,
        "custom_width": 0,
        "custom_height": 0,
        "frame_load_cap": VIDEO_CAP,
        "skip_first_frames": 0,
        "select_every_nth": 1,
    })
    g[sc] = node("ImageScale", {
        "image": [load, 0],
        "upscale_method": "lanczos",
        "width": tw,
        "height": th,
        "crop": "disabled",
    })
    return [sc, 0]


def build_u06_graph(seed: int, prefix: str) -> dict:
    g: dict = {}
    g["1"] = node("UNETLoader", {
        "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "weight_dtype": "default",
    })
    g["2"] = node("CLIPLoader", {
        "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "type": "minimax", "device": "default",
    })
    g["3"] = node("VAELoader", {"vae_name": "minimax_h3_video_vae_fp16.safetensors"})
    g["4"] = node("VAELoader", {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"})

    model_ref: list = ["1", 0]
    g["5"] = node("ModelAttentionBackend", {"model": model_ref, "attention": "comfy kitchen attention"})
    model_ref = ["5", 0]
    g["6"] = node("LoraLoaderModelOnly", {"model": model_ref, "lora_name": LORA_8STEP, "strength_model": 0.75})
    model_ref = ["6", 0]
    g["7"] = node("MiniMaxLowVRAMAttention", {"model": model_ref, "head_chunks": 4})
    model_ref = ["7", 0]
    g["8"] = node("MiniMaxChunkFeedForward", {"model": model_ref, "chunks": 2, "seq_threshold": 4096})
    model_ref = ["8", 0]

    r2v: dict = {
        "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
        "prompt": PROMPT, "width": WIDTH, "height": HEIGHT, "length": LENGTH,
        "ref_image_size": "max",
    }
    for i, img in enumerate(IMAGES):
        r2v[f"ref_images.ref_image_{i}"] = _resize_image_chain(g, img, f"i{i}", *IMG_SIZE)
    for i, vid in enumerate(VIDEOS):
        r2v[f"ref_videos.ref_video_{i}"] = _resize_video_chain(g, vid, f"v{i}", *VID_SIZE)
    for i, aud in enumerate(pick_audios(3)):
        aid = f"a{i}"
        g[aid] = node("LoadAudio", {"audio": aud})
        r2v[f"ref_audios.ref_audio_{i}"] = [aid, 0]
    g["20"] = node("MiniMaxH3ReferenceToVideo", r2v)

    g["70"] = node("RandomNoise", {"noise_seed": seed})
    g["71"] = node("BasicGuider", {"model": model_ref, "conditioning": ["20", 0]})
    g["72"] = node("BasicScheduler", {"model": model_ref, "scheduler": "beta", "steps": 8, "denoise": 1.0})
    g["73"] = node("KSamplerSelect", {"sampler_name": "euler"})
    g["74"] = node("SamplerCustomAdvanced", {
        "noise": ["70", 0], "guider": ["71", 0], "sampler": ["73", 0],
        "sigmas": ["72", 0], "latent_image": ["20", 1],
    })
    g["80"] = node("VAEDecode", {"samples": ["74", 0], "vae": ["3", 0]})
    g["81"] = node("VAEDecodeAudio", {"samples": ["74", 0], "vae": ["4", 0]})
    g["82"] = node("VHS_VideoCombine", {
        "images": ["80", 0], "audio": ["81", 0], "frame_rate": FPS,
        "loop_count": 0, "filename_prefix": prefix,
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16,
        "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True,
    })
    return g


def build_prod_graph(seed: int, prefix: str, *, mem: str = "none") -> dict:
    """Production stack with shared preprocess. mem: none | l1 | kj."""
    if mem not in ("none", "l1", "kj"):
        raise ValueError(f"unknown mem {mem!r}")
    base = build_graph(GenerateParams(
        mode="r2v",
        prompt=PROMPT,
        width=WIDTH, height=HEIGHT, length=LENGTH, fps=FPS, steps=4,
        seed=seed, turbo=True, turbo_lora=LORA_R2V, turbo_strength=1.0,
        attention_backend="sage_v2_memeff", sparge=True, sparge_topk=0.5,
        sparge_dense_first_steps=0, sparge_num_layers=50, sparge_audio_dense=True,
        unet_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ref_images=IMAGES[:1], ref_videos=VIDEOS[:1], ref_audios=[],
        ref_image_size="match",
        output_prefix=prefix,
        kj_lowvram=(mem == "kj"),
    ))
    # Replace ref2v node 6 inputs with resized chains
    g = dict(base)
    r2v_inputs = dict(g["6"]["inputs"])
    for i, img in enumerate(IMAGES):
        r2v_inputs[f"ref_images.ref_image_{i}"] = _resize_image_chain(g, img, f"p{i}", *IMG_SIZE)
    for i, vid in enumerate(VIDEOS):
        r2v_inputs[f"ref_videos.ref_video_{i}"] = _resize_video_chain(g, vid, f"pv{i}", *VID_SIZE)
    for i, aud in enumerate(pick_audios(3)):
        aid = f"pa{i}"
        g[aid] = node("LoadAudio", {"audio": aud})
        r2v_inputs[f"ref_audios.ref_audio_{i}"] = [aid, 0]
    g["6"]["inputs"] = r2v_inputs
    if mem == "l1":
        g = inject_lossless_mem(g, chunk_rows=4096, head_chunks=4, mode="step1")
    return g


def gpu_mem(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def meminfo_mb() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            parts = rest.split()
            if key in ("MemTotal", "MemAvailable", "MemFree", "Cached", "SwapFree", "SwapTotal") and parts:
                out[key] = int(parts[0]) // 1024
    except OSError:
        pass
    return out


def proc_rss_mb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
            if line.startswith("VmSwap:"):
                pass
    except (OSError, ValueError):
        return 0
    return 0


def find_comfy_pid(port: int, gpu: int) -> int:
    pid_file = ROOT / "dev" / "sage_lora_sparge" / "logs" / f"comfyui_{port}.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip().splitlines()[0])
            if Path(f"/proc/{pid}").exists():
                return pid
        except ValueError:
            pass
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    for line in out:
        try:
            return int(line.strip())
        except ValueError:
            continue
    return 0


def run_case(name: str, base_url: str, graph: dict, gpu: int, seed: int) -> dict:
    port = int(base_url.rsplit(":", 1)[-1].split("/")[0])
    cpid = find_comfy_pid(port, gpu)
    info0 = meminfo_mb()
    peak = {
        "vram_mb": gpu_mem(gpu),
        "rss_mb": proc_rss_mb(cpid) if cpid else 0,
        "mem_used_mb": info0.get("MemTotal", 0) - info0.get("MemAvailable", 0),
        "mem_avail_min_mb": info0.get("MemAvailable", 0),
    }
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.5):
            peak["vram_mb"] = max(peak["vram_mb"], gpu_mem(gpu))
            if cpid:
                peak["rss_mb"] = max(peak["rss_mb"], proc_rss_mb(cpid))
            info = meminfo_mb()
            if "MemAvailable" in info and "MemTotal" in info:
                used = info["MemTotal"] - info["MemAvailable"]
                peak["mem_used_mb"] = max(peak["mem_used_mb"], used)
                if peak["mem_avail_min_mb"]:
                    peak["mem_avail_min_mb"] = min(peak["mem_avail_min_mb"], info["MemAvailable"])
                else:
                    peak["mem_avail_min_mb"] = info["MemAvailable"]

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, graph)
        entry = wait_prompt(base_url, pid, timeout=4 * 3600)
    except Exception as exc:
        stop.set()
        t.join(timeout=2)
        return {
            "name": name, "gpu": gpu, "seed": seed, "status": "error",
            "e2e_sec": round(time.perf_counter() - t0, 1),
            "peak_vram_mb": peak["vram_mb"],
            "peak_rss_mb": peak["rss_mb"],
            "peak_sys_used_mb": peak["mem_used_mb"],
            "min_sys_avail_mb": peak["mem_avail_min_mb"],
            "comfy_pid": cpid,
            "error": str(exc),
        }
    stop.set()
    t.join(timeout=2)
    elapsed = time.perf_counter() - t0
    st = entry.get("status", {})
    err = None
    for m in st.get("messages", []):
        if m[0] == "execution_error":
            err = m[1]
    exec_sec = None
    t_start = t_end = None
    for m in st.get("messages", []):
        if m[0] == "execution_start":
            t_start = m[1].get("timestamp")
        if m[0] == "execution_success":
            t_end = m[1].get("timestamp")
    if t_start and t_end:
        exec_sec = round((t_end - t_start) / 1000, 1)
    info1 = meminfo_mb()
    return {
        "name": name,
        "gpu": gpu,
        "seed": seed,
        "prompt_id": pid,
        "status": st.get("status_str"),
        "e2e_sec": round(elapsed, 1),
        "exec_sec": exec_sec,
        "peak_vram_mb": peak["vram_mb"],
        "peak_rss_mb": peak["rss_mb"],
        "peak_sys_used_mb": peak["mem_used_mb"],
        "min_sys_avail_mb": peak["mem_avail_min_mb"],
        "sys_total_mb": info1.get("MemTotal"),
        "idle_rss_mb": proc_rss_mb(cpid) if cpid else None,
        "comfy_pid": cpid,
        "outputs": extract_outputs(entry),
        "error": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u06-url", default="http://127.0.0.1:8190")
    ap.add_argument("--prod-url", default="http://127.0.0.1:8191")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default=str(ROOT / "experiments/remote_workflows/logs/fair_compare_ab.json"))
    ap.add_argument("--only", choices=["u06", "prod_l0", "prod_l1", "prod_kj", "prod_mem_ab", "prod_mem_seq", "all"], default="all")
    ap.add_argument("--n-videos", type=int, default=2, choices=[1, 2, 3])
    args = ap.parse_args()

    global VIDEOS
    VIDEOS = ["h3_test_video_12s.mp4"] * args.n_videos
    tag = f"{args.n_videos}v"

    if args.only in ("prod_kj", "prod_mem_seq"):
        health(args.u06_url)
    elif args.only == "prod_l1":
        health(args.prod_url)
    else:
        health(args.u06_url)
        health(args.prod_url)

    idle_pid = find_comfy_pid(int(args.u06_url.rsplit(":", 1)[-1].split("/")[0]), 0)
    idle = meminfo_mb()
    shared = {
        "output": f"{WIDTH}x{HEIGHT}", "frames": LENGTH, "seconds": LENGTH / FPS,
        "ref_images": 9, "img_short_edge": 720, "img_size": list(IMG_SIZE),
        "ref_videos": args.n_videos, "vid_short_edge": 540, "vid_size": list(VID_SIZE),
        "ref_audios": 3, "videos": VIDEOS, "seed": args.seed,
    }
    shared["idle"] = {
        "comfy_pid": idle_pid,
        "rss_mb": proc_rss_mb(idle_pid) if idle_pid else None,
        "vram_mb": gpu_mem(0),
        "sys_used_mb": idle.get("MemTotal", 0) - idle.get("MemAvailable", 0),
        "sys_avail_mb": idle.get("MemAvailable"),
        "sys_total_mb": idle.get("MemTotal"),
    }
    results = {"shared": shared, "runs": []}
    box: dict = {}

    def _go(key: str, fn) -> None:
        print(f"=== {key} start ===", flush=True)
        box[key] = fn()
        print(f"=== {key} {box[key].get('status')} e2e={box[key].get('e2e_sec')}s peak={box[key].get('peak_vram_mb')}MB ===", flush=True)

    workers: list[threading.Thread] = []
    if args.only in ("all", "u06"):
        workers.append(threading.Thread(target=_go, args=(
            "gpu0_u06",
            lambda: run_case("gpu0_u06", args.u06_url, build_u06_graph(args.seed, "fair_ab/u06"), gpu=0, seed=args.seed),
        ), daemon=False))
    # prod_l0 can overlap U06 (different GPU); prod_l1 must wait for GPU1
    if args.only in ("all", "prod_l0"):
        workers.append(threading.Thread(target=_go, args=(
            "gpu1_prod_l0",
            lambda: run_case("gpu1_prod_l0", args.prod_url, build_prod_graph(args.seed, "fair_ab/prod_l0", mem="none"), gpu=1, seed=args.seed),
        ), daemon=False))
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    for key in ("gpu0_u06", "gpu1_prod_l0"):
        if key in box:
            results["runs"].append(box[key])

    if args.only in ("all", "prod_l1"):
        print("=== GPU1 prod L1-full ===", flush=True)
        results["runs"].append(run_case(
            "gpu1_prod_l1", args.prod_url,
            build_prod_graph(args.seed, "fair_ab/prod_l1", mem="l1"),
            gpu=1, seed=args.seed,
        ))

    if args.only in ("prod_kj", "prod_mem_ab"):
        workers.append(threading.Thread(target=_go, args=(
            "gpu0_prod_kj",
            lambda: run_case(
                "gpu0_prod_kj", args.u06_url,
                build_prod_graph(args.seed, f"fair_ab/prod_kj_{tag}", mem="kj"),
                gpu=0, seed=args.seed,
            ),
        ), daemon=False))
    if args.only == "prod_mem_ab":
        workers.append(threading.Thread(target=_go, args=(
            "gpu1_prod_l1",
            lambda: run_case(
                "gpu1_prod_l1", args.prod_url,
                build_prod_graph(args.seed, f"fair_ab/prod_l1_{tag}", mem="l1"),
                gpu=1, seed=args.seed,
            ),
        ), daemon=False))
    if args.only == "prod_kj":
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        if "gpu0_prod_kj" in box:
            results["runs"].append(box["gpu0_prod_kj"])
    elif args.only == "prod_mem_seq":
        print("=== GPU0 prod KJ (single ComfyUI) ===", flush=True)
        results["runs"].append(run_case(
            "gpu0_prod_kj", args.u06_url,
            build_prod_graph(args.seed, f"fair_ab/ram_kj_{tag}", mem="kj"),
            gpu=0, seed=args.seed,
        ))
        print("=== GPU0 prod L1-full (same ComfyUI) ===", flush=True)
        results["runs"].append(run_case(
            "gpu0_prod_l1", args.u06_url,
            build_prod_graph(args.seed, f"fair_ab/ram_l1_{tag}", mem="l1"),
            gpu=0, seed=args.seed,
        ))

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(r.get("status") != "success" for r in results["runs"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
