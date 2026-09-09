#!/usr/bin/env python3
"""GPU0 decode microbench: TRT fp16 decoder vs production PyTorch VAE.

Does not start Comfy, does not load DiT, does not touch GPU1.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

COMFY = Path("/mnt/luca/H3_infer/ComfyUI-master_cp")
DEPS = COMFY / "custom_nodes" / "ComfyUI-H3VAE_TRT" / ".deps"
ENGINE = COMFY / "models" / "vae" / "minimax_h3_vae_decoder.engine"
PROD_VAE = Path("/big_models/comfyui-minimax-H3/vae/minimax_h3_video_vae_fp16.safetensors")
LOG = Path("/mnt/luca/H3_infer/experiments/h3_vae_trt/logs")
LOG.mkdir(parents=True, exist_ok=True)

os.chdir(COMFY)
sys.path[:0] = [str(DEPS), str(COMFY)]
os.environ["LD_LIBRARY_PATH"] = f"{DEPS}/tensorrt_libs:{os.environ.get('LD_LIBRARY_PATH', '')}"


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def bench_trt(z_shape: tuple[int, ...], warmup: int = 1, runs: int = 2) -> dict:
    import torch

    plugin = COMFY / "custom_nodes" / "ComfyUI-H3VAE_TRT"
    if str(plugin) not in sys.path:
        sys.path.insert(0, str(plugin))
    from minimax_trt_node import AutoEngineRunner, MiniMaxH3TRTVAE  # type: ignore

    runner = AutoEngineRunner(str(ENGINE))
    vae = MiniMaxH3TRTVAE(decoder_runner=runner).cuda().eval()
    z = torch.randn(*z_shape, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(warmup):
            out = vae.decode(z)
            _sync()
        times = []
        for _ in range(runs):
            _sync()
            t0 = time.perf_counter()
            out = vae.decode(z)
            _sync()
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    vae.offload_runners_to_ram()
    torch.cuda.empty_cache()
    return {
        "backend": "trt_fp16",
        "engine": str(ENGINE),
        "z_shape": list(z_shape),
        "out_shape": list(out.shape),
        "seconds": times,
        "best_s": min(times),
        "peak_gb": round(peak, 3),
    }


def bench_pytorch(z_shape: tuple[int, ...], warmup: int = 1, runs: int = 1) -> dict:
    import torch
    import folder_paths  # noqa: F401
    import comfy.utils
    import comfy.sd

    sd = comfy.utils.load_torch_file(str(PROD_VAE), safe_load=True)
    vae = comfy.sd.VAE(sd=sd)
    vae.first_stage_model.cuda().eval()
    # Comfy video VAE decode expects [B, C, T, H, W] latents via vae.decode
    z = torch.randn(*z_shape, device="cuda", dtype=torch.float16)
    samples = {"samples": z}
    with torch.no_grad():
        for _ in range(warmup):
            out = vae.decode(z)
            _sync()
        times = []
        for _ in range(runs):
            _sync()
            t0 = time.perf_counter()
            out = vae.decode(z)
            _sync()
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    del vae, sd
    torch.cuda.empty_cache()
    return {
        "backend": "pytorch_fp16",
        "vae": str(PROD_VAE),
        "z_shape": list(z_shape),
        "out_shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
        "seconds": times,
        "best_s": min(times),
        "peak_gb": round(peak, 3),
    }


def main() -> None:
    import json
    import torch

    # 1280x704, ~5s: spatial /16; temporal ~124/4
    z_shape = (1, 24, 31, 44, 80)
    print("gpu", torch.cuda.get_device_name(0), "free-ish after alloc tracking")
    results = {"z_shape": list(z_shape), "gpu": torch.cuda.get_device_name(0)}
    torch.cuda.reset_peak_memory_stats()
    results["trt"] = bench_trt(z_shape)
    print("trt", results["trt"])
    try:
        torch.cuda.reset_peak_memory_stats()
        results["pytorch"] = bench_pytorch(z_shape)
        print("pytorch", results["pytorch"])
    except Exception as e:
        results["pytorch_error"] = f"{type(e).__name__}: {e}"
        print("pytorch failed", results["pytorch_error"])
    out = LOG / "decode_t2v5s.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
