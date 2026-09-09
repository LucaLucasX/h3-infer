#!/usr/bin/env python3
"""In-memory batched RealESRGAN video upscaler (no PNG dump)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def ffprobe_video(path: Path) -> dict:
    def _one(entries: str, stream: str = "v:0") -> str:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                stream,
                "-show_entries",
                entries,
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()

    w, h = _one("stream=width,height").splitlines()
    fps = _one("stream=avg_frame_rate") or "24/1"
    aidx = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return {
        "w": int(w),
        "h": int(h),
        "fps": fps,
        "audio": bool(aidx.stdout.strip()),
    }


def load_upsampler(realesr_root: Path, denoise: float, device: torch.device):
    sys.path.insert(0, str(realesr_root))
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    weights = realesr_root / "weights"
    gen = weights / "realesr-general-x4v3.pth"
    wdn = weights / "realesr-general-wdn-x4v3.pth"
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
    )
    if denoise != 1:
        model_path = [str(gen), str(wdn)]
        dni_weight = [denoise, 1.0 - denoise]
    else:
        model_path = str(gen)
        dni_weight = None
    up = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        half=True,
        device=device,
    )
    up.model.eval()
    return up


@torch.inference_mode()
def enhance_nhwc(model, frames_nhwc: np.ndarray, device: torch.device, outscale: int) -> np.ndarray:
    h, w = frames_nhwc.shape[1:3]
    x = torch.from_numpy(np.ascontiguousarray(frames_nhwc))
    x = x.permute(0, 3, 1, 2).to(device, dtype=torch.float16, non_blocking=True)
    x = x.div_(255.0)
    y4 = model(x)
    oh, ow = int(h * outscale), int(w * outscale)
    y = F.interpolate(y4.float(), size=(oh, ow), mode="bicubic", align_corners=False)
    y = y.clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
    return y.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def upscale_video(
    *,
    src: Path,
    dst: Path,
    up,
    device: torch.device,
    batch: int,
    scale: int,
) -> dict:
    meta = ffprobe_video(src)
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    ow, oh = w * scale, h * scale
    frame_in = w * h * 3

    dec = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "0",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        bufsize=frame_in * batch,
    )
    enc_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{ow}x{oh}",
        "-framerate",
        fps,
        "-i",
        "pipe:0",
    ]
    if meta["audio"]:
        enc_cmd += [
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:a",
            "copy",
            "-shortest",
        ]
    enc_cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=ow * oh * 3 * batch)

    n = 0
    t_infer = 0.0
    try:
        while True:
            raw = dec.stdout.read(frame_in * batch)
            if not raw:
                break
            n_this = len(raw) // frame_in
            if n_this == 0:
                break
            frames = (
                np.frombuffer(raw, dtype=np.uint8, count=n_this * frame_in)
                .reshape(n_this, h, w, 3)
                .copy()
            )
            t0 = time.perf_counter()
            out = enhance_nhwc(up.model, frames, device, scale)
            torch.cuda.synchronize(device)
            t_infer += time.perf_counter() - t0
            enc.stdin.write(out.tobytes())
            n += n_this
    finally:
        dec.stdout.close()
        enc.stdin.close()
        dec.wait(timeout=60)
        enc.wait(timeout=120)
        if dec.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg decode failed: {dec.returncode}")
        if enc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg encode failed: {enc.returncode}")
    return {"frames": n, "infer_s": t_infer, "w": w, "h": h, "ow": ow, "oh": oh}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--realesr-root", default="/mnt/luca/H3_infer/tools/Real-ESRGAN")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--denoise", type=float, default=0.3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(src_dir.glob("*.mp4"))
    device = torch.device(f"cuda:{args.gpu}")
    print(
        f"realesr batched: n={len(videos)} src={src_dir} out={out_dir} "
        f"model=realesr-general-x4v3 scale={args.scale} denoise={args.denoise} "
        f"gpu={args.gpu} batch={args.batch}",
        flush=True,
    )
    t_load = time.perf_counter()
    up = load_upsampler(Path(args.realesr_root), args.denoise, device)
    print(f"model loaded in {time.perf_counter() - t_load:.2f}s  device={device}", flush=True)

    ok = 0
    for i, v in enumerate(videos, 1):
        dst = out_dir / v.name
        if dst.exists() and not args.overwrite:
            print(f"[{i}/{len(videos)}] skip exists: {v.name}", flush=True)
            ok += 1
            continue
        print(f"[{i}/{len(videos)}] start: {v.name}", flush=True)
        t0 = time.perf_counter()
        tmp = dst.with_suffix(".tmp.mp4")
        try:
            if tmp.exists():
                tmp.unlink()
            info = upscale_video(
                src=v, dst=tmp, up=up, device=device, batch=args.batch, scale=args.scale
            )
            tmp.replace(dst)
            dt = time.perf_counter() - t0
            ok += 1
            print(
                f"[{i}/{len(videos)}] done: {v.name}  {dt:.2f}s  "
                f"frames={info['frames']} infer={info['infer_s']:.2f}s  "
                f"{info['w']}x{info['h']} -> {info['ow']}x{info['oh']}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            dt = time.perf_counter() - t0
            if tmp.exists():
                tmp.unlink()
            print(f"[{i}/{len(videos)}] ERROR: {v.name}  {dt:.2f}s  {e}", flush=True)
    print(f"finished: ok={ok}/{len(videos)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
