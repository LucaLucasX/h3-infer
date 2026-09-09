#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def has_audio(video: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video),
    ]
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return bool(out.stdout.strip())


def fps_of(video: Path) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    fps = out.stdout.strip()
    return fps or "24/1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--realesr-root", default="/mnt/luca/H3_infer/tools/Real-ESRGAN")
    ap.add_argument("--python", default="/mnt/luca/H3_infer/.venv_sage_bench/bin/python")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--denoise", type=float, default=0.3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    realesr_root = Path(args.realesr_root)
    py = Path(args.python)

    videos = sorted(src_dir.glob("*.mp4"))
    print(
        f"realesr upscaler: n={len(videos)} src={src_dir} out={out_dir} "
        f"model=realesr-general-x4v3 scale={args.scale} denoise={args.denoise} gpu={args.gpu} tile={args.tile}",
        flush=True,
    )
    if not videos:
        return

    # ensure script is executable via chosen python
    if not (realesr_root / "inference_realesrgan.py").exists():
        raise FileNotFoundError(realesr_root / "inference_realesrgan.py")

    out_dir.mkdir(parents=True, exist_ok=True)
    # copy source video list for traceability
    (out_dir / "_source_dir.txt").write_text(str(src_dir) + "\n", encoding="utf-8")

    ok = 0
    for i, v in enumerate(videos, 1):
        out_v = out_dir / v.name
        if out_v.exists() and not args.overwrite:
            print(f"[{i}/{len(videos)}] skip exists: {v.name}", flush=True)
            continue
        print(f"[{i}/{len(videos)}] start: {v.name}", flush=True)
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="realesr_video_") as td:
                td_path = Path(td)
                in_frames = td_path / "in"
                out_frames = td_path / "out"
                in_frames.mkdir(parents=True, exist_ok=True)
                out_frames.mkdir(parents=True, exist_ok=True)

                run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(v),
                        "-vsync",
                        "0",
                        str(in_frames / "%08d.png"),
                    ]
                )

                run(
                    [
                        str(py),
                        str(realesr_root / "inference_realesrgan.py"),
                        "-n",
                        "realesr-general-x4v3",
                        "-i",
                        str(in_frames),
                        "-o",
                        str(out_frames),
                        "-s",
                        str(args.scale),
                        "--suffix",
                        "out",
                        "--denoise_strength",
                        str(args.denoise),
                        "--tile",
                        str(args.tile),
                        "--gpu-id",
                        str(args.gpu),
                    ]
                )

                fps = fps_of(v)
                frame_pattern = str(out_frames / "%08d_out.png")
                if has_audio(v):
                    run(
                        [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-framerate",
                            fps,
                            "-i",
                            frame_pattern,
                            "-i",
                            str(v),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "faster",
                            "-crf",
                            "17",
                            "-pix_fmt",
                            "yuv420p",
                            "-c:a",
                            "copy",
                            str(out_v),
                        ]
                    )
                else:
                    run(
                        [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-framerate",
                            fps,
                            "-i",
                            frame_pattern,
                            "-c:v",
                            "libx264",
                            "-preset",
                            "faster",
                            "-crf",
                            "17",
                            "-pix_fmt",
                            "yuv420p",
                            str(out_v),
                        ]
                    )
            dt = time.perf_counter() - t0
            ok += 1
            print(f"[{i}/{len(videos)}] done: {v.name}  {dt:.2f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"[{i}/{len(videos)}] ERROR: {v.name}  {dt:.2f}s  {e}", flush=True)

    print(f"finished: ok={ok}/{len(videos)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
