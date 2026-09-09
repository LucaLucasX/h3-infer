#!/usr/bin/env python3
"""Official-native 1344×768 (~1.0 MP short-edge 768) complex-scene retake.

FORCE_CUDA off. Modes:
  - fl2va text-only (t2v / no keyframes) — full accel matrix like before
  - fl2va first+last frames — 8 / 20
  - ref2va image refs — 8 / 20
  - ref2va video ref — 8 / 20

Timing: wall-clock submit → history done (TE/VAE/IO included).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import TURBO_LORA_V4, GenerateParams, build_graph

# Official MiniMax H3 ImageToVideo / ReferenceToVideo defaults (short edge 768)
WIDTH, HEIGHT = 1344, 768
LENGTH = 124  # ~5s @ 24fps
SEED = 42
OUT_ROOT = "h3_official_1344x768"

COMPLEX_TEXT = """Cinematic short film, photorealistic, 24fps, shallow depth of field.

Cast:
- A 28-year-old East Asian woman with short black bob hair, red trench coat, carrying a black umbrella
- A 35-year-old man in a navy wool coat and glasses, holding a paper coffee cup
- Background extras: rainy Tokyo street crowd with neon reflections

Timeline:
[0.0s-1.2s] Close-up: raindrops hit the woman's umbrella; neon kanji signs blur in bokeh; soft jazz piano begins; distant traffic hiss.
[1.2s-2.5s] Medium shot: she lowers the umbrella slightly, looks across the crosswalk; the man on the opposite curb notices her; camera slow push-in; footsteps splash in puddles.
[2.5s-3.8s] Tracking shot beside them as they walk toward each other under warm streetlights; their reflections merge in wet asphalt; dialogue SFX: muffled city chatter, one bicycle bell.
[3.8s-5.0s] Over-the-shoulder: they stop face to face under a glowing convenience-store awning; she smiles faintly; he offers the coffee; light flare; music swells then softens; rain continues.

Camera language: naturalistic handheld with subtle stabilization, no jump cuts, no text overlays, no logos.
Audio: stereo rain bed, soft jazz trio, footsteps, one clear bicycle bell at 3.0s, no voiceover.
Lighting: night rain, cyan/magenta neon mixed with warm tungsten shop lights, film grain light.
""".strip()

COMPLEX_FL2VA = """Continue the rainy Tokyo night scene between the provided first and last frames.
Keep identity, wardrobe, and lighting consistent with both keyframes.
Photorealistic cinematic 24fps, shallow DOF, stereo rain + soft jazz, no text overlays.
Motion: she crosses toward him under neon reflections; they meet under a convenience-store awning; he offers coffee.
""".strip()

COMPLEX_REF_IMG = """Using reference <Picture 1> and <Picture 2> for character / appearance lock.
Cinematic rainy Tokyo night street, photorealistic, 24fps.
A woman in a red trench coat with umbrella and a man in a navy coat with coffee cup walk toward each other under neon, then meet under a convenience-store awning.
Audio: stereo rain, soft jazz, footsteps, one bicycle bell. No logos or text.
""".strip()

COMPLEX_REF_VID = """Using motion / style reference <Video 1>.
Cinematic rainy Tokyo night street meet-cute, photorealistic 24fps.
Woman with red trench coat and umbrella; man with navy coat and coffee; neon reflections; they walk, meet, exchange coffee.
Audio: stereo rain, soft jazz, footsteps, bicycle bell. No logos or text.
""".strip()

FIRST_FRAME = "01.png"
LAST_FRAME = "02.png"
REF_IMAGES = ["example.png", "i1.png"]
REF_VIDEO = "minimax_h3_ref_video_1.mp4"

# (label, overrides)
CASES: list[tuple[str, dict]] = [
    # --- fl2va text-only (original suite) ---
    ("01_text_direct_8", dict(kind="text", mode="t2v", steps=8, attention="pytorch", turbo=False)),
    ("02_text_direct_20", dict(kind="text", mode="t2v", steps=20, attention="pytorch", turbo=False)),
    ("03_text_twostep_8", dict(kind="text", mode="twostep", steps=8, attention="pytorch", turbo=False)),
    ("04_text_twostep_20", dict(kind="text", mode="twostep", steps=20, attention="pytorch", turbo=False)),
    ("05_text_twostep_sage2_8", dict(kind="text", mode="twostep", steps=8, attention="sage_v2_memeff", turbo=False)),
    ("06_text_twostep_sage2_20", dict(kind="text", mode="twostep", steps=20, attention="sage_v2_memeff", turbo=False)),
    ("07_text_twostep_lora_8", dict(kind="text", mode="twostep", steps=8, attention="pytorch", turbo=True)),
    ("08_text_twostep_lora_sage2_8", dict(kind="text", mode="twostep", steps=8, attention="sage_v2_memeff", turbo=True)),
    ("09_text_twostep_sage3_8", dict(kind="text", mode="twostep", steps=8, attention="sage_v3", turbo=False)),
    ("10_text_twostep_sage3_20", dict(kind="text", mode="twostep", steps=20, attention="sage_v3", turbo=False)),
    ("11_text_twostep_lora_sage3_8", dict(kind="text", mode="twostep", steps=8, attention="sage_v3", turbo=True)),
    ("12_text_direct_lora_sage2_8", dict(kind="text", mode="t2v", steps=8, attention="sage_v2_memeff", turbo=True)),
    ("13_text_direct_lora_sage3_8", dict(kind="text", mode="t2v", steps=8, attention="sage_v3", turbo=True)),
    # --- fl2va first+last ---
    ("14_fl2va_firstlast_8", dict(kind="firstlast", mode="i2v", steps=8, attention="pytorch", turbo=False)),
    ("15_fl2va_firstlast_20", dict(kind="firstlast", mode="i2v", steps=20, attention="pytorch", turbo=False)),
    # --- ref2va images ---
    ("16_ref2va_images_8", dict(kind="ref_images", mode="r2v", steps=8, attention="pytorch", turbo=False)),
    ("17_ref2va_images_20", dict(kind="ref_images", mode="r2v", steps=20, attention="pytorch", turbo=False)),
    # --- ref2va video ---
    ("18_ref2va_video_8", dict(kind="ref_video", mode="r2v", steps=8, attention="pytorch", turbo=False)),
    ("19_ref2va_video_20", dict(kind="ref_video", mode="r2v", steps=20, attention="pytorch", turbo=False)),
]


def make_params(label: str, ov: dict) -> GenerateParams:
    kind = ov["kind"]
    if kind == "text":
        prompt, image, last, refs, vids = COMPLEX_TEXT, "", None, [], []
    elif kind == "firstlast":
        prompt, image, last, refs, vids = COMPLEX_FL2VA, FIRST_FRAME, LAST_FRAME, [], []
    elif kind == "ref_images":
        prompt, image, last, refs, vids = COMPLEX_REF_IMG, "", None, list(REF_IMAGES), []
    elif kind == "ref_video":
        prompt, image, last, refs, vids = COMPLEX_REF_VID, "", None, [], [REF_VIDEO]
    else:
        raise ValueError(kind)

    return GenerateParams(
        mode=ov["mode"],
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=ov["steps"],
        seed=SEED,
        image=image,
        last_frame=last,
        ref_images=refs,
        ref_videos=vids,
        output_prefix=f"{OUT_ROOT}/{label}",
        sage_attention="disabled",
        attention_backend=ov["attention"],
        turbo=ov["turbo"],
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
        sampler="res_multistep",
        scheduler="simple",
    )


def run_one(base_url: str, label: str, ov: dict) -> dict:
    params = make_params(label, ov)
    params.validate()
    t0 = time.time()
    pid = submit_prompt(base_url, build_graph(params))
    entry = wait_prompt(base_url, pid, poll_interval=3.0, timeout=4 * 3600)
    elapsed = round(time.time() - t0, 2)
    st = entry.get("status") or {}
    outs = extract_outputs(entry)
    if st.get("status_str") == "error":
        raise ComfyUIError(f"{label}: {(st.get('messages') or [])[:3]}")
    return {
        "label": label,
        "kind": ov["kind"],
        "mode": params.mode,
        "steps": params.steps,
        "attention": ov["attention"],
        "turbo": params.turbo,
        "width": WIDTH,
        "height": HEIGHT,
        "elapsed_sec": elapsed,
        "prompt_id": pid,
        "outputs": outs,
        "output_prefix": params.output_prefix,
        "status": st.get("status_str"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_official_1344x768.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    cases = CASES
    if args.only:
        want = set(args.only)
        cases = [(l, o) for l, o in CASES if l in want]
        if not cases:
            raise SystemExit(f"no cases match --only {args.only}")

    print(
        f"Official canvas {WIDTH}x{HEIGHT}, length={LENGTH}, FORCE off, {len(cases)} cases → {OUT_ROOT}/",
        flush=True,
    )
    results = []
    for label, ov in cases:
        print(f"\n=== {label} ===", flush=True)
        try:
            r = run_one(base_url, label, ov)
            print(f"{label}: {r['elapsed_sec']}s", flush=True)
            results.append(r)
            # incremental save
            payload = {
                "config": {
                    "width": WIDTH,
                    "height": HEIGHT,
                    "length": LENGTH,
                    "seed": SEED,
                    "resolution_note": "official MiniMaxH3 default 1344x768 (768 short edge / ~1.0 MP landscape)",
                    "force_cuda": False,
                    "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
                    "timing": "wall-clock submit→history done (incl. TE/VAE/IO)",
                    "first_frame": FIRST_FRAME,
                    "last_frame": LAST_FRAME,
                    "ref_images": REF_IMAGES,
                    "ref_video": REF_VIDEO,
                },
                "results": results,
            }
            args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"{label}: FAILED {e}", flush=True)
            results.append({"label": label, "status": "error", "error": str(e)})
            args.out.write_text(
                json.dumps({"config": {"width": WIDTH, "height": HEIGHT}, "results": results}, indent=2, ensure_ascii=False)
                + "\n"
            )
            # continue remaining cases
            continue

    print(f"\nWrote {args.out}")
    ok = [r for r in results if r.get("status") == "success"]
    print(f"done: {len(ok)}/{len(results)} success")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
