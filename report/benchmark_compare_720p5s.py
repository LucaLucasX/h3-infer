#!/usr/bin/env python3
"""Clean A/B suite: official t2v prompt, 720P, 5s, unified video outputs.

Cases:
  1. direct_8 / direct_20          — original single-pass (no Sage, no LoRA)
  2. twostep_8 / twostep_20        — two-stage only
  3. twostep_sage_8 / _20          — two-stage + Sage MemEff
  4. twostep_lora_8                — two-stage + Turbo LoRA v4
  5. twostep_lora_sage_8           — two-stage + Turbo LoRA + Sage
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

from comfy_client import ComfyUIError, health, submit_prompt, wait_prompt, extract_outputs
from h3_graph import GenerateParams, TURBO_LORA_V4, build_graph

# From ComfyUI-master_cp/workflows/video_minimax_h3_t2v.json
OFFICIAL_T2V_PROMPT = (
    'Vaporwave title sequence look: pink and blue gradient palette, VHS tracking artifacts, '
    'Greek statue motifs, chrome palm trees, RGB chromatic aberration, lo-fi retro atmosphere, '
    'mood languid and nostalgic.\n\n'
    'Timeline:\n'
    '[0s-1s] VHS static opens the frame, the title "COMFYUI" appears with RGB split and a slight horizontal jitter.\n'
    '[1s-2.5s] Hard cut, a Greek plaster bust close-up, pink-purple gradient sky, a pixelated sun.\n'
    '[2.5s-4s] Clean "STARRING" credits appear, "LATENT" and "CONTROLNET" each shown exactly once.\n'
    '[4s-5s] Final card "DIRECTED BY COMFYUI" holds, one VHS tracking glitch settling into stability.\n\n'
    'Hard cuts only, transitions landing with tape jumps, no push-ins, no dissolves.\n\n'
    'Audio: lo-fi vaporwave score, slow drum machine with soft bass, VHS tape-noise sample joins at 2.5s, '
    'melody fading for the last 1s.\n\n'
    'All text must be clearly legible, do not misspell English, no Chinese characters, do not repeat names '
    'or job titles, no soft dissolves, no subtitle bars.'
)

# 720P-ish 16:9, multiples of 32 (1280x720 is not 32-aligned on H)
WIDTH, HEIGHT = 1280, 704
LENGTH = 124  # ~5.17s @ 24fps
SEED = 42
OUT_ROOT = "h3_compare_720p5s"

CASES: list[tuple[str, dict]] = [
    ("01_direct_8", dict(mode="t2v", steps=8, attention_backend="pytorch", turbo=False)),
    ("02_direct_20", dict(mode="t2v", steps=20, attention_backend="pytorch", turbo=False)),
    ("03_twostep_8", dict(mode="twostep", steps=8, attention_backend="pytorch", turbo=False)),
    ("04_twostep_20", dict(mode="twostep", steps=20, attention_backend="pytorch", turbo=False)),
    ("05_twostep_sage_8", dict(mode="twostep", steps=8, attention_backend="sage_v2_memeff", turbo=False)),
    ("06_twostep_sage_20", dict(mode="twostep", steps=20, attention_backend="sage_v2_memeff", turbo=False)),
    ("07_twostep_lora_8", dict(mode="twostep", steps=8, attention_backend="pytorch", turbo=True)),
    ("08_twostep_lora_sage_8", dict(mode="twostep", steps=8, attention_backend="sage_v2_memeff", turbo=True)),
    # SageAttention3 (PathchSageAttentionKJ + sageattn3)
    ("09_twostep_sage3_8", dict(mode="twostep", steps=8, attention_backend="sage_v3", turbo=False)),
    ("10_twostep_sage3_20", dict(mode="twostep", steps=20, attention_backend="sage_v3", turbo=False)),
    ("11_twostep_lora_sage3_8", dict(mode="twostep", steps=8, attention_backend="sage_v3", turbo=True)),
]


def make_params(label: str, overrides: dict) -> GenerateParams:
    return GenerateParams(
        mode=overrides["mode"],
        prompt=OFFICIAL_T2V_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=overrides["steps"],
        seed=SEED,
        image="",  # text-only for fair t2v comparison
        output_prefix=f"{OUT_ROOT}/{label}",
        sage_attention="disabled",
        attention_backend=overrides["attention_backend"],
        turbo=overrides["turbo"],
        turbo_lora=TURBO_LORA_V4,
        turbo_strength=1.0,
        sampler="res_multistep",
        scheduler="simple",
    )


def run_one(base_url: str, label: str, overrides: dict) -> dict:
    params = make_params(label, overrides)
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
        "mode": params.mode,
        "steps": params.steps,
        "sage": overrides["attention_backend"] != "pytorch",
        "turbo": params.turbo,
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
    ap.add_argument("--only", nargs="*", default=None, help="Optional subset of labels")
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_compare_720p5s.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    cases = CASES
    if args.only:
        want = set(args.only)
        cases = [c for c in CASES if c[0] in want]
        if not cases:
            raise SystemExit(f"no matching cases for {args.only}")

    print(f"Compare suite: {WIDTH}x{HEIGHT} (~720P) length={LENGTH} (~5s) seed={SEED}")
    print(f"Prompt: official video_minimax_h3_t2v.json")
    print(f"Outputs under: ComfyUI-master_cp/output/{OUT_ROOT}/")
    print(f"Cases: {[c[0] for c in cases]}")

    results: list[dict] = []
    for label, ov in cases:
        print(f"\n=== {label}  mode={ov['mode']} steps={ov['steps']} "
              f"sage={ov['attention_backend']} turbo={ov['turbo']} ===")
        try:
            row = run_one(base_url, label, ov)
            results.append(row)
            files = [o.get("filename") for o in row.get("outputs") or []]
            print(f"  {row['elapsed_sec']}s  files={files}")
        except Exception as e:
            results.append({"label": label, "error": str(e), **ov})
            print(f"  FAILED: {e}")

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "prompt_source": "workflows/video_minimax_h3_t2v.json",
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "lora": TURBO_LORA_V4,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}")

    print("\nSummary (wall-clock):")
    for r in results:
        if "elapsed_sec" in r:
            print(f"  {r['label']:28s} {r['elapsed_sec']:8.1f}s")
        else:
            print(f"  {r['label']:28s} FAILED")
    return 0 if all("elapsed_sec" in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
