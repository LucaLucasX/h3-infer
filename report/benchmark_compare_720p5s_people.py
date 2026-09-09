#!/usr/bin/env python3
"""Complex-prompt A/B suite (people / multi-shot). Separate output dir from vaporwave suite.

Same matrix as benchmark_compare_720p5s.py:
  direct 8/20, twostep 8/20, twostep+sage2 8/20, twostep+lora 8,
  twostep+lora+sage2 8, twostep+sage3 8/20, twostep+lora+sage3 8
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

# Complex multi-character / multi-shot prompt (t2v)
COMPLEX_PEOPLE_PROMPT = """Cinematic short film, photorealistic, 24fps, shallow depth of field.

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

WIDTH, HEIGHT = 1280, 704
LENGTH = 124
SEED = 42
OUT_ROOT = "h3_compare_720p5s_people"

CASES: list[tuple[str, dict]] = [
    ("01_direct_8", dict(mode="t2v", steps=8, attention_backend="pytorch", turbo=False)),
    ("02_direct_20", dict(mode="t2v", steps=20, attention_backend="pytorch", turbo=False)),
    ("03_twostep_8", dict(mode="twostep", steps=8, attention_backend="pytorch", turbo=False)),
    ("04_twostep_20", dict(mode="twostep", steps=20, attention_backend="pytorch", turbo=False)),
    ("05_twostep_sage2_8", dict(mode="twostep", steps=8, attention_backend="sage_v2_memeff", turbo=False)),
    ("06_twostep_sage2_20", dict(mode="twostep", steps=20, attention_backend="sage_v2_memeff", turbo=False)),
    ("07_twostep_lora_8", dict(mode="twostep", steps=8, attention_backend="pytorch", turbo=True)),
    ("08_twostep_lora_sage2_8", dict(mode="twostep", steps=8, attention_backend="sage_v2_memeff", turbo=True)),
    ("09_twostep_sage3_8", dict(mode="twostep", steps=8, attention_backend="sage_v3", turbo=False)),
    ("10_twostep_sage3_20", dict(mode="twostep", steps=20, attention_backend="sage_v3", turbo=False)),
    ("11_twostep_lora_sage3_8", dict(mode="twostep", steps=8, attention_backend="sage_v3", turbo=True)),
    # Direct + Sage + Turbo LoRA
    ("12_direct_lora_sage2_8", dict(mode="t2v", steps=8, attention_backend="sage_v2_memeff", turbo=True)),
    ("13_direct_lora_sage3_8", dict(mode="t2v", steps=8, attention_backend="sage_v3", turbo=True)),
]


def make_params(label: str, overrides: dict) -> GenerateParams:
    return GenerateParams(
        mode=overrides["mode"],
        prompt=COMPLEX_PEOPLE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
        steps=overrides["steps"],
        seed=SEED,
        image="",
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
        "attention": overrides["attention_backend"],
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
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=REPORT / "benchmark_compare_720p5s_people.json")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    health(base_url)

    cases = CASES
    if args.only:
        want = set(args.only)
        cases = [c for c in CASES if c[0] in want]

    print(f"People suite: {WIDTH}x{HEIGHT} length={LENGTH} seed={SEED}")
    print(f"Outputs: ComfyUI-master_cp/output/{OUT_ROOT}/")
    print(f"Cases: {[c[0] for c in cases]}")

    results: list[dict] = []
    for label, ov in cases:
        print(
            f"\n=== {label}  mode={ov['mode']} steps={ov['steps']} "
            f"attn={ov['attention_backend']} turbo={ov['turbo']} ===",
            flush=True,
        )
        try:
            row = run_one(base_url, label, ov)
            results.append(row)
            files = [o.get("filename") for o in row.get("outputs") or []]
            print(f"  {row['elapsed_sec']}s  files={files}", flush=True)
        except Exception as e:
            results.append({"label": label, "error": str(e), **ov})
            print(f"  FAILED: {e}", flush=True)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "seed": SEED,
            "prompt_theme": "complex multi-character rainy Tokyo night",
            "output_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "lora": TURBO_LORA_V4,
            "timing": "wall-clock submit→history done (incl. TE/VAE/IO)",
        },
        "prompt": COMPLEX_PEOPLE_PROMPT,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}", flush=True)
    print("\nSummary (wall-clock):", flush=True)
    for r in results:
        if "elapsed_sec" in r:
            print(f"  {r['label']:30s} {r['elapsed_sec']:8.1f}s", flush=True)
        else:
            print(f"  {r['label']:30s} FAILED", flush=True)
    return 0 if all("elapsed_sec" in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
