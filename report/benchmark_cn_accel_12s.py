#!/usr/bin/env python3
"""Chinese-scene 12s A/B: Sage2 baseline + one accel at a time (EXP :8190).

Prompt: Shanghai rainy-night dialogue (same complexity as prior Tokyo voice scene,
but Chinese street/signage — avoids Japanese neon text bleed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report"))

from benchmark_official_1344x768 import HEIGHT, SEED, WIDTH
from comfy_client import ComfyUIError, extract_outputs, health, submit_prompt, wait_prompt
from h3_graph import GenerateParams, build_graph, node

LENGTH = 288
FPS = 24.0
STEPS = 8
OUT_ROOT = "exp_cn_accel_12s"

# Same structure/complexity as COMPLEX_VOICE_TEXT, Chinese Shanghai setting.
CN_VOICE_TEXT = """Cinematic photorealistic rainy-night Shanghai Bund drama, 24fps, shallow depth of field, film grain.

Cast:
- Woman, 28, East Asian, short black bob, red trench coat, black umbrella
- Man, 35, navy wool coat, glasses, paper coffee cup
- Crowd extras; Chinese shop signs and LED billboards in Simplified Chinese only (no Japanese kana/kanji neon)

Timeline with spoken Mandarin dialogue (stereo, clear lip-sync, no voiceover narration):
[0.0s-2.5s] Close-up on the woman under warm shop lights; soft jazz piano; rain on umbrella.
  Woman (calm, slightly breathless): "对不起，我迟到了。"
[2.5s-5.0s] Medium shot at a zebra crossing near the Bund; she looks up; man notices from opposite curb; footsteps in puddles.
  Man (warm, surprised): "没关系，我也刚到。"
[5.0s-8.0s] Tracking shot as they walk toward each other under yellow streetlights; one bicycle bell at 6.2s.
  Woman (soft smile): "这雨好像不会停。"
  Man (gentle): "那正好，咖啡还热着。"
[8.0s-12.0s] Over-shoulder meet under a convenience-store awning with Chinese signage; he offers coffee; rain continues; jazz swells then softens.

Camera: naturalistic handheld, no jump cuts, no text overlays, no logos, no Japanese characters anywhere in frame.
Audio: stereo rain bed, soft jazz trio, footsteps, one bicycle bell, **clear Mandarin dialogue as above**, no narrator.
Lighting: night rain, cyan/magenta Chinese LED + warm shop tungsten.
""".strip()

COMMON = dict(
    mode="t2v",
    prompt=CN_VOICE_TEXT,
    width=WIDTH,
    height=HEIGHT,
    length=LENGTH,
    fps=FPS,
    steps=STEPS,
    seed=SEED,
    turbo=True,
    attention_backend="sage_v2_memeff",
    ref_images=[],
    ref_videos=[],
    image="",
)


def _base(label: str, **overrides) -> GenerateParams:
    kw = dict(COMMON)
    kw["output_prefix"] = f"{OUT_ROOT}/{label}"
    kw.update(overrides)
    return GenerateParams(**kw)


def _graph_plain(label: str, **overrides):
    return build_graph(_base(label, **overrides))


def _graph_sparge(label: str, topk: float, dense_first_steps: int = 0):
    # Replace Sage MemEff node "5" with Sparge (model from UNET).
    g = build_graph(_base(label))
    g["5"] = node(
        "MiniMaxH3SpargeAttnPatchExp",
        {
            "model": ["1", 0],
            "topk": float(topk),
            "dense_first_steps": int(dense_first_steps),
            "num_layers": 50,
        },
    )
    return g


def _graph_losa(label: str, mass: float):
    g = build_graph(_base(label))
    g["5"] = node(
        "MiniMaxH3LoSAApproxPatchExp",
        {
            "model": ["1", 0],
            "mass_thresh": float(mass),
            "profile_steps": 1,
            "num_layers": 50,
        },
    )
    return g


def _run(label: str, graph: dict, base_url: str) -> dict:
    t0 = time.perf_counter()
    try:
        pid = submit_prompt(base_url, graph)
        print(f"[{label}] prompt_id={pid}", flush=True)
        hist = wait_prompt(base_url, pid)
        elapsed = round(time.perf_counter() - t0, 2)
        st = (hist.get("status") or {}).get("status_str")
        return {
            "label": label,
            "status": "success" if st == "success" else f"comfy_{st}",
            "elapsed_sec": elapsed,
            "prompt_id": pid,
            "outputs": extract_outputs(hist),
        }
    except ComfyUIError as e:
        return {
            "label": label,
            "status": "error",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "error": str(e)[:4000],
        }


def _delete_prefix(prefix: str) -> None:
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
    if not out_dir.is_dir():
        return
    for p in out_dir.glob(f"{prefix}*"):
        p.unlink(missing_ok=True)
        print(f"deleted {p.name}", flush=True)


CASES = [
    ("01_baseline_sage2", lambda: _graph_plain("01_baseline_sage2")),
    ("02_teacache", lambda: _graph_plain("02_teacache", teacache=True)),
    ("03_spectrum", lambda: _graph_plain("03_spectrum", spectrum=True)),
    ("04_easycache", lambda: _graph_plain("04_easycache", easycache=True)),
    ("05_tespeed", lambda: _graph_plain("05_tespeed", tespeed=True)),
    ("06_losa_m0.9", lambda: _graph_losa("06_losa_m0.9", 0.9)),
    ("07_sparge_topk0.5", lambda: _graph_sparge("07_sparge_topk0.5", 0.5, 0)),
    ("08_sparge_dense2_topk0.5", lambda: _graph_sparge("08_sparge_dense2_topk0.5", 0.5, 2)),
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default="http://127.0.0.1:8190")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--only", nargs="*", help="Subset of labels")
    args = ap.parse_args()

    health(args.comfy)
    cases = CASES
    if args.only:
        allow = set(args.only)
        cases = [(l, fn) for l, fn in CASES if l in allow]

    print(
        f"CN accel 12s: {WIDTH}x{HEIGHT} length={LENGTH} turbo={STEPS} cases={len(cases)}",
        flush=True,
    )
    out_dir = ROOT / "ComfyUI-master_cp" / "output" / OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_warmup and not args.only:
        print("=== warmup (discard) ===", flush=True)
        w = _run("_warmup_discard", _graph_plain("_warmup_discard"), args.comfy)
        print(json.dumps({k: w[k] for k in ("label", "status", "elapsed_sec")}, ensure_ascii=False))
        _delete_prefix("_warmup_discard")

    results = []
    for label, make in cases:
        print(f"=== {label} ===", flush=True)
        r = _run(label, make(), args.comfy)
        results.append(r)
        print(
            json.dumps(
                {k: r.get(k) for k in ("label", "status", "elapsed_sec", "error")},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if r["status"] != "success":
            print(f"FAILED {label}: {r.get('error')}", flush=True)

    base_t = next(
        (r["elapsed_sec"] for r in results if r["label"].startswith("01_") and r.get("elapsed_sec")),
        None,
    )
    summary = []
    for r in results:
        row = {"label": r["label"], "elapsed_sec": r.get("elapsed_sec"), "status": r.get("status")}
        if base_t and r.get("elapsed_sec") and r.get("status") == "success":
            row["speedup_vs_baseline"] = round(base_t / r["elapsed_sec"], 3)
        summary.append(row)

    payload = {
        "config": {
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "steps": STEPS,
            "seed": SEED,
            "prompt": "Shanghai Bund rainy Mandarin dialogue (CN signs)",
            "comfy": args.comfy,
            "compare_dir": f"ComfyUI-master_cp/output/{OUT_ROOT}",
            "sparge_fix": "08 uses dense Sage for first 2 denoise steps then Sparge topk=0.5",
        },
        "summary": summary,
        "results": results,
    }
    out = ROOT / "experiments" / "sparse_attn" / "output" / "ab_cn_accel_12s.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
