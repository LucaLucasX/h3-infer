#!/usr/bin/env python3
"""Dump Sparge+Sage2+4-step T2V 5s graphs. Does not modify h3_graph.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/luca/H3_infer")
sys.path[:0] = [str(ROOT), str(ROOT / "dev" / "sage_lora_sparge")]

from h3_graph import GenerateParams, build_graph  # noqa: E402

OUT = Path(__file__).resolve().parent / "workflows"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "A cinematic wide shot of a woman walking through neon-lit rain at night, "
    "photorealistic, natural motion, no text."
)


def _params() -> GenerateParams:
    return GenerateParams(
        mode="t2v",
        prompt=PROMPT,
        width=1280,
        height=704,
        length=124,
        steps=4,
        seed=42,
        output_prefix="exp_h3_vae_trt/t2v5s",
        sage_attention="sage_v2_memeff",
        sparge=True,
        sparge_topk=0.5,
        turbo=True,
        fps=24.0,
    )


def main() -> None:
    p = _params()
    baseline = build_graph(p)
    (OUT / "t2v_5s_sage_sparge_4step_baseline_api.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )

    p.output_prefix = "exp_h3_vae_trt/t2v5s_trt"
    trt = build_graph(p)
    trt["3"] = {
        "class_type": "MiniMaxH3TRTVAELoader",
        "inputs": {
            "decoder": "minimax_h3_vae_decoder.engine",
            "encoder": "None",
        },
    }
    (OUT / "t2v_5s_sage_sparge_4step_trt_api.json").write_text(
        json.dumps(trt, indent=2), encoding="utf-8"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
