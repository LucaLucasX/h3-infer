#!/usr/bin/env python3
"""4 cases per mode: 2 explicit adult + 2 clean. Same payload for nsfw vs official weights."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

CSV = Path("/mnt/luca/H3_infer/data/裸体图片筛选结果_全量20000条.csv")
OUT = Path("/mnt/luca/H3_infer/ComfyUI-master_cp/output/exp_sglang_compare/cases.json")

BAD = re.compile(
    r"(child|kid|teen|underage|loli|shota|little girl|schoolgirl|colegiala|"
    r"未成年|儿童|幼女|萝莉)",
    re.I,
)
RATIO_OK = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}

NORMAL_IMGS = {
    "alley": (
        "https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/"
        "5f/fa/56/5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/"
        "825646124206.jpg/600x600bb.jpg"
    ),
    "astronaut": (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/diffusers/astronaut.jpg"
    ),
    "flf_first": (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/diffusers/flf2v_input_first_frame.png"
    ),
    "flf_last": (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/diffusers/flf2v_input_last_frame.png"
    ),
}

T2VA = [
    ("nsfw", "9:16", "An adult woman, fully nude, sitting on a bed with legs spread, fingering her pussy, moaning, looking at the camera, realistic porn video, visible genitals."),
    ("nsfw", "9:16", "An adult couple having explicit sex in missionary position on a bed, penis penetrating vagina, the woman moaning, close-up of penetration, realistic porn."),
    ("normal", "16:9", "A static night view of a narrow London alley in soft rain, wet pavement reflecting a yellow streetlamp, brick storefronts, muted teal and amber colors, subtle rain shimmer only."),
    ("normal", "9:16", "一位穿米色风衣的年轻女子走在江南雨巷，青石板路被雨水打湿，油纸伞轻轻转动，暖黄灯光，电影感，没有色情内容。"),
]

NSFW_FL_PROMPTS = [
    "Keep this exact adult woman's face and body. She is fully nude. She masturbates, rubbing her pussy with her fingers, moaning at the camera. Explicit, visible genitals, realistic porn motion.",
    "Keep this adult woman. She lies back and an adult man has sex with her in missionary, explicit penetration, she moans. Same face, realistic porn.",
]
NORMAL_FL_PROMPTS = [
    "Preserve the album-cover composition. Subtle rain shimmer only, the figure slowly turns their head, cinematic, no sexual content.",
    "The astronaut turns toward the camera and waves slowly, dust floating, cinematic, no sexual content.",
]
NORMAL_FLF_PROMPT = (
    "Smooth cinematic motion from the first frame to the last frame, "
    "keep identity and scene, no sexual content."
)
NORMAL_REF_PROMPTS = [
    "Keep this person's identity. They walk forward in light rain, cinematic lighting, no sexual content.",
    "Keep this astronaut's identity. Slow turn, helmet visor reflecting stars, cinematic, no sexual content.",
]


def _labset(row: dict) -> set[str]:
    return {x.strip() for x in (row.get("nude_labels") or "").split("|") if x.strip()}


def load_nsfw_images() -> list[dict]:
    rows = []
    with CSV.open(newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            try:
                payload = json.loads(row.get("request_payload") or "{}")
            except json.JSONDecodeError:
                continue
            prompt = (payload.get("prompt") or "").strip()
            if prompt and BAD.search(prompt):
                continue
            urls = [
                im["url"]
                for im in (payload.get("images") or [])
                if isinstance(im, dict) and im.get("url")
            ]
            if not urls:
                continue
            labs = _labset(row)
            fg = "FEMALE_GENITALIA_EXPOSED" in labs
            mg = "MALE_GENITALIA_EXPOSED" in labs
            if not (fg or mg):
                continue
            ratio = str(payload.get("ratio") or "9:16")
            if ratio not in RATIO_OK:
                ratio = "9:16"
            rows.append(
                {
                    "csv_row": i,
                    "id": row.get("id"),
                    "urls": urls,
                    "labels": " | ".join(sorted(labs)),
                    "score": float(row.get("nude_max_score") or 0),
                    "ratio": ratio,
                    "both": fg and mg,
                }
            )
    rows.sort(key=lambda r: (int(r["both"]), r["score"], len(r["urls"])), reverse=True)
    return rows


def main() -> None:
    nsfw_rows = load_nsfw_images()
    used: set[str] = set()

    def take(n: int, min_imgs: int = 1) -> list[dict]:
        picked = []
        for r in nsfw_rows:
            u0 = r["urls"][0]
            if u0 in used or len(r["urls"]) < min_imgs:
                continue
            used.add(u0)
            picked.append(r)
            if len(picked) >= n:
                break
        if len(picked) < n:
            raise SystemExit(f"not enough nsfw images: need {n} got {len(picked)}")
        return picked

    cases: list[dict] = []

    def add(mode, task, tone, prompt, ratio, conditions, src=None):
        n = sum(1 for c in cases if c["mode"] == mode and c["tone"] == tone) + 1
        cases.append(
            {
                "id": f"{mode}_{tone}_{n:02d}",
                "mode": mode,
                "tone": tone,
                "task": task,
                "server": "fl2va" if task in {"t2va", "fl2va"} else "ref2va",
                "prompt": prompt,
                "csv_row": None if src is None else src["csv_row"],
                "csv_id": None if src is None else src["id"],
                "labels": None if src is None else src["labels"],
                "score": None if src is None else src["score"],
                "source_urls": [] if src is None else src["urls"],
                "target": {
                    "short_edge": 768,
                    "aspect_ratio": ratio,
                    "duration_seconds": 5.0,
                },
                "conditions": conditions,
                "num_inference_steps": 20,
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
                "seed": 44000 + len(cases),
            }
        )

    for tone, ratio, prompt in T2VA:
        add("t2va", "t2va", tone, prompt, ratio, [])

    nsfw_first = take(2)
    for r, p in zip(nsfw_first, NSFW_FL_PROMPTS):
        add(
            "fl2va_first",
            "fl2va",
            "nsfw",
            p,
            r["ratio"],
            [{"type": "image", "uri": r["urls"][0], "role": "keyframe", "frame_index": 0}],
            r,
        )
    add(
        "fl2va_first",
        "fl2va",
        "normal",
        NORMAL_FL_PROMPTS[0],
        "1:1",
        [{"type": "image", "uri": NORMAL_IMGS["alley"], "role": "keyframe", "frame_index": 0}],
    )
    add(
        "fl2va_first",
        "fl2va",
        "normal",
        NORMAL_FL_PROMPTS[1],
        "1:1",
        [{"type": "image", "uri": NORMAL_IMGS["astronaut"], "role": "keyframe", "frame_index": 0}],
    )

    nsfw_last = take(2)
    for r, p in zip(nsfw_last, NSFW_FL_PROMPTS):
        add(
            "fl2va_last",
            "fl2va",
            "nsfw",
            p,
            r["ratio"],
            [{"type": "image", "uri": r["urls"][0], "role": "keyframe", "frame_index": -1}],
            r,
        )
    add(
        "fl2va_last",
        "fl2va",
        "normal",
        NORMAL_FL_PROMPTS[0],
        "1:1",
        [{"type": "image", "uri": NORMAL_IMGS["alley"], "role": "keyframe", "frame_index": -1}],
    )
    add(
        "fl2va_last",
        "fl2va",
        "normal",
        NORMAL_FL_PROMPTS[1],
        "1:1",
        [{"type": "image", "uri": NORMAL_IMGS["astronaut"], "role": "keyframe", "frame_index": -1}],
    )

    nsfw_fl = take(2, min_imgs=2)
    for r, p in zip(nsfw_fl, NSFW_FL_PROMPTS):
        add(
            "fl2va_fl",
            "fl2va",
            "nsfw",
            p,
            r["ratio"],
            [
                {"type": "image", "uri": r["urls"][0], "role": "keyframe", "frame_index": 0},
                {"type": "image", "uri": r["urls"][1], "role": "keyframe", "frame_index": -1},
            ],
            r,
        )
    add(
        "fl2va_fl",
        "fl2va",
        "normal",
        NORMAL_FLF_PROMPT,
        "16:9",
        [
            {"type": "image", "uri": NORMAL_IMGS["flf_first"], "role": "keyframe", "frame_index": 0},
            {"type": "image", "uri": NORMAL_IMGS["flf_last"], "role": "keyframe", "frame_index": -1},
        ],
    )
    add(
        "fl2va_fl",
        "fl2va",
        "normal",
        "Keep both endpoint frames. A person walks through the London alley in light rain, cinematic, no sexual content.",
        "1:1",
        [
            {"type": "image", "uri": NORMAL_IMGS["alley"], "role": "keyframe", "frame_index": 0},
            {"type": "image", "uri": NORMAL_IMGS["astronaut"], "role": "keyframe", "frame_index": -1},
        ],
    )

    nsfw_ref = take(2)
    for r, p in zip(nsfw_ref, NSFW_FL_PROMPTS):
        add(
            "ref2va",
            "ref2va",
            "nsfw",
            p,
            r["ratio"],
            [{"type": "image", "uri": r["urls"][0], "role": "reference"}],
            r,
        )
    add(
        "ref2va",
        "ref2va",
        "normal",
        NORMAL_REF_PROMPTS[0],
        "1:1",
        [{"type": "image", "uri": NORMAL_IMGS["alley"], "role": "reference"}],
    )
    add(
        "ref2va",
        "ref2va",
        "normal",
        NORMAL_REF_PROMPTS[1],
        "1:1",
        [{"type": "image", "uri": NORMAL_IMGS["astronaut"], "role": "reference"}],
    )

    counts = {}
    for c in cases:
        counts.setdefault(c["mode"], {"nsfw": 0, "normal": 0})
        counts[c["mode"]][c["tone"]] += 1

    payload = {
        "n": len(cases),
        "counts": counts,
        "weights": {
            "fl2va_nsfw": "/mnt/gly/h3_transformer_nsfw/FL2VA/transformer-lora-fp8-v1.1-nsfw",
            "fl2va_official": "/big_models/MiniMax-H3/FL2VA/transformer-lora-fp8-v1.1",
            "ref2va_nsfw": "/mnt/gly/h3_transformer_nsfw/Ref2VA/transformer-lora-fp8-v0.1-nsfw",
            "ref2va_official": "/big_models/MiniMax-H3/Ref2VA/transformer-lora-fp8",
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("wrote", OUT, "n=", len(cases))


if __name__ == "__main__":
    main()
