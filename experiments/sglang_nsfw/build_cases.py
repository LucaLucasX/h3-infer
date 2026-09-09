#!/usr/bin/env python3
"""Sample explicit adult cases. CSV image URLs + pornographic video prompts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

CSV = Path("/mnt/luca/H3_infer/data/裸体图片筛选结果_全量20000条.csv")
OUT = Path("/mnt/luca/H3_infer/ComfyUI-master_cp/output/exp_sglang_nsfw/cases.json")

BAD = re.compile(
    r"(child|kid|teen|underage|loli|shota|little girl|schoolgirl|colegiala|"
    r"未成年|儿童|幼女|萝莉|jailbait)",
    re.I,
)
RATIO_OK = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}

# Explicit adult video prompts. CSV text is mostly i2i edits; these are the
# actual NSFW tests, paired with genitalia-detected reference/keyframe images.
T2VA_PROMPTS = [
    ("9:16", "An adult woman, fully nude, sitting on a bed with legs spread, fingering her pussy, moaning, looking at the camera, realistic porn video, visible genitals."),
    ("9:16", "An adult couple having explicit sex in missionary position on a bed, penis penetrating vagina, the woman moaning and holding the man, realistic porn, close-up of penetration."),
    ("9:16", "An adult woman giving a blowjob, mouth on penis, saliva, looking up at the camera, explicit oral sex, realistic porn close-up."),
    ("16:9", "An adult woman on all fours, doggystyle sex, a man thrusting into her from behind, breasts swinging, explicit penetration, bedroom, realistic porn."),
    ("9:16", "An adult woman riding a man cowgirl, bouncing on his cock, pussy stretched around the penis, moaning, bouncing breasts, realistic porn."),
    ("9:16", "Two adult women, fully nude, one licking the other's pussy, lesbian oral sex, wet, moaning, realistic porn, bed."),
    ("1:1", "An adult nude woman standing, spreading her labia with both hands, showing her pussy to the camera, slow turn, explicit, realistic."),
    ("9:16", "An adult couple having standing sex against a wall, man lifting the woman, explicit penetration, moaning, realistic porn."),
    ("9:16", "An adult nude woman masturbating with a dildo, inserting it into her vagina, moaning, legs spread on a sofa, realistic porn."),
    ("16:9", "An adult couple in 69 position, mutual oral sex, penis in mouth and tongue on pussy, explicit, realistic porn, bed."),
]

FL_PROMPTS = [
    "Keep this exact adult woman's face and body. She is fully nude. She starts masturbating, rubbing her pussy with her fingers, moaning, looking at the camera. Explicit, visible genitals, realistic porn motion.",
    "Keep this adult woman. She lies back, spreads her legs, and an adult man has sex with her in missionary, explicit penetration, she moans. Realistic porn, same face.",
    "Keep this adult woman. She gets on her knees and gives a blowjob, mouth on a penis, saliva, looking at camera. Explicit oral sex, same face and body.",
    "Keep this adult woman. She turns around, gets on all fours, and is fucked doggystyle, breasts swinging, explicit penetration. Same face, realistic porn.",
    "Keep this adult woman. She sits on a man and rides him cowgirl, bouncing, pussy around his cock, moaning. Explicit, same face and body.",
    "Keep this adult woman fully nude. She oils her breasts and pussy, then fingers herself, slow explicit masturbation to camera.",
    "Keep this adult woman. She and another adult woman kiss and then perform lesbian oral sex, licking pussy, moaning. Explicit, same faces.",
    "Keep this adult woman. She is taken from behind standing, explicit sex, one hand on her breast, moaning at the camera. Same face.",
    "Keep this adult woman. She lies on her back, legs up, explicit close-up of her pussy being licked, she moans. Same face and body.",
    "Keep this adult woman. She gives a handjob then a blowjob, cum on her breasts at the end. Explicit, same face, realistic porn.",
]


def _labels(row: dict) -> set[str]:
    return {x.strip() for x in (row.get("nude_labels") or "").split("|") if x.strip()}


def main() -> None:
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
            labs = _labels(row)
            try:
                score = float(row.get("nude_max_score") or 0)
            except ValueError:
                score = 0.0
            ratio = str(payload.get("ratio") or "9:16")
            if ratio not in RATIO_OK:
                ratio = "9:16"
            rows.append(
                {
                    "csv_row": i,
                    "id": row.get("id"),
                    "prompt": prompt,
                    "urls": urls,
                    "labels": " | ".join(sorted(labs)),
                    "labset": labs,
                    "score": score,
                    "ratio": ratio,
                    "fg": "FEMALE_GENITALIA_EXPOSED" in labs,
                    "mg": "MALE_GENITALIA_EXPOSED" in labs,
                    "breast": "FEMALE_BREAST_EXPOSED" in labs,
                }
            )

    # Prefer the most explicit detections: genitals, then breasts.
    def rank(r: dict) -> tuple:
        return (
            int(r["fg"]) + int(r["mg"]),
            int(r["breast"]),
            r["score"],
            len(r["urls"]),
        )

    rows.sort(key=rank, reverse=True)
    used: set[str] = set()

    def take(n: int, *, min_imgs: int = 1, need_fg: bool = True) -> list[dict]:
        picked = []
        for r in rows:
            u0 = r["urls"][0]
            if u0 in used:
                continue
            if len(r["urls"]) < min_imgs:
                continue
            if need_fg and not (r["fg"] or r["mg"]):
                continue
            used.add(u0)
            picked.append(r)
            if len(picked) >= n:
                break
        if len(picked) < n:
            for r in rows:
                u0 = r["urls"][0]
                if u0 in used or len(r["urls"]) < min_imgs:
                    continue
                used.add(u0)
                picked.append(r)
                if len(picked) >= n:
                    break
        return picked

    cases: list[dict] = []

    def add(mode: str, task: str, r: dict | None, conditions: list, prompt: str, ratio: str) -> None:
        idx = sum(1 for c in cases if c["mode"] == mode) + 1
        cases.append(
            {
                "id": f"{mode}_{idx:02d}",
                "mode": mode,
                "task": task,
                "server": "fl2va" if task in {"t2va", "fl2va"} else "ref2va",
                "prompt": prompt,
                "csv_row": None if r is None else r["csv_row"],
                "csv_id": None if r is None else r["id"],
                "labels": None if r is None else r["labels"],
                "score": None if r is None else r["score"],
                "source_urls": [] if r is None else r["urls"],
                "target": {
                    "short_edge": 768,
                    "aspect_ratio": ratio,
                    "duration_seconds": 5.0,
                },
                "conditions": conditions,
                "num_inference_steps": 20,
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
                "seed": 43000 + len(cases),
            }
        )

    for ratio, prompt in T2VA_PROMPTS:
        add("t2va", "t2va", None, [], prompt, ratio)

    for r, p in zip(take(10, need_fg=True), FL_PROMPTS):
        add(
            "fl2va_first",
            "fl2va",
            r,
            [{"type": "image", "uri": r["urls"][0], "role": "keyframe", "frame_index": 0}],
            p,
            r["ratio"],
        )

    for r, p in zip(take(10, need_fg=True), FL_PROMPTS):
        add(
            "fl2va_last",
            "fl2va",
            r,
            [{"type": "image", "uri": r["urls"][0], "role": "keyframe", "frame_index": -1}],
            p,
            r["ratio"],
        )

    for r, p in zip(take(10, min_imgs=2, need_fg=True), FL_PROMPTS):
        add(
            "fl2va_fl",
            "fl2va",
            r,
            [
                {"type": "image", "uri": r["urls"][0], "role": "keyframe", "frame_index": 0},
                {"type": "image", "uri": r["urls"][1], "role": "keyframe", "frame_index": -1},
            ],
            p,
            r["ratio"],
        )

    for r, p in zip(take(10, need_fg=True), FL_PROMPTS):
        add(
            "ref2va",
            "ref2va",
            r,
            [{"type": "image", "uri": r["urls"][0], "role": "reference"}],
            p,
            r["ratio"],
        )

    payload = {
        "csv": str(CSV),
        "n": len(cases),
        "note": "CSV images are genitalia-detected nudes; prompts are explicit adult sex actions for NSFW weight test.",
        "counts": {
            m: sum(1 for c in cases if c["mode"] == m)
            for m in ("t2va", "fl2va_first", "fl2va_last", "fl2va_fl", "ref2va")
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["counts"], ensure_ascii=False))
    print("wrote", OUT, "n=", len(cases))
    # sanity: fl/ref all have fg or mg
    for m in ("fl2va_first", "fl2va_last", "fl2va_fl", "ref2va"):
        bad = [c for c in cases if c["mode"] == m and "GENITALIA_EXPOSED" not in (c["labels"] or "")]
        print(m, "missing genitalia labels", len(bad))


if __name__ == "__main__":
    main()
