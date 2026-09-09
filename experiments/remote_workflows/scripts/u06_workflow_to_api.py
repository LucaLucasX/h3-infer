#!/usr/bin/env python3
"""Convert ComfyUI UI workflow JSON to /prompt API format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _widget_names(class_type: str, object_info: dict) -> list[str]:
    info = object_info.get(class_type, {})
    names: list[str] = []
    for section in ("required", "optional"):
        block = info.get("input", {}).get(section, {})
        for name, spec in block.items():
            if isinstance(spec, list) and len(spec) >= 2 and spec[0] in (
                "INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMBO_STRING",
            ):
                names.append(name)
    return names


def workflow_to_api(workflow: dict, object_info: dict, *, include_muted: bool = True) -> dict[str, Any]:
    nodes_by_id = {n["id"]: n for n in workflow["nodes"]}
    link_by_id = {l[0]: l for l in workflow["links"]}

    skip_types = {"MarkdownNote", "Note", "Reroute"}

    prompt: dict[str, Any] = {}
    for node in workflow["nodes"]:
        if node["type"] in skip_types:
            continue
        mode = node.get("mode", 0)
        if mode == 4 and not include_muted:
            continue
        if mode == 4 and include_muted:
            pass  # include muted nodes in full-config runs

        nid = str(node["id"])
        class_type = node["type"]
        inputs: dict[str, Any] = {}

        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            link = link_by_id[link_id]
            src_id, src_slot = str(link[1]), link[2]
            if not include_muted and int(src_id) in nodes_by_id and nodes_by_id[int(src_id)].get("mode", 0) == 4:
                continue
            inputs[inp["name"]] = [src_id, src_slot]

        wvals = node.get("widgets_values")
        wnamed = node.get("widgets_values_named") or {}
        if isinstance(wnamed, dict) and wnamed:
            for k, v in wnamed.items():
                if k in ("upload", "videopreview", "choose video to upload", "🎲 Manual Random Seed", "audioUI"):
                    continue
                if k not in inputs:
                    inputs[k] = v
        elif isinstance(wvals, list):
            wnames = _widget_names(class_type, object_info)
            wi = 0
            for name in wnames:
                if name in inputs:
                    continue
                if wi < len(wvals):
                    inputs[name] = wvals[wi]
                    wi += 1
        elif isinstance(wvals, dict):
            for k, v in wvals.items():
                if k in ("videopreview",):
                    continue
                if k not in inputs:
                    inputs[k] = v

        prompt[nid] = {"class_type": class_type, "inputs": inputs}
        title = node.get("title") or class_type
        prompt[nid]["_meta"] = {"title": title}

    if not include_muted:
        # drop nodes whose required linked inputs point to missing nodes
        present = set(prompt)
        changed = True
        while changed:
            changed = False
            drop = []
            for nid, body in prompt.items():
                for v in body["inputs"].values():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                        if v[0] not in present:
                            drop.append(nid)
                            changed = True
                            break
            for nid in drop:
                prompt.pop(nid, None)
                present.discard(nid)

    return prompt


def patch_u06_full(prompt: dict, *, duration: float = 12.0, width: int = 1344, height: int = 768,
                   seed: int | None = None, image_files: list[str] | None = None,
                   video_file: str | None = None, audio_file: str | None = None) -> dict:
    if "132" in prompt:
        prompt["132"]["inputs"]["value"] = float(duration)
    if "665" in prompt:
        ins = prompt["665"]["inputs"]
        ins["预设分辨率"] = "自定义"
        ins["自定义宽"] = width
        ins["自定义高"] = height
        ins["横竖对调"] = False
    if seed is not None and "142" in prompt:
        prompt["142"]["inputs"]["seed"] = seed
    if "136" in prompt:
        prompt["136"]["inputs"]["ref_image_size"] = "max"

    load_ids = ["137", "139", "144", "151", "653", "654", "655", "656", "657"]
    if image_files:
        for i, nid in enumerate(load_ids):
            if nid in prompt and i < len(image_files):
                prompt[nid]["inputs"]["image"] = image_files[i]

    for nid in ("638", "659", "660"):
        if video_file and nid in prompt:
            ins = prompt[nid]["inputs"]
            if isinstance(ins.get("video"), str):
                ins["video"] = video_file
            ins["frame_load_cap"] = 360
            ins["force_rate"] = 24.0

    for nid in ("719", "720", "721"):
        if audio_file and nid in prompt:
            prompt[nid]["inputs"]["audio"] = audio_file

  # wire all 9 images + 3 videos + 3 audios into ref2v
    if "136" in prompt:
        r = prompt["136"]["inputs"]
        img_nodes = load_ids
        for i, nid in enumerate(img_nodes):
            if nid in prompt:
                r[f"ref_images.ref_image_{i}"] = [nid, 0]
        for i, chain_tail in enumerate(("737", "740", "743")):
            if chain_tail in prompt:
                r[f"ref_videos.ref_video_{i}"] = [chain_tail, 0]
        for i, nid in enumerate(("735", "733", "734")):
            if nid in prompt:
                r[f"ref_audios.ref_audio_{i}"] = [nid, 0]

    if "732" in prompt:
        prompt["732"]["inputs"]["filename_prefix"] = "u06_full_12s"

    return prompt


def main() -> None:
    import argparse
    import urllib.request

    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--base", default="https://u93272-7879ab9f6162.bjb2.seetacloud.com:8443")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--out", default="/tmp/u06_full_12s_prompt.json")
    args = ap.parse_args()

    wf = json.loads(Path(args.workflow).read_text())
    with urllib.request.urlopen(f"{args.base.rstrip('/')}/object_info", timeout=120) as r:
        obj = json.load(r)

    prompt = workflow_to_api(wf, obj, include_muted=True)
    prompt = patch_u06_full(prompt, duration=args.duration)
    Path(args.out).write_text(json.dumps(prompt, ensure_ascii=False, indent=2))
    print(f"wrote {args.out} nodes={len(prompt)}")


if __name__ == "__main__":
    main()
