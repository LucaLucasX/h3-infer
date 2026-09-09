#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folder_paths
from utils.extra_config import load_extra_path_config


IMAGE_NAMES = [
    "realref_bust_01.jpg",
    "realref_bust_02.jpg",
    "realref_bust_03.jpg",
    "realref_face_01.jpg",
    "realref_face_02.jpg",
    "realref_face_03.jpg",
    "realref_full_01.png",
    "realref_full_02.png",
    "realref_full_03.png",
]


def load_image(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16_baseline", "bf16_causal", "bf16_resident", "bf16_packed", "bf16_varlen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=9)
    parser.add_argument("--video-blocks", type=int, default=0)
    args = parser.parse_args()
    load_extra_path_config(ROOT / "extra_model_paths.yaml")
    import nodes
    import comfy.model_management
    from comfy.text_encoders.minimax import MiniMaxQwen3VL
    from comfy_extras.nodes_minimax_h3 import _short_edge_resize

    clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]
    ref_items = []
    loaded_images = []
    for name in IMAGE_NAMES[:args.images]:
        loaded_images.append(load_image(ROOT / "input" / name))
    for image in loaded_images:
        ref_items.append({"type": "image", "data": _short_edge_resize(image, 704)})
    if args.video_blocks:
        frame = _short_edge_resize(loaded_images[0], 544)
        frames = frame.repeat(args.video_blocks * 2, 1, 1, 1)
        ref_items.append({"type": "video", "data": frames, "timestamps": [i / 2 for i in range(frames.shape[0])]})
    tokens = clip.tokenize("A cinematic portrait with natural motion.", minimax_ref_items=ref_items)
    clip.load_model(tokens)
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"execution_device": device})
    encoder = clip.cond_stage_model.qwen3vl_32b
    transformer = encoder.transformer
    packed_method = MiniMaxQwen3VL.preprocess_embeds
    vision_attentions = [block.attn for block in transformer.visual.blocks]

    def run(packed, causal, varlen):
        if packed:
            MiniMaxQwen3VL.preprocess_embeds = packed_method
        elif hasattr(MiniMaxQwen3VL, "preprocess_embeds"):
            del MiniMaxQwen3VL.preprocess_embeds
        transformer.model.use_causal_sdpa = causal
        for attention in vision_attentions:
            attention.use_varlen_flash = varlen
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with comfy.model_management.cuda_device_context(device):
            output = clip.cond_stage_model.encode_token_weights(tokens)
        torch.cuda.synchronize()
        return {
            "seconds": time.perf_counter() - started,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "hidden": output[0].detach().cpu(),
            "tags": output[2]["minimax_token_tags"].detach().cpu(),
        }

    packed_enabled = args.mode in ("bf16_resident", "bf16_packed", "bf16_varlen")
    causal_enabled = args.mode != "bf16_baseline"
    varlen_enabled = args.mode == "bf16_varlen"
    result = run(packed_enabled, causal_enabled, varlen_enabled)
    MiniMaxQwen3VL.preprocess_embeds = packed_method
    transformer.model.use_causal_sdpa = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": result.pop("hidden"), "tags": result.pop("tags")}, args.output.with_suffix(".pt"))
    report = {"mode": args.mode, **result}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
