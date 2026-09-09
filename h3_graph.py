"""h3_graph — ComfyUI workflow builder for MiniMax H3.

Public API
----------
GenerateParams   dataclass describing one generation request
build_graph()    GenerateParams → ComfyUI /prompt dict
node()           helper to construct a single node dict
AttentionBackend type alias (str)
TURBO_LORA_V4   default turbo LoRA filename
twostep_pass1_size / twostep_pass2_size  geometry helpers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Default 4-step turbo LoRA (fl2v); same value historically stored as TURBO_LORA_V4
TURBO_LORA_V4 = "lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"

AttentionBackend = Literal[
    "pytorch",
    "sage_v1",
    "sage_v1_triton",
    "sage_v2_memeff",
    "sage_v3",
    "disabled",
]

# Model filenames (ComfyUI looks these up via extra_model_paths.yaml)
_UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
_UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
_CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
_AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"

# Sage attention patch node names keyed by backend alias
_SAGE_NODE: dict[str, str] = {
    "sage_v1":        "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "sage_v1_triton": "MiniMaxH3SageAttentionPatch",
    "sage_v2_memeff": "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "sage_v3":        "MiniMaxH3SageAttentionV3Patch",
}

# Default sigma shifts
_SHIFT_VIDEO = 12.0
_SHIFT_AUDIO = 3.0

# Community effect embeddings (Comfy-Org/MiniMax-H3 embeddings/, silveroxides)
H3_EFFECT_EMBEDDINGS: dict[str, str] = {
    "none": "",
    "art_is_explosion": "minimaxh3_art_is_explosion",
    "storm_magic": "minimaxh3_storm_magic",
    "dark_magic": "minimaxh3_dark_magic",
    "bullet_time": "minimaxh3_bullet_time",
    "fire_breath": "minimaxh3_fire_breath",
    "blooming_flowers": "minimaxh3_blooming_flowers",
    "four_seasons": "minimaxh3_four_seasons",
    "kiss_camera": "minimaxh3_kiss_camera",
    "spiral_ascent": "minimaxh3_spiral_ascent",
    "truman_show": "minimaxh3_truman_show",
}

# Twostep constants
_TWOSTEP_SCALE = 0.5        # pass-1 resolution factor
_TWOSTEP_SPLIT_FRAC = 0.5   # fraction of steps in pass-1


# ---------------------------------------------------------------------------
# node() helper
# ---------------------------------------------------------------------------

def node(class_type: str, inputs: dict) -> dict:
    """Return a single ComfyUI prompt node dict."""
    return {"class_type": class_type, "inputs": inputs}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def twostep_pass1_size(width: int, height: int) -> tuple[int, int]:
    """Resolution of the first (low-res) pass in twostep mode."""
    multiple = 32
    w = max(multiple, round(width * _TWOSTEP_SCALE / multiple) * multiple)
    h = max(multiple, round(height * _TWOSTEP_SCALE / multiple) * multiple)
    return w, h


def twostep_pass2_size(p1w: int, p1h: int) -> tuple[int, int]:
    """Approximate upscaled canvas for pass-2 (reencode from pass-1)."""
    multiple = 32
    w = max(multiple, round(p1w / _TWOSTEP_SCALE / multiple) * multiple)
    h = max(multiple, round(p1h / _TWOSTEP_SCALE / multiple) * multiple)
    return w, h


def apply_effect_embedding(prompt: str, effect_embedding: str = "none") -> str:
    """Prepend `embedding:<name>,` when a preset effect is selected; `none` leaves prompt unchanged."""
    key = (effect_embedding or "none").strip()
    if not key or key.lower() == "none":
        return prompt
    embed_name = H3_EFFECT_EMBEDDINGS.get(key)
    if embed_name is None:
        if key.startswith("minimaxh3_"):
            embed_name = key
        elif key in H3_EFFECT_EMBEDDINGS.values():
            embed_name = key
        else:
            choices = ", ".join(sorted(k for k in H3_EFFECT_EMBEDDINGS if k != "none"))
            raise ValueError(f"unknown effect_embedding {key!r}; choose one of: none, {choices}")
    if not embed_name:
        return prompt
    body = (prompt or "").strip()
    if not body:
        return f"embedding:{embed_name}"
    return f"embedding:{embed_name}, {body}"


# ---------------------------------------------------------------------------
# GenerateParams
# ---------------------------------------------------------------------------

@dataclass
class GenerateParams:
    """All parameters for one H3 generation.

    mode:
        "t2v"      – text-to-video (fl2va, no keyframes)
        "i2v"      – image-to-video (fl2va with first/last keyframe)
        "r2v"      – reference-to-video (ref2va, multi-ref images/videos)
        "twostep"  – two-pass upscale (low-res pass1 → upscale → pass2)
    """

    # ---- required ----
    mode: Literal["t2v", "i2v", "r2v", "twostep"] = "t2v"
    prompt: str = ""
    width: int = 1280
    height: int = 704
    length: int = 124          # frames @ 24 fps; snapped to 17k+5 grid
    steps: int = 20
    seed: int = 42
    output_prefix: str = "h3_output/out"

    # ---- keyframe / reference inputs ----
    image: str = ""            # first-frame filename (in ComfyUI input/)
    last_frame: str | None = None  # last-frame filename
    ref_images: list[str] = field(default_factory=list)   # ≤9 for r2v
    ref_videos: list[str] = field(default_factory=list)   # ≤3 for r2v
    ref_audios: list[str] = field(default_factory=list)   # ≤3 standalone audio refs

    # ---- acceleration ----
    attention_backend: AttentionBackend = "pytorch"
    # legacy alias; if set, overrides attention_backend
    sage_attention: str = "disabled"

    # ---- turbo LoRA ----
    turbo: bool = False
    turbo_lora: str = TURBO_LORA_V4
    turbo_strength: float = 1.0

    # ---- SpargeAttn (replaces Sage patch when enabled; dense path is Sage2) ----
    sparge: bool = False
    sparge_topk: float = 0.5
    sparge_dense_first_steps: int = 0
    sparge_num_layers: int = 50
    sparge_audio_dense: bool = True

    # ---- weights (empty → mode-dependent production defaults) ----
    unet_name: str = ""
    clip_name: str = ""

    # ---- sampler / scheduler ----
    fps: float = 24.0
    sampler: str = "res_multistep"
    scheduler: str = "simple"

    # ---- accel flags (used by build_graph) ----
    fp16_accumulation: bool = False
    easycache: bool = False
    teacache: bool = False
    tespeed: bool = False
    spectrum: bool = False

    # ---- remote U06 lossless VRAM (KJNodes; math-equivalent chunking) ----
    kj_lowvram: bool = False
    kj_head_chunks: int = 4
    kj_ffn_chunks: int = 2
    kj_seq_threshold: int = 4096

    # ---- twostep options ----
    twostep_upscale_method: str = "bicubic"
    twostep_audio_denoise: float = 0.5
    twostep_full_pass1: bool = False
    twostep_reencode_pass2_cond: bool = True

    # ---- ref2va sizing ----
    ref_image_size: str = "match"  # "match"=704 short | "max"=2048 short；长边 round 32

    # ---- effect embedding preset (ComfyUI `embedding:` syntax in TE prompt) ----
    effect_embedding: str = "none"

    # -----------------------------------------------------------------------
    def resolved_prompt(self) -> str:
        return apply_effect_embedding(self.prompt, self.effect_embedding)

    def _resolved_attention(self) -> AttentionBackend:
        """Return the effective attention backend (handles legacy sage_attention)."""
        # sage_attention="disabled" means "use attention_backend directly"
        if self.sage_attention not in ("disabled", ""):
            return self.sage_attention  # type: ignore[return-value]
        return self.attention_backend

    def validate(self) -> None:
        """Raise ValueError for obviously invalid combinations."""
        if self.mode not in ("t2v", "i2v", "r2v", "twostep"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "i2v" and not self.image and not self.last_frame:
            raise ValueError("i2v mode requires at least one of image / last_frame")
        if self.mode == "r2v" and not self.ref_images and not self.ref_videos:
            raise ValueError("r2v mode requires ref_images or ref_videos")
        if self.width % 32 or self.height % 32:
            raise ValueError(f"width/height must be multiples of 32, got {self.width}x{self.height}")

    def resolve_twostep(self) -> tuple[int, int, int, int, int]:
        """Return (p1w, p1h, p2w, p2h, split_steps) for twostep mode."""
        p1w, p1h = twostep_pass1_size(self.width, self.height)
        p2w, p2h = twostep_pass2_size(p1w, p1h)
        split = max(1, int(math.ceil(self.steps * _TWOSTEP_SPLIT_FRAC)))
        return p1w, p1h, p2w, p2h, split


# ---------------------------------------------------------------------------
# Internal graph-building helpers
# ---------------------------------------------------------------------------

def _align_length(length: int) -> int:
    """Snap frame count to the 17k+5 grid (minimum 5)."""
    n = max(5, length)
    while n % 17 != 5:
        n += 1
    return n


def _default_unet(p: GenerateParams) -> str:
    if p.unet_name:
        return p.unet_name
    if p.mode == "r2v":
        return _UNET_REF2VA
    return _UNET_FL2VA


def _base_nodes(p: GenerateParams) -> dict:
    """Nodes shared by all modes: UNETLoader, CLIPLoader, VAELoaders."""
    g: dict[str, dict] = {}

    # 1: UNETLoader
    g["1"] = node("UNETLoader", {
        "unet_name": _default_unet(p),
        "weight_dtype": "default",
    })

    # 2: CLIPLoader  (type must be "minimax")
    g["2"] = node("CLIPLoader", {
        "clip_name": p.clip_name or _CLIP_NAME,
        "type": "minimax",
        "device": "default",
    })

    # 3: VAELoader (video)
    g["3"] = node("VAELoader", {"vae_name": _VAE_NAME})

    # 4: VAELoader (audio)
    g["4"] = node("VAELoader", {"vae_name": _AUDIO_VAE_NAME})

    # After possible patches, model wire starts at node 1 output.
    # Patches (sage, teacache, tespeed…) will insert nodes and re-wire.
    return g


def _apply_attention_patch(g: dict, model_ref: list, p: GenerateParams) -> list:
    """Insert Sparge and/or Sage as node 5 (downstream always consumes [5,0] when present).

    Production stack uses SpargeAttn (topk=0.5) whose dense path is Sage2.
    Scripts historically overwrite g['5'] with MiniMaxH3SpargeAttnPatchExp.
    """
    if p.sparge:
        g["5"] = node(
            "MiniMaxH3SpargeAttnPatchExp",
            {
                "model": model_ref,
                "topk": float(p.sparge_topk),
                "dense_first_steps": int(p.sparge_dense_first_steps),
                "num_layers": int(p.sparge_num_layers),
                "audio_dense": bool(p.sparge_audio_dense),
            },
        )
        return ["5", 0]

    attn = p._resolved_attention()
    if attn in ("pytorch", "disabled", ""):
        return model_ref
    sage_class = _SAGE_NODE.get(attn)
    if sage_class is None:
        return model_ref
    g["5"] = node(sage_class, {"model": model_ref})
    return ["5", 0]


def _apply_turbo_lora(g: dict, model_ref: list, p: GenerateParams) -> list:
    """Insert MiniMaxH3TurboLoRA node if turbo=True."""
    if not p.turbo:
        return model_ref
    nid = "21"
    g[nid] = node("MiniMaxH3TurboLoRA", {
        "model": model_ref,
        "lora_name": p.turbo_lora,
        "strength": p.turbo_strength,
        "low_vram": False,
    })
    return [nid, 0]


def _apply_kj_lowvram(g: dict, model_ref: list, p: GenerateParams) -> list:
    """U06 MiniMaxLowVRAMAttention + MiniMaxChunkFeedForward (lossless token/head chunks)."""
    if not p.kj_lowvram:
        return model_ref
    g["23"] = node("MiniMaxLowVRAMAttention", {
        "model": model_ref,
        "head_chunks": int(p.kj_head_chunks),
    })
    g["24"] = node("MiniMaxChunkFeedForward", {
        "model": ["23", 0],
        "chunks": int(p.kj_ffn_chunks),
        "seq_threshold": int(p.kj_seq_threshold),
    })
    return ["24", 0]


def _apply_accel_patches(g: dict, model_ref: list, p: GenerateParams) -> list:
    """Insert optional acceleration-patch nodes (tespeed, teacache, spectrum…)."""
    nid_counter = [30]

    def _next() -> str:
        nid = str(nid_counter[0])
        nid_counter[0] += 1
        return nid

    if p.tespeed:
        nid = _next()
        g[nid] = node("TESpeedMiniMaxH3", {
            "model": model_ref,
            "processing_control_value": 0.12,
            "processing_percent_1": 0.1,
            "processing_percent_2": 0.9,
            "mcs": 2,
            "device": "auto",
            "cache_depth": 0.75,
        })
        model_ref = [nid, 0]

    if p.teacache:
        nid = _next()
        g[nid] = node("MiniMaxH3TeaCacheModelPatch", {
            "model": model_ref,
            "rel_l1_thresh": 0.15,
            "start_step": 0,
            "end_step": -1,
        })
        model_ref = [nid, 0]

    if p.easycache:
        nid = _next()
        g[nid] = node("MiniMaxH3EasyCacheModelPatch", {
            "model": model_ref,
            "residual_diff_threshold": 0.08,
        })
        model_ref = [nid, 0]

    if p.spectrum:
        nid = _next()
        g[nid] = node("SpectrumMiniMaxH3Patch", {
            "model": model_ref,
        })
        model_ref = [nid, 0]

    if p.fp16_accumulation:
        nid = _next()
        g[nid] = node("MiniMaxH3FP16AccumulationPatch", {
            "model": model_ref,
        })
        model_ref = [nid, 0]

    return model_ref


def _sigma_shift_node(g: dict, model_ref: list) -> list:
    """Insert MiniMaxH3SigmaShift, return new model ref."""
    nid = "40"
    g[nid] = node("MiniMaxH3SigmaShift", {
        "model": model_ref,
        "shift_video": _SHIFT_VIDEO,
        "shift_audio": _SHIFT_AUDIO,
    })
    return [nid, 0]


def _sampler_nodes(
    g: dict,
    model_ref: list,
    cond_ref: list,
    latent_ref: list,
    p: GenerateParams,
    *,
    node_prefix: str = "",
) -> list:
    """
    Insert the sampling chain:
      BasicGuider → noise → sigmas → SamplerCustomAdvanced → split latent.

    Returns [output_latent_video_ref, output_latent_audio_ref].
    """
    pfx = node_prefix

    noise_nid = pfx + "50"
    guider_nid = pfx + "51"
    sigmas_nid = pfx + "52"
    sampler_nid = pfx + "53"

    g[noise_nid] = node("RandomNoise", {"noise_seed": p.seed})
    g[guider_nid] = node("BasicGuider", {
        "model": model_ref,
        "conditioning": cond_ref,
    })

    if p.turbo:
        # Turbo sampler provides its own sigma sequence
        sampler_class = "MiniMaxH3TurboSampler"
        g[sigmas_nid] = node("BasicScheduler", {
            "model": model_ref,
            "scheduler": p.scheduler,
            "steps": p.steps,
            "denoise": 1.0,
        })
        sampler_nid_inner = pfx + "54"
        g[sampler_nid_inner] = node(sampler_class, {})
        sampler_ref = [sampler_nid_inner, 0]
    else:
        g[sigmas_nid] = node("BasicScheduler", {
            "model": model_ref,
            "scheduler": p.scheduler,
            "steps": p.steps,
            "denoise": 1.0,
        })
        sampler_inner_nid = pfx + "54"
        g[sampler_inner_nid] = node("KSamplerSelect", {"sampler_name": p.sampler})
        sampler_ref = [sampler_inner_nid, 0]

    g[sampler_nid] = node("SamplerCustomAdvanced", {
        "noise": [noise_nid, 0],
        "guider": [guider_nid, 0],
        "sampler": sampler_ref,
        "sigmas": [sigmas_nid, 0],
        "latent_image": latent_ref,
    })

    # Nested AV latent: the same output[0] is decoded by both VAEs.
    return [[sampler_nid, 0], [sampler_nid, 0]]


def _decode_and_save(
    g: dict,
    video_latent_ref: list,
    audio_latent_ref: list,
    vae_ref: list,
    audio_vae_ref: list,
    output_prefix: str,
    *,
    node_prefix: str = "",
    fps: float = 24.0,
) -> None:
    """Insert VAEDecode → VAEDecodeAudio → CreateVideo → SaveVideo."""
    pfx = node_prefix

    vdec = pfx + "60"
    adec = pfx + "61"
    comb = pfx + "62"
    save = pfx + "63"

    g[vdec] = node("VAEDecode", {
        "samples": video_latent_ref,
        "vae": vae_ref,
    })
    g[adec] = node("VAEDecodeAudio", {
        "samples": audio_latent_ref,
        "vae": audio_vae_ref,
    })
    g[comb] = node("CreateVideo", {
        "images": [vdec, 0],
        "audio": [adec, 0],
        "fps": fps,
        "bit_depth": 8,
    })
    g[save] = node("SaveVideo", {
        "video": [comb, 0],
        "filename_prefix": output_prefix,
        "format": "auto",
        "codec": "auto",
    })


# ---------------------------------------------------------------------------
# Mode-specific conditioning nodes
# ---------------------------------------------------------------------------

def _t2v_cond(g: dict, clip_ref: list, vae_ref: list, p: GenerateParams) -> tuple[list, list]:
    """MiniMaxH3ImageToVideo (t2v: no keyframes)."""
    nid = "6"
    length = _align_length(p.length)
    g[nid] = node("MiniMaxH3ImageToVideo", {
        "clip": clip_ref,
        "vae": vae_ref,
        "prompt": p.resolved_prompt(),
        "width": p.width,
        "height": p.height,
        "length": length,
    })
    return [nid, 0], [nid, 1]  # conditioning, latent


def _i2v_cond(g: dict, clip_ref: list, vae_ref: list, p: GenerateParams) -> tuple[list, list]:
    """MiniMaxH3ImageToVideo (fl2va: with first/last keyframes)."""
    nid = "6"
    length = _align_length(p.length)
    inputs: dict = {
        "clip": clip_ref,
        "vae": vae_ref,
        "prompt": p.resolved_prompt(),
        "width": p.width,
        "height": p.height,
        "length": length,
    }
    img_nid_counter = [80]

    def _load_image(fname: str) -> list:
        nid_ = str(img_nid_counter[0])
        img_nid_counter[0] += 1
        g[nid_] = node("LoadImage", {"image": fname})
        return [nid_, 0]

    if p.image:
        inputs["first_frame"] = _load_image(p.image)
    if p.last_frame:
        inputs["last_frame"] = _load_image(p.last_frame)

    g[nid] = node("MiniMaxH3ImageToVideo", inputs)
    return [nid, 0], [nid, 1]


def _r2v_cond(
    g: dict, clip_ref: list, vae_ref: list, audio_vae_ref: list, p: GenerateParams
) -> tuple[list, list]:
    """MiniMaxH3ReferenceToVideo (ref2va)."""
    nid = "6"
    length = _align_length(p.length)
    inputs: dict = {
        "clip": clip_ref,
        "vae": vae_ref,
        "audio_vae": audio_vae_ref,
        "prompt": p.resolved_prompt(),
        "width": p.width,
        "height": p.height,
        "length": length,
        "ref_image_size": p.ref_image_size,
    }
    img_nid_counter = [80]
    vid_nid_counter = [90]

    for i, fname in enumerate(p.ref_images[:9]):
        load_nid = str(img_nid_counter[0])
        img_nid_counter[0] += 1
        g[load_nid] = node("LoadImage", {"image": fname})
        inputs[f"ref_images.ref_image_{i}"] = [load_nid, 0]

    for i, fname in enumerate(p.ref_videos[:3]):
        load_nid = str(vid_nid_counter[0])
        vid_nid_counter[0] += 1
        split_nid = str(vid_nid_counter[0])
        vid_nid_counter[0] += 1
        g[load_nid] = node("LoadVideo", {"file": fname})
        g[split_nid] = node("GetVideoComponents", {"video": [load_nid, 0]})
        inputs[f"ref_videos.ref_video_{i}"] = [split_nid, 0]
        inputs[f"ref_video_audios.ref_video_audio_{i}"] = [split_nid, 1]

    aud_nid_counter = [70]
    for i, fname in enumerate(p.ref_audios[:3]):
        load_nid = str(aud_nid_counter[0])
        aud_nid_counter[0] += 1
        g[load_nid] = node("LoadAudio", {"audio": fname})
        inputs[f"ref_audios.ref_audio_{i}"] = [load_nid, 0]

    g[nid] = node("MiniMaxH3ReferenceToVideo", inputs)
    return [nid, 0], [nid, 1]


# ---------------------------------------------------------------------------
# twostep: two-pass low-res → upscale → high-res
# ---------------------------------------------------------------------------

def _build_twostep(g: dict, model_ref: list, p: GenerateParams) -> None:
    """Build a two-pass workflow into g."""
    clip_ref = ["2", 0]
    vae_ref = ["3", 0]
    audio_vae_ref = ["4", 0]

    p1w, p1h, p2w, p2h, split = p.resolve_twostep()

    # Pass-1 params
    p1 = GenerateParams(
        mode="t2v" if not p.image else "i2v",
        prompt=p.resolved_prompt(),
        width=p1w,
        height=p1h,
        length=p.length,
        steps=split,
        seed=p.seed,
        image=p.image,
        last_frame=p.last_frame,
        turbo=p.turbo,
        turbo_lora=p.turbo_lora,
        turbo_strength=p.turbo_strength,
        sampler=p.sampler,
        scheduler=p.scheduler,
        output_prefix=p.output_prefix + "_pass1",
    )

    if p1.mode == "t2v":
        cond_ref1, latent_ref1 = _t2v_cond(g, clip_ref, vae_ref, p1)
    else:
        cond_ref1, latent_ref1 = _i2v_cond(g, clip_ref, vae_ref, p1)

    latent_outs1 = _sampler_nodes(g, model_ref, cond_ref1, latent_ref1, p1, node_prefix="p1_")

    # Decode pass-1 video, upscale, reencode
    vdec1 = "p1_vdec"
    g[vdec1] = node("VAEDecode", {"samples": latent_outs1[0], "vae": vae_ref})

    upscale_nid = "p1_up"
    g[upscale_nid] = node("ImageScale", {
        "image": [vdec1, 0],
        "upscale_method": p.twostep_upscale_method,
        "width": p2w,
        "height": p2h,
        "crop": "disabled",
    })

    reencode_nid = "p1_renc"
    g[reencode_nid] = node("VAEEncode", {
        "pixels": [upscale_nid, 0],
        "vae": vae_ref,
    })

    # Pass-2 conditioning at full resolution
    p2 = GenerateParams(
        mode="t2v" if not p.image else "i2v",
        prompt=p.resolved_prompt(),
        width=p2w,
        height=p2h,
        length=p.length,
        steps=p.steps - split,
        seed=p.seed,
        image=p.image,
        last_frame=p.last_frame,
        turbo=p.turbo,
        turbo_lora=p.turbo_lora,
        turbo_strength=p.turbo_strength,
        sampler=p.sampler,
        scheduler=p.scheduler,
        output_prefix=p.output_prefix,
    )

    if p2.mode == "t2v":
        cond_ref2, _ = _t2v_cond(g, clip_ref, vae_ref, p2)
    else:
        cond_ref2, _ = _i2v_cond(g, clip_ref, vae_ref, p2)

    latent_ref2 = [reencode_nid, 0]

    latent_outs2 = _sampler_nodes(g, model_ref, cond_ref2, latent_ref2, p2, node_prefix="p2_")

    _decode_and_save(
        g,
        latent_outs2[0], latent_outs2[1],
        vae_ref, audio_vae_ref,
        p.output_prefix,
        node_prefix="p2_",
        fps=p.fps,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_graph(p: GenerateParams) -> dict:
    """Build a ComfyUI API prompt dict from GenerateParams.

    Returns a flat dict of node_id → node_dict.
    """
    g = _base_nodes(p)

    clip_ref: list = ["2", 0]
    vae_ref: list = ["3", 0]
    audio_vae_ref: list = ["4", 0]
    model_ref: list = ["1", 0]

    # ---- attention patch (Sparge and/or Sage as node 5) ----
    model_ref = _apply_attention_patch(g, model_ref, p)

    # ---- turbo LoRA ----
    model_ref = _apply_turbo_lora(g, model_ref, p)

    # ---- U06 lossless VRAM (KJ LowVRAM attn + chunked FFN) ----
    model_ref = _apply_kj_lowvram(g, model_ref, p)

    # ---- acceleration patches ----
    model_ref = _apply_accel_patches(g, model_ref, p)

    # ---- sigma shift (official turbo workflow skips this; turbo sampler owns AV clocks) ----
    if not p.turbo:
        model_ref = _sigma_shift_node(g, model_ref)

    # ---- twostep shortcut ----
    if p.mode == "twostep":
        _build_twostep(g, model_ref, p)
        return g

    # ---- conditioning ----
    if p.mode == "t2v":
        cond_ref, latent_ref = _t2v_cond(g, clip_ref, vae_ref, p)
    elif p.mode == "i2v":
        cond_ref, latent_ref = _i2v_cond(g, clip_ref, vae_ref, p)
    elif p.mode == "r2v":
        cond_ref, latent_ref = _r2v_cond(g, clip_ref, vae_ref, audio_vae_ref, p)
    else:
        raise ValueError(f"unknown mode {p.mode!r}")

    # ---- sampling ----
    latent_outs = _sampler_nodes(g, model_ref, cond_ref, latent_ref, p)

    # ---- decode + save ----
    _decode_and_save(
        g,
        latent_outs[0], latent_outs[1],
        vae_ref, audio_vae_ref,
        p.output_prefix,
        fps=p.fps,
    )

    return g
