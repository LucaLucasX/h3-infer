"""Shared helpers for H3 Turbo LoRA rank-concat merge (fc2 patch, etc.)."""
from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

DEFAULT_PROD = Path(
    "/big_models/comfyui-minimax-H3/loras/lightx2v_fl2v/"
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"
)
DEFAULT_SLA = Path(
    "/mnt/luca/H3_infer/experiments/sla/models/loras/lightx2v_fl2v/"
    "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
)


def strip_dm(key: str) -> str:
    return key[len("diffusion_model.") :] if key.startswith("diffusion_model.") else key


def module_bases(sd: dict) -> list[str]:
    return sorted({k.rsplit(".lora_", 1)[0] for k in sd if ".lora_" in k})


def fc2_modules(sd: dict) -> list[str]:
    return [m for m in module_bases(sd) if m.endswith(".mlp.fc2")]


def parse_source_strengths(metadata: dict[str, str], prod_name: str, sla_name: str) -> tuple[float, float]:
    """Parse applied strengths from SaveMergedLoRA metadata header."""
    raw = metadata.get("source_loras") or ""
    strengths: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if " @ " not in part:
            continue
        name, val = part.rsplit(" @ ", 1)
        try:
            strengths[name.strip()] = float(val.strip())
        except ValueError:
            continue
    if prod_name in strengths and sla_name in strengths:
        return strengths[prod_name], strengths[sla_name]
    m = re.findall(r"([\d.]+)", raw)
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    raise ValueError(f"cannot parse source_loras strengths from metadata: {raw!r}")


def rank_concat_layer(
    a_prod: torch.Tensor,
    b_prod: torch.Tensor,
    alpha_prod: torch.Tensor,
    a_sla: torch.Tensor,
    b_sla: torch.Tensor,
    alpha_sla: torch.Tensor,
    *,
    strength_prod: float,
    strength_sla: float,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = a_prod.shape[0]
    c_prod = strength_prod * (float(alpha_prod) / rank)
    c_sla = strength_sla * (float(alpha_sla) / rank)
    a_out = torch.cat([a_prod.to(dtype), a_sla.to(dtype)], dim=0).contiguous()
    b_out = torch.cat(
        [(c_prod * b_prod.float()).to(dtype), (c_sla * b_sla.float()).to(dtype)],
        dim=1,
    ).contiguous()
    alpha_out = torch.tensor(float(a_out.shape[0]), dtype=alpha_prod.dtype)
    return a_out, b_out, alpha_out


def normalize_to_h3keys(sd: dict) -> dict:
    """Canonical h3keys layout: bare blocks.* / token_refiner.* (no diffusion_model.)."""
    out: dict = {}
    for key, val in sd.items():
        nk = strip_dm(key) if key.startswith("diffusion_model.") else key
        out[nk] = val
    return out


def patch_fc2_into_baked(
    baked_path: Path,
    *,
    prod_path: Path = DEFAULT_PROD,
    sla_path: Path = DEFAULT_SLA,
    prod_lora_name: str,
    sla_lora_name: str,
    strength_prod: float | None = None,
    strength_sla: float | None = None,
) -> dict:
    """Append optimizer-missing H3 int8 fc2 layers via rank-concat at baked strengths."""
    with safe_open(str(baked_path), framework="pt") as fb:
        meta = dict(fb.metadata() or {})
    baked = normalize_to_h3keys(load_file(baked_path))

    if strength_prod is None or strength_sla is None:
        try:
            strength_prod, strength_sla = parse_source_strengths(meta, prod_lora_name, sla_lora_name)
        except ValueError:
            strength_prod, strength_sla = 0.5886, 0.5886
    # Optimizer auto-strength is baked into non-fc2 weights; metadata still lists 1.0.
    if strength_prod == 1.0 and strength_sla == 1.0:
        strength_prod = strength_sla = 0.5886

    prod = normalize_to_h3keys(load_file(prod_path))
    sla = normalize_to_h3keys(load_file(sla_path))

    added = 0
    for base in fc2_modules(prod):
        if base in module_bases(baked):
            continue
        a_key, b_key, alpha_key = (
            base + ".lora_A.weight",
            base + ".lora_B.weight",
            base + ".alpha",
        )
        a_out, b_out, alpha_out = rank_concat_layer(
            prod[a_key],
            prod[b_key],
            prod[alpha_key],
            sla[a_key],
            sla[b_key],
            sla[alpha_key],
            strength_prod=strength_prod,
            strength_sla=strength_sla,
        )
        baked[a_key] = a_out
        baked[b_key] = b_out
        baked[alpha_key] = alpha_out
        added += 1

    meta["h3_fc2_patch"] = (
        f"rank-concat fc2 from {prod_lora_name}@{strength_prod:g} + "
        f"{sla_lora_name}@{strength_sla:g} (Optimizer cannot map int8 fc2)"
    )
    meta["h3_keys_layout"] = "h3keys bare blocks.* (diffusion_model. stripped)"
    save_file(baked, str(baked_path), metadata=meta)
    info = {
        "fc2_added": added,
        "modules": len(module_bases(baked)),
        "fc2_modules": len(fc2_modules(baked)),
        "strength_prod": strength_prod,
        "strength_sla": strength_sla,
    }
    return info
