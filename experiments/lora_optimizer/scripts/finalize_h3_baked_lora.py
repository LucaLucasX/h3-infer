"""Finalize LoRA Optimizer bake into TurboLoRA-compatible h3keys (208 modules)."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from h3_lora_merge_utils import (
    DEFAULT_PROD,
    DEFAULT_SLA,
    fc2_modules,
    module_bases,
    normalize_to_h3keys,
    rank_concat_layer,
    strip_dm,
)


def _rename_ab(sd: dict) -> dict:
    """Optimizer SaveMergedLoRA uses lora_up=B, lora_down=A; h3keys uses lora_A/B."""
    out: dict = {}
    for key, val in sd.items():
        if key.endswith(".lora_up.weight"):
            out[key.replace(".lora_up.weight", ".lora_B.weight")] = val
        elif key.endswith(".lora_down.weight"):
            out[key.replace(".lora_down.weight", ".lora_A.weight")] = val
        else:
            out[key] = val
    return out


def finalize_optimizer_baked_for_h3(
    baked_path: Path,
    *,
    prod_path: Path = DEFAULT_PROD,
    sla_path: Path = DEFAULT_SLA,
    strength_prod: float = 0.5886,
    strength_sla: float = 0.5886,
) -> dict:
    """Merge Optimizer-baked layers + rank-concat fallback for keys Optimizer failed to save."""
    with safe_open(str(baked_path), framework="pt") as fb:
        meta = dict(fb.metadata() or {})
    baked = _rename_ab(normalize_to_h3keys(load_file(baked_path)))
    prod = normalize_to_h3keys(load_file(prod_path))
    sla = normalize_to_h3keys(load_file(sla_path))

    out: dict = {}
    baked_modules = set(module_bases(baked))
    from_optimizer = 0
    from_fallback = 0

    for base in module_bases(prod):
        a_key, b_key, alpha_key = (
            base + ".lora_A.weight",
            base + ".lora_B.weight",
            base + ".alpha",
        )
        if base in baked_modules and all(k in baked for k in (a_key, b_key, alpha_key)):
            out[a_key] = baked[a_key]
            out[b_key] = baked[b_key]
            out[alpha_key] = baked[alpha_key]
            from_optimizer += 1
            continue
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
        out[a_key] = a_out
        out[b_key] = b_out
        out[alpha_key] = alpha_out
        from_fallback += 1

    meta["h3_finalize"] = (
        f"optimizer_layers={from_optimizer} fallback_rank_concat={from_fallback} "
        f"strengths={strength_prod:g}/{strength_sla:g}"
    )
    meta["h3_keys_layout"] = "h3keys lora_A/B for MiniMaxH3TurboLoRA"
    save_file(out, str(baked_path), metadata=meta)
    return {
        "modules": len(module_bases(out)),
        "fc2_modules": len(fc2_modules(out)),
        "from_optimizer": from_optimizer,
        "from_fallback": from_fallback,
        "strength_prod": strength_prod,
        "strength_sla": strength_sla,
    }
