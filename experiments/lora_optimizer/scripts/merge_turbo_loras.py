#!/usr/bin/env python3
"""Offline additive merge of two H3 Turbo LoRAs into one h3keys safetensors file.

Uses exact rank concatenation so MiniMaxH3TurboLoRA can load the result (including
all 52 mlp.fc2 layers on the INT8 pruned base). This avoids LoRA Optimizer's
in-memory patch path, which drops fc2 on H3.

  Δ = s1*(α/r) B1 A1 + s2*(α/r) B2 A2
  A' = cat([A1, A2], 0)
  B' = cat([s1*(α1/r) B1, s2*(α2/r) B2], 1)
  α' = rank(A')  => effective scale α'/rank' = 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROD = Path(
    "/big_models/comfyui-minimax-H3/loras/lightx2v_fl2v/"
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_h3keys.safetensors"
)
DEFAULT_SLA = (
    REPO
    / "experiments/sla/models/loras/lightx2v_fl2v/"
    "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
)
DEFAULT_OUT = Path(
    "/big_models/comfyui-minimax-H3/loras/lightx2v_fl2v/"
    "minimax_h3_fl2v_turbo_4step_merge_prod0.7_sla0.3_h3keys.safetensors"
)


def _strip_dm(key: str) -> str:
    return key[len("diffusion_model.") :] if key.startswith("diffusion_model.") else key


def merge_loras(
    prod_path: Path,
    sla_path: Path,
    out_path: Path,
    *,
    strength_prod: float,
    strength_sla: float,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    with safe_open(str(prod_path), framework="pt") as fp, safe_open(str(sla_path), framework="pt") as fs:
        prod_keys = set(fp.keys())
        sla_map = {_strip_dm(k): k for k in fs.keys()}
        missing = prod_keys ^ set(sla_map)
        if missing:
            raise ValueError(
                f"key mismatch: prod={len(prod_keys)} sla_norm={len(sla_map)} "
                f"symdiff={len(missing)} sample={sorted(missing)[:5]}"
            )

        bases = sorted(
            k[: -len(".lora_A.weight")]
            for k in prod_keys
            if k.endswith(".lora_A.weight")
        )
        out: dict[str, torch.Tensor] = {}
        for base in bases:
            a_key, b_key, alpha_key = (
                base + ".lora_A.weight",
                base + ".lora_B.weight",
                base + ".alpha",
            )
            a_prod = fp.get_tensor(a_key)
            b_prod = fp.get_tensor(b_key)
            alpha_prod = fp.get_tensor(alpha_key)
            a_sla = fs.get_tensor(sla_map[a_key])
            b_sla = fs.get_tensor(sla_map[b_key])
            alpha_sla = fs.get_tensor(sla_map[alpha_key])
            if a_prod.shape != a_sla.shape or b_prod.shape != b_sla.shape:
                raise ValueError(f"shape mismatch at {base}: prod {a_prod.shape}/{b_prod.shape} "
                                 f"sla {a_sla.shape}/{b_sla.shape}")

            rank = a_prod.shape[0]
            c_prod = strength_prod * (float(alpha_prod) / rank)
            c_sla = strength_sla * (float(alpha_sla) / rank)
            a_out = torch.cat([a_prod.to(dtype), a_sla.to(dtype)], dim=0).contiguous()
            b_out = torch.cat(
                [(c_prod * b_prod.float()).to(dtype), (c_sla * b_sla.float()).to(dtype)],
                dim=1,
            ).contiguous()
            out[a_key] = a_out
            out[b_key] = b_out
            out[alpha_key] = torch.tensor(float(a_out.shape[0]), dtype=alpha_prod.dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))
    return out


def _summary(sd: dict[str, torch.Tensor]) -> dict:
    mods = sorted({k.rsplit(".lora_", 1)[0] for k in sd if ".lora_" in k})
    return {
        "tensors": len(sd),
        "modules": len(mods),
        "fc2_modules": sum(1 for m in mods if "fc2" in m),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline merge H3 Turbo LoRAs (rank concat).")
    ap.add_argument("--prod", type=Path, default=DEFAULT_PROD)
    ap.add_argument("--sla", type=Path, default=DEFAULT_SLA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strength-prod", type=float, default=0.7)
    ap.add_argument("--strength-sla", type=float, default=0.3)
    ap.add_argument("--verify-only", action="store_true",
                    help="Only check output exists and has 208 modules incl. fc2.")
    args = ap.parse_args()

    if args.verify_only:
        if not args.out.is_file():
            raise SystemExit(f"missing merged file: {args.out}")
        from safetensors.torch import load_file

        info = _summary(load_file(args.out))
        print(info)
        if info["modules"] != 208 or info["fc2_modules"] != 52:
            raise SystemExit(f"bad merged layout: {info}")
        print(f"ok: {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")
        return

    out = merge_loras(
        args.prod,
        args.sla,
        args.out,
        strength_prod=args.strength_prod,
        strength_sla=args.strength_sla,
    )
    info = _summary(out)
    nbytes = sum(v.numel() * v.element_size() for v in out.values())
    print(info)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB, raw {nbytes / 1e9:.2f} GB)")
    if info["modules"] != 208 or info["fc2_modules"] != 52:
        raise SystemExit(f"unexpected module layout: {info}")


if __name__ == "__main__":
    main()
