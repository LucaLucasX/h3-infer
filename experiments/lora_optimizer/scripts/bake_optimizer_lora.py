#!/usr/bin/env python3
"""Bake prod+sla LoRAs via LoRA Optimizer analysis + SaveMergedLoRA, then patch H3 fc2.

Workflow:
  1. ComfyUI: UNETLoader -> LoRAStack(prod+sla) -> LoRAOptimizer -> SaveMergedLoRA
  2. Post: inject 52 mlp.fc2 layers (Optimizer cannot map INT8 fc2 on H3 pruned base)
  3. Load baked file with MiniMaxH3TurboLoRA (not LoRA Optimizer at inference time)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_client import ComfyUIError, health, submit_prompt, wait_prompt
from h3_graph import node
from loras import LORA_OPTIMIZER_BAKED, LORA_PROD_T2V, LORA_SLA_T2V
from h3_lora_merge_utils import parse_source_strengths
from finalize_h3_baked_lora import finalize_optimizer_baked_for_h3

LORAS_SAVE_DIR = EXP / "models" / "loras"
BAKED_PATH = LORAS_SAVE_DIR / "lightx2v_fl2v" / Path(LORA_OPTIMIZER_BAKED).name
META_PATH = EXP / "logs" / "optimizer_bake.json"


def build_bake_graph(
    *,
    prod_lora: str,
    sla_lora: str,
    save_folder: str,
    filename: str,
    optimization_mode: str = "additive",
    preserve: bool = True,
    save_rank: int = 0,
) -> dict[str, dict]:
    g: dict[str, dict] = {}
    g["1"] = node("UNETLoader", {
        "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "weight_dtype": "default",
    })
    g["10"] = node("LoRAStack", {
        "lora_name": prod_lora,
        "strength": 1.0,
        "conflict_mode": "all",
        "key_filter": "all",
        "preserve": preserve,
    })
    g["11"] = node("LoRAStack", {
        "lora_name": sla_lora,
        "strength": 1.0,
        "conflict_mode": "all",
        "key_filter": "all",
        "preserve": preserve,
        "lora_stack": ["10", 0],
    })
    g["20"] = node("LoRAOptimizer", {
        "model": ["1", 0],
        "lora_stack": ["11", 0],
        "output_strength": 1.0,
        "clip_strength_multiplier": 1.0,
        "auto_strength": "enabled",
        "auto_strength_floor": -1.0,
        "free_vram_between_passes": "disabled",
        "vram_budget": 0.0,
        "optimization_mode": optimization_mode,
        "cache_patches": "disabled",
        "patch_compression": "smart",
        "svd_device": "gpu",
        "normalize_keys": "enabled",
        "sparsification": "disabled",
        "sparsification_density": 0.7,
        "dare_dampening": 0.0,
        "merge_refinement": "none",
        "strategy_set": "full",
        "architecture_preset": "auto",
        "merge_strategy_override": "",
        "settings_source": "manual",
    })
    g["30"] = node("SaveMergedLoRA", {
        "lora_data": ["20", 4],
        "save_folder": save_folder,
        "filename": filename,
        "save_rank": int(save_rank),
        "bake_strength": True,
        "prompt": "",
        "description": "LoRA Optimizer bake: prod+sla additive, fc2 patched post-save",
    })
    return g


def _parse_report_strengths(report: str) -> tuple[float | None, float | None]:
    """Extract auto-strength scale factors from optimizer analysis report."""
    vals = [float(x) for x in re.findall(r"model 1\.0 -> ([\d.]+)", report)]
    if len(vals) >= 2:
        return vals[0], vals[1]
    if len(vals) == 1:
        return vals[0], vals[0]
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake prod+sla via LoRA Optimizer + SaveMergedLoRA.")
    ap.add_argument("--comfy", default="http://127.0.0.1:8195")
    ap.add_argument("--prod-lora", default=LORA_PROD_T2V)
    ap.add_argument("--sla-lora", default=LORA_SLA_T2V)
    ap.add_argument("--out-filename", default="lightx2v_fl2v/minimax_h3_fl2v_turbo_4step_optimizer_baked_h3keys")
    ap.add_argument("--optimization-mode", default="additive", choices=["additive", "per_prefix", "global"])
    ap.add_argument("--no-preserve", action="store_true")
    ap.add_argument("--save-rank", type=int, default=0, help="0=auto (sum of input ranks)")
    ap.add_argument("--skip-finalize", action="store_true",
                    help="Skip h3keys finalize (debug only)")
    ap.add_argument("--skip-bake", action="store_true", help="Only run fc2 patch on existing baked file")
    args = ap.parse_args()

    save_folder = str(LORAS_SAVE_DIR.resolve())
    LORAS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (LORAS_SAVE_DIR / "lightx2v_fl2v").mkdir(parents=True, exist_ok=True)
    baked_path = LORAS_SAVE_DIR / f"{args.out_filename}.safetensors"

    report_text = ""
    if not args.skip_bake:
        health(args.comfy)
        graph = build_bake_graph(
            prod_lora=args.prod_lora,
            sla_lora=args.sla_lora,
            save_folder=save_folder,
            filename=args.out_filename,
            optimization_mode=args.optimization_mode,
            preserve=not args.no_preserve,
            save_rank=args.save_rank,
        )
        # Bust ComfyUI classic cache so SaveMergedLoRA always re-runs.
        graph["30"]["inputs"]["description"] = f"bake_ts={time.time():.3f}"
        print(f"baking via LoRA Optimizer on {args.comfy} ...", flush=True)
        t0 = time.perf_counter()
        pid = submit_prompt(args.comfy, graph)
        print(f"prompt_id={pid}", flush=True)
        hist = wait_prompt(args.comfy, pid)
        elapsed = round(time.perf_counter() - t0, 1)
        st = (hist.get("status") or {}).get("status_str")
        if st != "success":
            raise ComfyUIError(f"bake failed: status={st} hist={json.dumps(hist, ensure_ascii=False)[:2000]}")
        # Optimizer report is output slot 2 on node 20
        outputs = hist.get("outputs") or {}
        for out in outputs.values():
            if isinstance(out, dict) and out.get("text"):
                report_text = "\n".join(out["text"])
                break
        if not baked_path.is_file():
            raise ComfyUIError(f"bake finished but file missing: {baked_path}")
        print(f"baked in {elapsed}s -> {baked_path}", flush=True)
    elif not baked_path.is_file():
        raise SystemExit(f"--skip-bake but missing {baked_path}")

    finalize_info = {}
    if not args.skip_finalize:
        s_prod, s_sla = _parse_report_strengths(report_text)
        if s_prod is None:
            s_prod = s_sla = 0.5886
        finalize_info = finalize_optimizer_baked_for_h3(
            baked_path,
            strength_prod=s_prod,
            strength_sla=s_sla,
        )
        print(f"finalize: {finalize_info}", flush=True)
        if finalize_info["modules"] != 208 or finalize_info["fc2_modules"] != 52:
            raise SystemExit(f"unexpected layout after finalize: {finalize_info}")

    payload = {
        "baked_path": str(baked_path),
        "comfy_lora_name": args.out_filename + ".safetensors",
        "prod_lora": args.prod_lora,
        "sla_lora": args.sla_lora,
        "optimization_mode": args.optimization_mode,
        "preserve": not args.no_preserve,
        "finalize": finalize_info,
        "report_excerpt": report_text[:4000] if report_text else None,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {META_PATH}", flush=True)
    print(f"load in TurboLoRA: {args.out_filename}.safetensors strength=1.0", flush=True)


if __name__ == "__main__":
    main()
