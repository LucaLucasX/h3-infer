"""MiniMax-H3 SLA (LightX2V dynamic_sparse_attn) attention patch for ComfyUI.

Matches official Turbo-SLA inference:
  sparsity_ratio=0.85 → topk_ratio=0.15, operator=sage2
  block map from LightX2V sla_util.get_block_map (mean-pool topk)
  sparse kernel via spas_sage_attn.block_sparse_sage2_attn_cuda

If deps are missing, NODE_CLASS_MAPPINGS stays empty so Comfy still boots.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

_EXP = Path(__file__).resolve().parents[2]
_VENDOR_SLA = _EXP / "vendor" / "lightx2v_sla"
if str(_VENDOR_SLA.parent) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SLA.parent))

try:
    from comfy.ldm.modules.attention import attention_pytorch, wrap_attn
except Exception as e:  # noqa: BLE001
    logging.warning("H3-SLAAttn-Exp: comfy attention import failed: %s", e)
else:
    try:
        from lightx2v_sla import get_block_map, get_cuda_arch
        from spas_sage_attn import block_sparse_sage2_attn_cuda

        _SLA_OK = True
    except Exception as e:  # noqa: BLE001
        logging.info("H3-SLAAttn-Exp: SLA deps unavailable (%s) — node disabled", e)
        _SLA_OK = False

    if _SLA_OK:

        def _to_qkv(q, k, v, heads, skip_reshape):
            in_dtype = v.dtype
            if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
                q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            if skip_reshape:
                b, _, _, dim_head = q.shape
                tensor_layout = "HND"
            else:
                b, _, dim_head = q.shape
                dim_head //= heads
                q, k, v = map(lambda t: t.view(b, -1, heads, dim_head), (q, k, v))
                tensor_layout = "NHD"
            return q, k, v, b, dim_head, tensor_layout, in_dtype

        def _from_out(out, b, heads, dim_head, tensor_layout, in_dtype, skip_output_reshape):
            out = out.to(in_dtype)
            if skip_output_reshape:
                if tensor_layout == "NHD":
                    out = out.transpose(1, 2)
            else:
                if tensor_layout == "HND":
                    out = out.transpose(1, 2)
                out = out.reshape(b, -1, heads * dim_head)
            return out

        def _as_hnd(q, k, v, tensor_layout: str):
            if tensor_layout == "HND":
                return q, k, v
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        def _audio_token_ranges(seq: int, kwargs: dict) -> list[tuple[int, int]]:
            to = kwargs.get("transformer_options") or {}
            segs = to.get("h3_pack_segments")
            pack_len = to.get("h3_pack_seq_len")
            if not segs or pack_len not in (None, seq):
                return []
            return [(int(a), int(b)) for a, b, kind in segs if kind in ("audio", "ref_audio")]

        def _boost_audio_blocks(block_map, seq: int, ranges: list[tuple[int, int]], blkq: int, blkk: int):
            """Keep all K for audio Q-blocks, and keep audio K-blocks for every Q."""
            if not ranges:
                return block_map
            nq, nk = block_map.shape[-2], block_map.shape[-1]
            for a, b in ranges:
                a = max(0, min(seq, a))
                b = max(0, min(seq, b))
                if b <= a:
                    continue
                q0, q1 = a // blkq, min(nq, (b + blkq - 1) // blkq)
                k0, k1 = a // blkk, min(nk, (b + blkk - 1) // blkk)
                if q1 > q0:
                    block_map[:, :, q0:q1, :] = 1
                if k1 > k0:
                    block_map[:, :, :, k0:k1] = 1
            return block_map

        def _dense_attn(q, k, v, tensor_layout: str):
            try:
                from sageattention import sageattn

                return sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=False)
            except Exception:
                if tensor_layout == "HND":
                    qn, kn, vn = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                else:
                    qn, kn, vn = q, k, v
                b, s, h, d = qn.shape
                out = attention_pytorch(
                    qn.reshape(b, s, h * d),
                    kn.reshape(b, s, h * d),
                    vn.reshape(b, s, h * d),
                    h,
                    skip_reshape=False,
                    skip_output_reshape=False,
                )
                out = out.view(b, s, h, d)
                return out.transpose(1, 2) if tensor_layout == "HND" else out

        def _sla_sage2(q, k, v, tensor_layout: str, topk_ratio: float, audio_dense: bool, kwargs: dict):
            """LightX2V dynamic_sparse_attn / operator=sage2."""
            q_h, k_h, v_h = _as_hnd(q, k, v, tensor_layout)
            q_h = q_h.contiguous()
            k_h = k_h.contiguous()
            v_h = v_h.contiguous()

            arch = get_cuda_arch(q_h.device.index)
            if arch == "sm90":
                blkq, blkk = 64, 128
            else:
                # sm80/sm86/sm120 (5090) — same as LightX2V non-sm90 branch
                blkq, blkk = 128, 64

            sparse_map, _lut, _real_topk = get_block_map(
                q_h, k_h, topk_ratio=float(topk_ratio), BLKQ=blkq, BLKK=blkk
            )
            if audio_dense:
                seq = int(q_h.shape[-2])
                ranges = _audio_token_ranges(seq, kwargs)
                sparse_map = _boost_audio_blocks(sparse_map, seq, ranges, blkq, blkk)
            # block_sparse_sage2 expects 0/1 / bool mask_id
            mask_id = sparse_map.bool() if sparse_map.dtype != torch.bool else sparse_map
            out = block_sparse_sage2_attn_cuda(
                q_h, k_h, v_h, mask_id=mask_id, tensor_layout="HND",
            )
            return out if tensor_layout == "HND" else out.transpose(1, 2)

        def _make_sla_attn(
            sparsity_ratio: float,
            dense_first_steps: int = 0,
            num_layers: int = 50,
            audio_dense: bool = True,
        ):
            state = {"n": 0}
            dense_calls = max(0, int(dense_first_steps)) * max(1, int(num_layers))
            topk_ratio = max(1e-4, min(1.0, 1.0 - float(sparsity_ratio)))

            @wrap_attn
            def attention_sla(
                q,
                k,
                v,
                heads,
                mask=None,
                attn_precision=None,
                skip_reshape=False,
                skip_output_reshape=False,
                **kwargs,
            ):
                if mask is not None:
                    return attention_pytorch(
                        q, k, v, heads, mask=mask,
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                        **kwargs,
                    )
                q, k, v, b, dim_head, tensor_layout, in_dtype = _to_qkv(
                    q, k, v, heads, skip_reshape
                )
                use_dense = state["n"] < dense_calls
                state["n"] += 1
                if use_dense:
                    out = _dense_attn(q, k, v, tensor_layout)
                else:
                    out = _sla_sage2(
                        q, k, v, tensor_layout, topk_ratio, bool(audio_dense), kwargs
                    )
                return _from_out(
                    out, b, heads, dim_head, tensor_layout, in_dtype, skip_output_reshape
                )

            return attention_sla

        class MiniMaxH3SLAAttnPatchExp:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "model": ("MODEL",),
                        "sparsity_ratio": (
                            "FLOAT",
                            {
                                "default": 0.85,
                                "min": 0.0,
                                "max": 0.95,
                                "step": 0.05,
                                "tooltip": "LightX2V SLA sparsity; 0.85 → keep topk=0.15 blocks",
                            },
                        ),
                        "dense_first_steps": (
                            "INT",
                            {
                                "default": 0,
                                "min": 0,
                                "max": 8,
                                "tooltip": "First N denoise steps use dense Sage2",
                            },
                        ),
                        "num_layers": (
                            "INT",
                            {
                                "default": 50,
                                "min": 1,
                                "max": 128,
                                "tooltip": "DiT self-attn count per forward",
                            },
                        ),
                        "audio_dense": (
                            "BOOLEAN",
                            {
                                "default": True,
                                "tooltip": "Keep audio token blocks dense; only video stays sparse",
                            },
                        ),
                    }
                }

            RETURN_TYPES = ("MODEL",)
            FUNCTION = "patch"
            CATEGORY = "experimental/h3_sla"
            DESCRIPTION = (
                "EXPERIMENT ONLY. LightX2V dynamic_sparse_attn (SLA block map + sage2). "
                "Official Turbo-SLA uses sparsity_ratio=0.85."
            )

            def patch(self, model, sparsity_ratio, dense_first_steps=0, num_layers=50, audio_dense=True):
                model_clone = model.clone()
                attn = _make_sla_attn(
                    float(sparsity_ratio),
                    int(dense_first_steps),
                    int(num_layers),
                    bool(audio_dense),
                )

                def attention_override(func, *args, **kwargs):
                    return attn(*args, **kwargs)

                model_clone.model_options["transformer_options"][
                    "optimized_attention_override"
                ] = attention_override
                logging.info(
                    "H3-SLAAttn-Exp: SLA sparsity=%.3f topk=%.3f dense_first_steps=%d num_layers=%d audio_dense=%s",
                    float(sparsity_ratio),
                    1.0 - float(sparsity_ratio),
                    int(dense_first_steps),
                    int(num_layers),
                    bool(audio_dense),
                )
                return (model_clone,)

        NODE_CLASS_MAPPINGS = {
            "MiniMaxH3SLAAttnPatchExp": MiniMaxH3SLAAttnPatchExp,
        }
        NODE_DISPLAY_NAME_MAPPINGS = {
            "MiniMaxH3SLAAttnPatchExp": "MiniMax H3 SLA Attn Patch (EXP)",
        }
