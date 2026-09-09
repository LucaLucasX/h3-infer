"""Experimental MiniMax-H3 SpargeAttn / LoSA frozen-mask attention patch.

Safe defaults: if spas_sage_attn is missing, NODE_CLASS_MAPPINGS stays empty so
daily ComfyUI (:8188) still boots.
"""

from __future__ import annotations

import logging

import torch

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

try:
    from comfy.ldm.modules.attention import attention_pytorch, wrap_attn
except Exception as e:  # noqa: BLE001
    logging.warning("H3-SpargeAttn-Exp: comfy attention import failed: %s", e)
else:
    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda, spas_sage2_attn_meansim_topk_cuda
        from spas_sage_attn.core import get_cuda_arch_versions
        from spas_sage_attn.utils import get_block_map_meansim, get_block_map_meansim_fuse_quant

        _SPAS_OK = True
    except Exception as e:  # noqa: BLE001
        logging.info("H3-SpargeAttn-Exp: spas_sage_attn not available (%s) — node disabled", e)
        _SPAS_OK = False

    if _SPAS_OK:

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

        def _as_hnd(q, k, v, tensor_layout):
            if tensor_layout == "HND":
                return q, k, v
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        def _dense_attn(q, k, v, tensor_layout: str):
            """Early-step dense path: SageAttention if available, else pytorch SDPA."""
            try:
                from sageattention import sageattn

                return sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=False)
            except Exception:
                # Fall back via comfy pytorch attention on NHD [B,S,H,D]
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

        def _audio_token_ranges(seq: int, kwargs: dict) -> list[tuple[int, int]]:
            to = kwargs.get("transformer_options") or {}
            segs = to.get("h3_pack_segments")
            pack_len = to.get("h3_pack_seq_len")
            if not segs or pack_len not in (None, seq):
                return []
            return [(int(a), int(b)) for a, b, kind in segs if kind in ("audio", "ref_audio")]

        def _boost_audio_blocks(block_map, seq: int, ranges: list[tuple[int, int]], blkq: int, blkk: int):
            """Keep all K for audio Q-blocks, and keep audio K-blocks for every Q (video stays topk)."""
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
                    block_map[:, :, q0:q1, :] = True
                if k1 > k0:
                    block_map[:, :, :, k0:k1] = True
            return block_map

        def _sparge_audio_boosted(q, k, v, tensor_layout: str, topk: float, kwargs: dict):
            q_h, k_h, v_h = _as_hnd(q, k, v, tensor_layout)
            seq = int(q_h.shape[-2])
            ranges = _audio_token_ranges(seq, kwargs)
            if not ranges:
                return spas_sage2_attn_meansim_topk_cuda(
                    q, k, v, topk=float(topk), is_causal=False, tensor_layout=tensor_layout,
                )
            arch = get_cuda_arch_versions()[q_h.device.index]
            blkq, blkk = (64, 128) if arch == "sm90" else (128, 64)
            km = k_h.mean(dim=-2, keepdim=True)
            block_map, _, _, _, _ = get_block_map_meansim_fuse_quant(
                q_h.contiguous(), k_h.contiguous(), km,
                is_causal=False, simthreshd1=-0.1, cdfthreshd=None, topk=float(topk),
                return_lut=False, attention_sink=False, BLKQ=blkq, BLKK=blkk,
            )
            block_map = _boost_audio_blocks(block_map, seq, ranges, blkq, blkk)
            out = block_sparse_sage2_attn_cuda(
                q_h, k_h, v_h, mask_id=block_map, tensor_layout="HND",
            )
            return out if tensor_layout == "HND" else out.transpose(1, 2)

        def _make_sparge_attn(topk: float, dense_first_steps: int = 0, num_layers: int = 50, audio_dense: bool = True):
            state = {"n": 0}
            dense_calls = max(0, int(dense_first_steps)) * max(1, int(num_layers))

            @wrap_attn
            def attention_sparge(
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
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs,
                    )
                q, k, v, b, dim_head, tensor_layout, in_dtype = _to_qkv(q, k, v, heads, skip_reshape)
                use_dense = state["n"] < dense_calls
                state["n"] += 1
                if use_dense:
                    out = _dense_attn(q, k, v, tensor_layout)
                elif audio_dense:
                    out = _sparge_audio_boosted(q, k, v, tensor_layout, float(topk), kwargs)
                else:
                    out = spas_sage2_attn_meansim_topk_cuda(
                        q, k, v, topk=float(topk), is_causal=False, tensor_layout=tensor_layout,
                    )
                return _from_out(out, b, heads, dim_head, tensor_layout, in_dtype, skip_output_reshape)

            return attention_sparge

        def _make_losa_attn(mass_thresh: float, profile_steps: int, num_layers: int):
            """LoSA-style: first profile_steps forwards build per-layer block masks (cdf≈mass), then freeze.

            Mask shape matches SpargeAttn block_sparse API:
            (B, H, ceil(S/128), ceil(S/64)) bool/0-1.
            """
            state = {
                "n": 0,
                "masks": {},  # layer_idx -> mask_id on device
                "shape": None,
                "frozen": False,
                "sparsity_sum": 0.0,
                "sparsity_n": 0,
            }
            profile_calls = max(1, int(profile_steps)) * max(1, int(num_layers))
            mass = float(mass_thresh)
            n_layers = max(1, int(num_layers))

            @wrap_attn
            def attention_losa(
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
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs,
                    )
                q, k, v, b, dim_head, tensor_layout, in_dtype = _to_qkv(q, k, v, heads, skip_reshape)
                q_h, k_h, v_h = _as_hnd(q, k, v, tensor_layout)
                seq = q_h.shape[-2]
                shape_key = (b, q_h.shape[1], seq, dim_head, str(q_h.device))
                layer_idx = state["n"] % n_layers
                state["n"] += 1

                if state["shape"] != shape_key:
                    state["shape"] = shape_key
                    state["masks"].clear()
                    state["frozen"] = False
                    # re-profile from this call
                    state["n"] = 1
                    layer_idx = 0

                if not state["frozen"] or layer_idx not in state["masks"]:
                    # Build retained-mass block map (cdfthreshd ≈ LoSA mass threshold).
                    block_map = get_block_map_meansim(
                        q_h.contiguous(),
                        k_h.contiguous(),
                        is_causal=False,
                        BLKQ=128,
                        BLKK=64,
                        simthreshd1=-0.1,
                        cdfthreshd=mass,
                        topk=None,
                        return_lut=False,
                        attention_sink=False,
                    )
                    # Sparge block_sparse expects 0/1 map; bool is fine via triton path.
                    state["masks"][layer_idx] = block_map.detach()
                    kept = float(block_map.float().mean().item())
                    state["sparsity_sum"] += 1.0 - kept
                    state["sparsity_n"] += 1
                    if state["n"] >= profile_calls and len(state["masks"]) >= min(n_layers, state["n"]):
                        state["frozen"] = True
                        avg_sp = state["sparsity_sum"] / max(1, state["sparsity_n"])
                        logging.info(
                            "H3-SpargeAttn-Exp: LoSA masks FROZEN after %d calls "
                            "(layers=%d mass=%.3f avg_block_sparsity=%.3f)",
                            state["n"],
                            len(state["masks"]),
                            mass,
                            avg_sp,
                        )

                mask_id = state["masks"][layer_idx]
                out = block_sparse_sage2_attn_cuda(
                    q_h, k_h, v_h, mask_id=mask_id, tensor_layout="HND",
                )
                # block_sparse returns HND
                if tensor_layout == "NHD":
                    out = out.transpose(1, 2)
                return _from_out(out, b, heads, dim_head, tensor_layout, in_dtype, skip_output_reshape)

            return attention_losa

        class MiniMaxH3SpargeAttnPatchExp:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "model": ("MODEL",),
                        "topk": (
                            "FLOAT",
                            {
                                "default": 0.5,
                                "min": 0.05,
                                "max": 1.0,
                                "step": 0.05,
                                "tooltip": "Higher = denser/more accurate; lower = sparser/faster",
                            },
                        ),
                        "dense_first_steps": (
                            "INT",
                            {
                                "default": 0,
                                "min": 0,
                                "max": 8,
                                "tooltip": "First N denoise steps use dense Sage (fix early audio boom)",
                            },
                        ),
                        "num_layers": (
                            "INT",
                            {
                                "default": 50,
                                "min": 1,
                                "max": 128,
                                "tooltip": "DiT self-attn count per forward (for dense_first_steps)",
                            },
                        ),
                        "audio_dense": (
                            "BOOLEAN",
                            {
                                "default": True,
                                "tooltip": "Keep audio Q/K blocks dense; video tokens stay at topk",
                            },
                        ),
                    }
                }

            RETURN_TYPES = ("MODEL",)
            FUNCTION = "patch"
            CATEGORY = "experimental/h3_sparge"
            DESCRIPTION = (
                "EXPERIMENT ONLY. SpargeAttn topk sparse attention. "
                "audio_dense keeps audio token blocks (tiny fraction of seq) fully dense. "
                "Optional dense_first_steps uses Sage for early denoise steps."
            )

            def patch(self, model, topk, dense_first_steps=0, num_layers=50, audio_dense=True):
                model_clone = model.clone()
                attn = _make_sparge_attn(topk, dense_first_steps, num_layers, audio_dense=bool(audio_dense))

                def attention_override(func, *args, **kwargs):
                    return attn(*args, **kwargs)

                model_clone.model_options["transformer_options"]["optimized_attention_override"] = (
                    attention_override
                )
                logging.info(
                    "H3-SpargeAttn-Exp: SpargeAttn topk=%.3f dense_first_steps=%d num_layers=%d audio_dense=%s",
                    float(topk),
                    int(dense_first_steps),
                    int(num_layers),
                    bool(audio_dense),
                )
                return (model_clone,)

        class MiniMaxH3LoSAApproxPatchExp:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "model": ("MODEL",),
                        "mass_thresh": (
                            "FLOAT",
                            {
                                "default": 0.99,
                                "min": 0.80,
                                "max": 0.99,
                                "step": 0.01,
                                "tooltip": "LoSA retained-mass CDF threshold for frozen block map",
                            },
                        ),
                        "profile_steps": (
                            "INT",
                            {
                                "default": 1,
                                "min": 1,
                                "max": 4,
                                "tooltip": "How many denoise forwards to build masks before freeze",
                            },
                        ),
                        "num_layers": (
                            "INT",
                            {
                                "default": 50,
                                "min": 1,
                                "max": 128,
                                "tooltip": "H3 DiT self-attn count per forward (default 50)",
                            },
                        ),
                    }
                }

            RETURN_TYPES = ("MODEL",)
            FUNCTION = "patch"
            CATEGORY = "experimental/h3_sparge"
            DESCRIPTION = (
                "EXPERIMENT ONLY. LoSA-style: profile per-layer block maps (cdf≈mass), "
                "freeze, then block_sparse_sage2_attn_cuda."
            )

            def patch(self, model, mass_thresh, profile_steps, num_layers=50):
                model_clone = model.clone()
                attn = _make_losa_attn(mass_thresh, profile_steps, num_layers)

                def attention_override(func, *args, **kwargs):
                    return attn(*args, **kwargs)

                model_clone.model_options["transformer_options"]["optimized_attention_override"] = (
                    attention_override
                )
                logging.info(
                    "H3-SpargeAttn-Exp: LoSA frozen-mask mass=%.3f profile_steps=%d num_layers=%d",
                    float(mass_thresh),
                    int(profile_steps),
                    int(num_layers),
                )
                return (model_clone,)

        NODE_CLASS_MAPPINGS = {
            "MiniMaxH3SpargeAttnPatchExp": MiniMaxH3SpargeAttnPatchExp,
            "MiniMaxH3LoSAApproxPatchExp": MiniMaxH3LoSAApproxPatchExp,
        }
        NODE_DISPLAY_NAME_MAPPINGS = {
            "MiniMaxH3SpargeAttnPatchExp": "MiniMax H3 SpargeAttn Patch (EXP)",
            "MiniMaxH3LoSAApproxPatchExp": "MiniMax H3 LoSA Frozen-Mask Patch (EXP)",
        }
