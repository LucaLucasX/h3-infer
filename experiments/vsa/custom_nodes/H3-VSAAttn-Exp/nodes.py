"""MiniMax-H3 VSA-style attention patch (pure sparse, no FastH3 gate).

Coarse: mean-pool tokens into Sage2-native tiles, Top-K video-to-video cubes.
Fine: spas_sage block_sparse_sage2_attn_cuda.

H3 packed sequence [text|cond|audio|video]:
  - non-video queries are dense
  - non-video keys are always selected (VSA-H3 'exempt')
  - video queries keep `keep_percent` of video key tiles

Gate / to_gate_compress is intentionally unused: current LightX2V / Agnes
weights do not carry it, so this is the paper's zero-init 'pure sparse' path.

If deps are missing, NODE_CLASS_MAPPINGS stays empty so Comfy still boots.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

try:
    from comfy.ldm.modules.attention import attention_pytorch, wrap_attn
except Exception as e:  # noqa: BLE001
    logging.warning("H3-VSAAttn-Exp: comfy attention import failed: %s", e)
else:
    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda
        from spas_sage_attn.core import get_cuda_arch_versions

        _VSA_OK = True
    except Exception as e:  # noqa: BLE001
        logging.info("H3-VSAAttn-Exp: spas_sage_attn not available (%s) — node disabled", e)
        _VSA_OK = False

    if _VSA_OK:

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

        def _prefix_video_ranges(seq: int, kwargs: dict):
            to = kwargs.get("transformer_options") or {}
            segs = to.get("h3_pack_segments")
            pack_len = to.get("h3_pack_seq_len")
            if not segs or pack_len not in (None, seq):
                return [], [(0, seq)]
            prefix, video = [], []
            for a, b, kind in segs:
                a, b = int(a), int(b)
                if kind == "video":
                    video.append((a, b))
                else:
                    prefix.append((a, b))
            if not video:
                return prefix, [(0, seq)]
            return prefix, video

        def _overlap_tiles(ranges: list[tuple[int, int]], seq: int, tile: int, n_tiles: int, device):
            mask = torch.zeros(n_tiles, dtype=torch.bool, device=device)
            for a, b in ranges:
                a = max(0, min(seq, a))
                b = max(0, min(seq, b))
                if b <= a:
                    continue
                t0 = a // tile
                t1 = min(n_tiles, (b + tile - 1) // tile)
                if t1 > t0:
                    mask[t0:t1] = True
            return mask

        def _pool_tiles(x: torch.Tensor, tile: int):
            """Mean-pool [B,H,S,D] into [B,H,n,D], ignoring pad in the last tile."""
            b, h, s, d = x.shape
            n = (s + tile - 1) // tile
            pad = n * tile - s
            if pad:
                x = F.pad(x, (0, 0, 0, pad))
            valid = torch.ones(s, device=x.device, dtype=x.dtype)
            if pad:
                valid = F.pad(valid, (0, pad))
            tiles = x.view(b, h, n, tile, d)
            w = valid.view(1, 1, n, tile, 1)
            return tiles.mul(w).sum(dim=3) / valid.view(1, 1, n, tile, 1).sum(dim=3).clamp(min=1.0), n

        def _vsa_block_map(q_h, k_h, keep_ratio: float, kwargs: dict, blkq: int, blkk: int):
            b, heads, seq, dim = q_h.shape
            qc, nq = _pool_tiles(q_h, blkq)
            kc, nk = _pool_tiles(k_h, blkk)
            scale = dim ** -0.5
            scores = torch.matmul(qc.float(), kc.float().transpose(-1, -2)) * scale

            prefix, video = _prefix_video_ranges(seq, kwargs)
            device = q_h.device
            prefix_q = _overlap_tiles(prefix, seq, blkq, nq, device)
            prefix_k = _overlap_tiles(prefix, seq, blkk, nk, device)
            video_k = _overlap_tiles(video, seq, blkk, nk, device)
            video_q = ~prefix_q
            if not bool(video_k.any()):
                video_k = torch.ones(nk, dtype=torch.bool, device=device)
                video_q = torch.ones(nq, dtype=torch.bool, device=device)

            n_vk = int(video_k.sum().item())
            k_keep = max(1, min(n_vk, int(round(n_vk * float(keep_ratio)))))

            scores_vid = scores.masked_fill(~video_k.view(1, 1, 1, nk), float("-inf"))
            topk_idx = scores_vid.topk(k_keep, dim=-1).indices
            block_map = torch.zeros(b, heads, nq, nk, dtype=torch.bool, device=device)
            block_map.scatter_(-1, topk_idx, True)
            if bool(prefix_q.any()):
                block_map[:, :, prefix_q, :] = True
            if bool(prefix_k.any()):
                block_map[:, :, :, prefix_k] = True
            if bool(video_q.any()):
                # video Q must keep the selected video K even if scatter hit -inf pads
                pass

            if not getattr(_vsa_block_map, "_logged", False):
                logging.info(
                    "H3-VSAAttn-Exp: keep=%.3f k_keep=%d/%d video_q=%d/%d prefix_q=%d "
                    "tiles q=%d k=%d blk=%dx%d seq=%d (pure sparse, no gate)",
                    float(keep_ratio),
                    k_keep,
                    n_vk,
                    int(video_q.sum().item()),
                    nq,
                    int(prefix_q.sum().item()),
                    nq,
                    nk,
                    blkq,
                    blkk,
                    seq,
                )
                _vsa_block_map._logged = True
            return block_map

        def _vsa_sparse(q, k, v, tensor_layout: str, keep_ratio: float, kwargs: dict):
            q_h, k_h, v_h = _as_hnd(q, k, v, tensor_layout)
            q_h = q_h.contiguous()
            k_h = k_h.contiguous()
            v_h = v_h.contiguous()
            arch = get_cuda_arch_versions()[q_h.device.index]
            blkq, blkk = (64, 128) if arch == "sm90" else (128, 64)
            block_map = _vsa_block_map(q_h, k_h, keep_ratio, kwargs, blkq, blkk)
            out = block_sparse_sage2_attn_cuda(
                q_h, k_h, v_h, mask_id=block_map, tensor_layout="HND",
            )
            return out if tensor_layout == "HND" else out.transpose(1, 2)

        def _make_vsa_attn(
            keep_percent: float,
            dense_first_steps: int = 0,
            num_layers: int = 50,
        ):
            state = {"n": 0}
            dense_calls = max(0, int(dense_first_steps)) * max(1, int(num_layers))
            keep_ratio = max(1e-4, min(1.0, float(keep_percent) / 100.0))

            @wrap_attn
            def attention_vsa(
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
                    out = _vsa_sparse(q, k, v, tensor_layout, keep_ratio, kwargs)
                return _from_out(
                    out, b, heads, dim_head, tensor_layout, in_dtype, skip_output_reshape
                )

            return attention_vsa

        class MiniMaxH3VSAAttnPatchExp:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "model": ("MODEL",),
                        "keep_percent": (
                            "FLOAT",
                            {
                                "default": 10.0,
                                "min": 1.0,
                                "max": 100.0,
                                "step": 1.0,
                                "tooltip": "Keep this % of video-to-video tiles. FastH3 uses 10 (90% sparse).",
                            },
                        ),
                        "dense_first_steps": (
                            "INT",
                            {
                                "default": 0,
                                "min": 0,
                                "max": 8,
                                "tooltip": "First N denoise steps use dense Sage2.",
                            },
                        ),
                        "num_layers": (
                            "INT",
                            {
                                "default": 50,
                                "min": 1,
                                "max": 128,
                            },
                        ),
                    }
                }

            RETURN_TYPES = ("MODEL",)
            FUNCTION = "patch"
            CATEGORY = "experimental/h3_vsa"
            DESCRIPTION = (
                "EXPERIMENT ONLY. VSA-H3 pure sparse (coarse Top-K + Sage2 block-sparse). "
                "No to_gate_compress. FastH3 keep_percent=10."
            )

            def patch(self, model, keep_percent, dense_first_steps=0, num_layers=50):
                model_clone = model.clone()
                attn = _make_vsa_attn(
                    float(keep_percent),
                    int(dense_first_steps),
                    int(num_layers),
                )

                def attention_override(func, *args, **kwargs):
                    return attn(*args, **kwargs)

                model_clone.model_options["transformer_options"][
                    "optimized_attention_override"
                ] = attention_override
                logging.info(
                    "H3-VSAAttn-Exp: keep_percent=%.1f dense_first_steps=%d num_layers=%d",
                    float(keep_percent),
                    int(dense_first_steps),
                    int(num_layers),
                )
                return (model_clone,)

        NODE_CLASS_MAPPINGS = {
            "MiniMaxH3VSAAttnPatchExp": MiniMaxH3VSAAttnPatchExp,
        }
        NODE_DISPLAY_NAME_MAPPINGS = {
            "MiniMaxH3VSAAttnPatchExp": "MiniMax H3 VSA Attn Patch (EXP)",
        }
