"""Opt-in lossless VRAM patch for MiniMax H3 packed-sequence activations.

step1: token-chunk MLP/QKV/out-proj (preallocated) + optional head-group Sparge.
step2: step1 MLP/linears + head-group QKV pipeline (no fused [S,3HD] buffer).

Place AFTER MiniMaxH3TurboLoRA. Production graphs that omit this node are unchanged.
"""

from __future__ import annotations

import logging

import torch

from .chunk_util import chunked_apply, chunked_call, iter_row_chunks

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

_LOG = logging.getLogger("ComfyUI-H3-LosslessMem")
_HEAD_CHUNK_LOGGED = False


def _make_mlp_forward(mlp, chunk_rows: int):
    orig = mlp.forward

    def forward(x):
        return chunked_call(orig, x, chunk_rows)

    return forward


def _make_linear_forward(linear, chunk_rows: int):
    orig = linear.forward

    def forward(x, *args, **kwargs):
        if not torch.is_tensor(x) or x.dim() < 2:
            return orig(x, *args, **kwargs)
        if args or kwargs:
            return chunked_apply(lambda sl: orig(sl, *args, **kwargs), x, chunk_rows)
        return chunked_call(orig, x, chunk_rows)

    return forward


def _sparge_v_workspace_bytes(seq: int, heads: int, head_dim: int) -> int:
    padded = (int(seq) + 127) // 128 * 128
    return int(heads) * int(head_dim) * padded * 2


def _rope_token_slice(rope_freqs, start: int, end: int):
    if rope_freqs is None:
        return None
    if rope_freqs.dim() >= 2 and rope_freqs.shape[1] >= end:
        return rope_freqs[:, start:end]
    return rope_freqs[start:end]


def _optimized_attention_head_chunks(q, k, v, transformer_options, chunk_heads: int):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    h_total = int(q.shape[1])
    dim = int(q.shape[-1])
    chunk_heads = max(1, min(int(chunk_heads), h_total))
    out = None
    for h0 in range(0, h_total, chunk_heads):
        h1 = min(h_total, h0 + chunk_heads)
        n_h = h1 - h0
        out_h = optimized_attention(
            AttentionTensorContainer(q[:, h0:h1].contiguous()),
            AttentionTensorContainer(k[:, h0:h1].contiguous()),
            AttentionTensorContainer(v[:, h0:h1].contiguous()),
            n_h,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        if out is None:
            b, s, _ = out_h.shape
            out = torch.empty((b, s, h_total * dim), dtype=out_h.dtype, device=out_h.device)
        out[:, :, h0 * dim: h1 * dim].copy_(out_h)
        del out_h
    return out


def _project_qkv_head_group(attn, x, chunk_rows: int, h0: int, h1: int, rope_freqs, qw, kw, rot):
    import comfy.model_management
    import comfy.quant_ops

    s = int(x.shape[0])
    heads, dim = attn.heads, attn.head_dim
    inner = heads * dim
    n_h = h1 - h0
    q_buf = torch.empty((s, n_h, dim), dtype=x.dtype, device=x.device)
    k_buf = torch.empty_like(q_buf)
    v_buf = torch.empty_like(q_buf)
    for start, end in iter_row_chunks(s, chunk_rows):
        y = attn.qkv_proj(x[start:end])
        n = end - start
        q, k, v = y.split(inner, dim=-1)
        del y
        q = q.reshape(n, heads, dim)[:, h0:h1]
        k = k.reshape(n, heads, dim)[:, h0:h1]
        v = v.reshape(n, heads, dim)[:, h0:h1]
        if rope_freqs is not None:
            q = q.reshape(1, n, n_h, dim)
            k = k.reshape(1, n, n_h, dim)
            rf = _rope_token_slice(rope_freqs, start, end)
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rf, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q, k = q[0], k[0]
        else:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q_buf[start:end].copy_(q)
        k_buf[start:end].copy_(k)
        v_buf[start:end].copy_(v)
        del q, k, v
    return q_buf, k_buf, v_buf


def _sparge_one_group(q, k, v, n_h, transformer_options):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    q_hnd = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    k_hnd = k.permute(1, 0, 2).unsqueeze(0).contiguous()
    v_hnd = v.permute(1, 0, 2).unsqueeze(0).contiguous()
    del q, k, v
    out = optimized_attention(
        AttentionTensorContainer(q_hnd),
        AttentionTensorContainer(k_hnd),
        AttentionTensorContainer(v_hnd),
        n_h,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    del q_hnd, k_hnd, v_hnd
    return out.squeeze(0)


def _make_attn_forward_step1(attn, chunk_rows: int, head_chunks: int):
    import comfy.model_management
    import comfy.quant_ops

    def forward(x, rope_freqs=None, transformer_options={}):
        global _HEAD_CHUNK_LOGGED
        s = x.shape[0]
        qkv = chunked_call(attn.qkv_proj, x, chunk_rows)
        q, k, v = qkv.split(attn.heads * attn.head_dim, dim=-1)
        v = v.view(s, attn.heads, attn.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, attn.heads, attn.head_dim)
            k = k.view(1, s, attn.heads, attn.head_dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
            k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
        del qkv
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        if x.is_cuda:
            torch.cuda.empty_cache()
        if not _HEAD_CHUNK_LOGGED:
            _HEAD_CHUNK_LOGGED = True
            need_full = _sparge_v_workspace_bytes(s, attn.heads, attn.head_dim)
            msg = (
                f"[H3-LosslessMem] step1 seq={s} heads={attn.heads} "
                f"chunk={head_chunks} full_V~{need_full / 1024 ** 2:.0f}MB"
            )
            _LOG.warning(msg)
            print(msg, flush=True)
        out = _optimized_attention_head_chunks(q, k, v, transformer_options, head_chunks)
        return chunked_call(attn.out_proj, out.squeeze(0), chunk_rows)

    return forward


def _make_attn_forward_step2(attn, chunk_rows: int, head_chunks: int):
    import comfy.model_management

    def forward(x, rope_freqs=None, transformer_options={}):
        global _HEAD_CHUNK_LOGGED
        s = int(x.shape[0])
        heads, dim = attn.heads, attn.head_dim
        n_chunk = max(1, min(int(head_chunks), heads))
        qw = kw = rot = None
        if rope_freqs is not None:
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
        if not _HEAD_CHUNK_LOGGED:
            _HEAD_CHUNK_LOGGED = True
            need_full = _sparge_v_workspace_bytes(s, heads, dim)
            msg = (
                f"[H3-LosslessMem] step2 seq={s} heads={heads} group={n_chunk} "
                f"live_QKV={n_chunk}/{heads} full_V~{need_full / 1024 ** 2:.0f}MB"
            )
            _LOG.warning(msg)
            print(msg, flush=True)
        attn_buf = torch.empty((s, heads * dim), dtype=x.dtype, device=x.device)
        for h0 in range(0, heads, n_chunk):
            h1 = min(heads, h0 + n_chunk)
            n_h = h1 - h0
            q, k, v = _project_qkv_head_group(
                attn, x, chunk_rows, h0, h1, rope_freqs, qw, kw, rot)
            out_h = _sparge_one_group(q, k, v, n_h, transformer_options)
            attn_buf[:, h0 * dim: h1 * dim].copy_(out_h)
            del out_h, q, k, v
            if x.is_cuda:
                torch.cuda.empty_cache()
        return chunked_call(attn.out_proj, attn_buf, chunk_rows)

    return forward


class MiniMaxH3LosslessMemPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["step1", "step2"], {"default": "step1"}),
                "chunk_rows": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 131072, "step": 256},
                ),
                "head_chunks": (
                    "INT",
                    {"default": 4, "min": 1, "max": 64, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "experimental/h3_vram"
    DESCRIPTION = (
        "EXPERIMENT. step1=prealloc token chunks + head-group Sparge; "
        "step2=also split fused QKV by head group. After Turbo LoRA."
    )

    def patch(self, model, mode="step1", chunk_rows=4096, head_chunks=4):
        global _HEAD_CHUNK_LOGGED
        _HEAD_CHUNK_LOGGED = False
        mode = str(mode)
        chunk_rows = int(chunk_rows)
        head_chunks = int(head_chunks)
        make_attn = _make_attn_forward_step2 if mode == "step2" else _make_attn_forward_step1
        new_model = model.clone()
        dm = new_model.get_model_object("diffusion_model")
        n_mlp = n_attn = n_lin = 0

        blocks = list(getattr(dm, "blocks", []))
        refiner = getattr(dm, "token_refiner", None)
        refiner_blocks = list(getattr(refiner, "blocks", [])) if refiner is not None else []

        for block in blocks + refiner_blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                path = _module_path(dm, mlp)
                if path:
                    new_model.add_object_patch(path + ".forward", _make_mlp_forward(mlp, chunk_rows))
                    n_mlp += 1
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "qkv_proj"):
                path = _module_path(dm, attn)
                if path:
                    new_model.add_object_patch(
                        path + ".forward",
                        make_attn(attn, chunk_rows, head_chunks),
                    )
                    n_attn += 1

        for name in ("video_patch_proj", "audio_patch_proj", "condition_proj"):
            lin = getattr(dm, name, None)
            if lin is None:
                continue
            path = _module_path(dm, lin)
            if path:
                new_model.add_object_patch(path + ".forward", _make_linear_forward(lin, chunk_rows))
                n_lin += 1

        print(
            f"[H3-LosslessMem] {mode} chunk_rows={chunk_rows} head_chunks={head_chunks} "
            f"mlp={n_mlp} attn={n_attn} linears={n_lin}",
            flush=True,
        )
        if n_mlp == 0:
            raise RuntimeError(
                "MiniMaxH3LosslessMemPatch: no DiT MLP modules found — this is not a MiniMax H3 model."
            )
        return (new_model,)


def _module_path(root, target) -> str | None:
    for name, mod in root.named_modules():
        if mod is target:
            return "diffusion_model." + name if name else "diffusion_model"
    return None


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LosslessMemPatch": MiniMaxH3LosslessMemPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LosslessMemPatch": "MiniMax H3 Lossless Mem Patch (chunked MLP/QKV)",
}
