"""Token-row chunking helpers for MiniMax H3 packed sequences.

INT8 ConvRot activations are quantized per row (amax over the hidden dim), so
slicing the sequence dimension and concatenating is bit-identical to a single
call as long as a chunk never takes the kitchen m==1 specialized kernel.

``chunked_call`` writes into a preallocated output so peak VRAM is one full
result plus one chunk, not two full results (list-of-parts + torch.cat).
"""

from __future__ import annotations


def iter_row_chunks(n: int, chunk_rows: int):
    """Yield [start, end) row ranges covering ``n``.

    The last chunk is never size 1 (unless ``n==1``), because Comfy Kitchen's
    INT8 ConvRot path switches to a different m=1 kernel.
    """
    if n <= 0:
        return
    chunk_rows = max(int(chunk_rows), 1)
    start = 0
    while start < n:
        end = min(start + chunk_rows, n)
        if end < n and (n - end) < 2:
            end = n
        yield start, end
        start = end


def chunked_call(module, x, chunk_rows: int):
    """Call ``module(x[s:e])`` over row chunks into a preallocated output."""
    return chunked_apply(module, x, chunk_rows)


def chunked_apply(fn, x, chunk_rows: int):
    """Like ``chunked_call`` but ``fn`` is any callable ``x_slice -> y_slice``."""
    import torch

    if x is None or not hasattr(x, "shape") or x.dim() < 2:
        return fn(x)
    n = int(x.shape[0])
    if n <= int(chunk_rows):
        return fn(x)
    spans = list(iter_row_chunks(n, chunk_rows))
    s0, e0 = spans[0]
    first = fn(x[s0:e0])
    out = torch.empty((n, *first.shape[1:]), dtype=first.dtype, device=first.device)
    out[s0:e0].copy_(first)
    del first
    for start, end in spans[1:]:
        out[start:end].copy_(fn(x[start:end]))
    return out
