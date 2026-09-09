"""Insert MiniMaxH3LosslessMemPatch after the production Turbo LoRA node.

The node is opt-in per /prompt: omit it and the production graph is unchanged.
Use should_inject_lossless_mem() at request-build time to turn it on only when
the packed sequence is past the production OOM cliff.
"""
from __future__ import annotations


# Production RTX 5090 32GB cliffs (12s @ 1376x768, sage2+sparge+4-step turbo):
#   no ref video, 9 images: OK
#   1 video <= 8s, 0 images: OK (peak ~31.8GB)
#   1 video >= 10s, 0 images: OOM before sampling
#   9 images + 1 video <= 5s: OK
#   9 images + 1 video >= 8s: OOM after TE, before sampling
def should_inject_lossless_mem(
    *,
    output_seconds: float,
    n_images: int = 0,
    video_seconds: list[float] | None = None,
) -> bool:
    """True when this request would OOM on the production graph.

    No reference video → always False (image/audio-only 12s does not OOM).
    Conservative on multi-video: production never cleared 2+ long refs at 12s.
    """
    videos = [float(v) for v in (video_seconds or []) if float(v) > 0]
    if not videos:
        return False
    n_vid = len(videos)
    longest = max(videos)
    n_images = int(n_images)
    # 12s-class output is the tight regime.
    if output_seconds >= 10:
        if longest >= 10:
            return True
        if n_images >= 9 and longest >= 8:
            return True
        if n_vid >= 2:
            return True
        return False
    # 5s output is safer; only stack many long refs.
    if n_vid >= 3 and longest >= 5:
        return True
    if n_vid >= 2 and n_images >= 9 and longest >= 8:
        return True
    return False


def inject_lossless_mem(
    graph: dict,
    chunk_rows: int = 4096,
    head_chunks: int = 4,
    mode: str = "step1",
) -> dict:
    """Rewire MODEL consumers of the last production patch onto node 22.

    Production turbo graph uses node 21 (MiniMaxH3TurboLoRA). If turbo is off,
    the attention patch is node 5. Never mutates node 1 (UNETLoader).
    """
    src = "21" if "21" in graph else ("5" if "5" in graph else "1")
    if src == "1":
        raise ValueError("refusing to patch UNETLoader directly; expected node 5 or 21")
    graph = {k: v for k, v in graph.items()}
    graph["22"] = {
        "class_type": "MiniMaxH3LosslessMemPatch",
        "inputs": {
            "model": [src, 0],
            "mode": str(mode),
            "chunk_rows": int(chunk_rows),
            "head_chunks": int(head_chunks),
        },
    }
    for nid, nd in graph.items():
        if nid == "22":
            continue
        inputs = nd.get("inputs") or {}
        for key, val in list(inputs.items()):
            if val == [src, 0]:
                inputs[key] = ["22", 0]
    return graph


def inject_kj_lowvram(
    graph: dict,
    head_chunks: int = 4,
    ffn_chunks: int = 2,
    seq_threshold: int = 4096,
) -> dict:
    """Insert U06 MiniMaxLowVRAMAttention + MiniMaxChunkFeedForward after Turbo LoRA.

    Same wiring target as inject_lossless_mem (node 21, else 5). Uses nodes 23/24
    so it does not collide with L1-full node 22.
    """
    src = "21" if "21" in graph else ("5" if "5" in graph else "1")
    if src == "1":
        raise ValueError("refusing to patch UNETLoader directly; expected node 5 or 21")
    graph = {k: v for k, v in graph.items()}
    graph["23"] = {
        "class_type": "MiniMaxLowVRAMAttention",
        "inputs": {
            "model": [src, 0],
            "head_chunks": int(head_chunks),
        },
    }
    graph["24"] = {
        "class_type": "MiniMaxChunkFeedForward",
        "inputs": {
            "model": ["23", 0],
            "chunks": int(ffn_chunks),
            "seq_threshold": int(seq_threshold),
        },
    }
    for nid, nd in graph.items():
        if nid in ("23", "24"):
            continue
        inputs = nd.get("inputs") or {}
        for key, val in list(inputs.items()):
            if val == [src, 0]:
                inputs[key] = ["24", 0]
    return graph
