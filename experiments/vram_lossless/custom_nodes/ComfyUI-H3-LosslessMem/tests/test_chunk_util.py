#!/usr/bin/env python3
"""Local tests for row-chunk coverage and dense Linear equality."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from chunk_util import chunked_apply, chunked_call, iter_row_chunks  # noqa: E402


def test_iter_never_leaves_tail_of_one():
    for n in (1, 2, 3, 4095, 4096, 4097, 4098, 100000):
        spans = list(iter_row_chunks(n, 4096))
        covered = []
        for s, e in spans:
            assert e > s
            if n > 1:
                assert (e - s) != 1 or n == 1
            covered.extend(range(s, e))
        assert covered == list(range(n)), (n, spans)


def test_chunked_linear_matches_full():
    torch.manual_seed(0)
    lin = torch.nn.Linear(32, 64, bias=True)
    x = torch.randn(5000, 32)
    full = lin(x)
    chunked = chunked_call(lin, x, 512)
    torch.testing.assert_close(chunked, full, rtol=0, atol=0)


def test_chunked_apply_prealloc_matches_full():
    torch.manual_seed(1)

    def fn(x):
        return x @ torch.ones(x.shape[1], 48)

    x = torch.randn(3000, 16)
    torch.testing.assert_close(chunked_apply(fn, x, 256), fn(x), rtol=0, atol=0)


if __name__ == "__main__":
    test_iter_never_leaves_tail_of_one()
    test_chunked_linear_matches_full()
    test_chunked_apply_prealloc_matches_full()
    print("ok")
