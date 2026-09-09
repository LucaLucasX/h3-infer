#!/usr/bin/env python3
"""Build sageattention / sageattn3 from source in-process (CUDA 13 + torch cu130)."""
from __future__ import annotations

import os
import sys


def patch_cuda_check() -> None:
    import torch.utils.cpp_extension as ce

    def _noop(*_a, **_k):
        return None

    ce._check_cuda_version = _noop  # type: ignore[attr-defined]


def install(repo_subdir: str | None = None) -> None:
    root = os.environ.get("SAGEATTN_SRC", "/tmp/SageAttention")
    cwd = os.path.join(root, repo_subdir) if repo_subdir else root
    os.chdir(cwd)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "4")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    patch_cuda_check()
    sys.argv = ["setup.py", "install"]
    import runpy

    runpy.run_path("setup.py", run_name="__main__")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sageattention"
    if target == "sageattention":
        install()
    elif target == "sageattn3":
        install("sageattention3_blackwell")
    else:
        raise SystemExit(f"unknown target: {target}")
