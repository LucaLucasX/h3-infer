"""Load h3_graph — tries .py source first, then falls back to .pyc bytecode."""

from __future__ import annotations

import importlib.util
import marshal
import sys
import types
from pathlib import Path

if "h3_graph" not in sys.modules:
    ROOT = Path(__file__).resolve().parents[1]
    py_src = ROOT / "h3_graph.py"
    pyc = ROOT / "__pycache__" / "h3_graph.cpython-311.pyc"

    if py_src.exists():
        # Preferred: load from .py source directly
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import importlib
        importlib.import_module("h3_graph")
    elif pyc.exists():
        # Fallback: load from compiled bytecode
        with open(pyc, "rb") as f:
            f.read(16)
            code = marshal.load(f)
        mod = types.ModuleType("h3_graph")
        mod.__file__ = str(pyc)
        sys.modules["h3_graph"] = mod
        exec(code, mod.__dict__)
    else:
        raise ImportError(
            f"h3_graph not found: tried {py_src} and {pyc}"
        )
