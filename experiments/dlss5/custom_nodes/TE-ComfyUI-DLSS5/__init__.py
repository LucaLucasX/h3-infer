"""TE-ComfyUI-DLSS5

ComfyUI nodes for streaming video through a separately built NVIDIA NGX
backend.  The Python package intentionally contains no NVIDIA SDK code.
"""

import os
import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

LOGGER = logging.getLogger("TE-ComfyUI-DLSS5")
if os.name != "nt":
    LOGGER.warning(
        "TE-ComfyUI-DLSS5 is loaded, but Feature 18 Neural Rendering needs "
        "Windows + D3D12 + nvngx_dlssnr.dll. Nodes register; execute will fail on Linux."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
