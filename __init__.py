"""ComfyUI-Temporal-Face-Detailer

Video FaceDetailer for SDXL: detects and tracks faces across a frame
batch, re-details each tracked face with stable seeds/crops, and blends
results temporally to suppress flicker.
"""

from .tfd.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
