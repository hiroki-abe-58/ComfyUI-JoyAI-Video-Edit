"""JoyAI-Video-Edit for ComfyUI (unofficial integration of https://github.com/jd-opensource/JoyAI-Video-Edit).

Only ``comfy_entrypoint`` (V3) is exported: a legacy NODE_CLASS_MAPPINGS would be checked first by ComfyUI and
shadow the V3 path.
"""

import logging

try:
    from .joyai_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API
    logging.error("[JoyAI-Video-Edit] not loaded: %s", e)
