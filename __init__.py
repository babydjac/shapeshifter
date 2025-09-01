from .nodes import (
    SimpleVideoLoader,
    SimpleImageLoader,
    ShapeShifterLoader,
)

NODE_CLASS_MAPPINGS = {
    "SimpleVideoLoader": SimpleVideoLoader,
    "SimpleImageLoader": SimpleImageLoader,
    "ShapeShifterLoader": ShapeShifterLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleVideoLoader": "🎬 Simple Video Loader",
    "SimpleImageLoader": "🖼️ Simple Image Loader",
    "ShapeShifterLoader": "🌀 Shape‑Shifter Loader",
}

# Serve frontend assets
WEB_DIRECTORY = "./js"

# Register routes (optional preview endpoint)
try:
    from . import routes  # noqa: F401
except Exception as e:
    # Avoid hard-failing ComfyUI startup on optional route errors
    print(f"[shapeshifter] routes import skipped: {e}")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
