from typing import Any

from .config_framos import FramosCameraConfig

try:
    from .framos import FramosCamera
except Exception:
    # Allow config-only imports in environments missing pyrealsense2 or
    # FRAMOS librealsense2 at runtime.
    FramosCamera: Any = None

__all__ = ["FramosCamera", "FramosCameraConfig"]
