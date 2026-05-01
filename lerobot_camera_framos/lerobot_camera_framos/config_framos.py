from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("framos_camera")
@dataclass
class FramosCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    serial_number: str = ""
    enable_color: bool = True
    enable_depth: bool = True
    color_width: int = 1280
    color_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    align_to: str = "color"
    color_format: str = "rgb8"
    depth_format: str = "z16"
    #: librealsense only accepts discrete FPS (typically 6/15/30/60/90 on D415e).
    #: If unset, FPS is snapped from `CameraConfig.fps` automatically in `FramosCamera`.
    streaming_fps: int | None = None
    options: dict[str, float] = field(default_factory=dict)
