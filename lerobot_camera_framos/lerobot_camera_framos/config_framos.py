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
    color_format: str = "bgr8"
    depth_format: str = "z16"
    options: dict[str, float] = field(default_factory=dict)
