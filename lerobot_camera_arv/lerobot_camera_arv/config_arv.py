from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("arv_camera")
@dataclass
class ArvCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    pixel_format: str = "Mono8"
