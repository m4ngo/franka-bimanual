from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("my_cool_robot")
@dataclass
class MyCoolRobotConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=480,
                height=640,
            ),
        }
    )