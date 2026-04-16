from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("my_cool_robot")
@dataclass
class BimanualFrankaConfig(RobotConfig):
    l_server_ip: str
    l_robot_ip: str
    l_gripper_ip: str
    l_port: int
    r_server_ip: str
    r_robot_ip: str
    r_gripper_ip: str
    r_port: int
    use_ee_delta: bool
    # cameras: dict[str, CameraConfig] = field(
    #     default_factory=lambda: {
    #         "cam_1": OpenCVCameraConfig(
    #             index_or_path=2,
    #             fps=30,
    #             width=480,
    #             height=640,
    #         ),
    #     }
    # )