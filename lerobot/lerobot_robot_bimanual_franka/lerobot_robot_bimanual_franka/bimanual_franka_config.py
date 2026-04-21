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
    active_arms: tuple[str, ...] = ("l", "r")

    def __post_init__(self):
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

        if len(self.active_arms) == 0:
            raise ValueError("active_arms must contain at least one arm: 'l' and/or 'r'.")

        invalid = [arm for arm in self.active_arms if arm not in ("l", "r")]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers: {invalid}. Allowed values are 'l' and 'r'.")

        self.active_arms = tuple(dict.fromkeys(self.active_arms))
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