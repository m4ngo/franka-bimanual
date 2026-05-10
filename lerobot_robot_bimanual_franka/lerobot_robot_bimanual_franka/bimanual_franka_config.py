from dataclasses import dataclass, field

from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore
from lerobot_camera_framos import FramosCameraConfig  # type: ignore

_VALID_ARMS: tuple[str, ...] = ("l", "r")


@RobotConfig.register_subclass("bimanual_franka")
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
    use_ee_pos: bool
    active_arms: tuple[str, ...] = _VALID_ARMS
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": ArvCameraConfig(name="gripper_bfs_23595723", ip="192.168.0.142", fps=15, width=224, height=224),
            "cam_2": FramosCameraConfig(name="workspace_framos_d71", ip="192.168.0.116", serial_number="6CD146030D71", fps=15, width=224, height=224),
            "cam_3": ArvCameraConfig(name="gripper_bfs_23595719", ip="192.168.1.138", fps=15, width=224, height=224),
            "cam_4": ArvCameraConfig(name="gripper_bfs_23595720", ip="192.168.1.139", fps=15, width=224, height=224),
            "cam_5": ArvCameraConfig(name="gripper_bfs_23595724", ip="192.168.1.143", fps=15, width=224, height=224),
            "cam_6": FramosCameraConfig(name="workspace_framos_d63", ip="192.168.1.102", serial_number="6CD146030D63", fps=15, width=224, height=224),
        }
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if not self.active_arms:
            raise ValueError("active_arms must contain at least one arm: 'l' and/or 'r'.")

        invalid = [arm for arm in self.active_arms if arm not in _VALID_ARMS]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers: {invalid}. Allowed: {_VALID_ARMS}.")

        self.active_arms = tuple(dict.fromkeys(self.active_arms))

        camera_names = [camera.name for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")
