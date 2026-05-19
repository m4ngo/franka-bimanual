from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore
from lerobot_camera_framos import FramosCameraConfig  # type: ignore


@RobotConfig.register_subclass("single_arm_franka")
@dataclass
class SingleArmFrankaConfig(RobotConfig):
    r_server_ip: str
    r_robot_ip: str
    r_gripper_ip: str
    r_port: int
    use_ee_pos: bool
    active_arms: tuple[str, ...] = ("r",)
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_3": ArvCameraConfig(name="gripper_bfs_23595719", ip="192.168.1.138", fps=30, width=224, height=224),
            "cam_4": ArvCameraConfig(name="gripper_bfs_23595720", ip="192.168.1.139", fps=30, width=224, height=224),
            "cam_6": FramosCameraConfig(name="workspace_framos_d63", ip="192.168.1.102", serial_number="6CD146030D63", fps=30, width=224, height=224),
        }
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if not self.active_arms:
            raise ValueError("active_arms must contain 'r'.")

        invalid = [arm for arm in self.active_arms if arm != "r"]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers for single_arm_franka: {invalid}. Allowed: ('r',).")

        self.active_arms = ("r",)

        camera_names = [camera.name for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")