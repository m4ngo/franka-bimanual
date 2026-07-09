from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore
from lerobot_camera_framos import FramosCameraConfig  # type: ignore

from .bimanual_franka_config import ControlMode


@RobotConfig.register_subclass("single_arm_franka")
@dataclass
class SingleArmFrankaConfig(RobotConfig):
    r_server_ip: str
    r_robot_ip: str
    r_gripper_ip: str
    r_port: int
    control_mode: ControlMode
    active_arms: tuple[str, ...] = ("r",)
    use_noise: bool = False
    noise_pos_scale: float = 0.005   # metres, added to position output each step
    noise_rot_scale: float = 0.02    # radians (axis-angle), added to rotation output each step
    depth: bool = True
    depth_cam: str = "cam_2_scene"
    world_in_robot_translation_m: tuple[float, float, float] = (0.669, 0.003, 0.120)
    world_in_robot_quat_wxyz: tuple[float, float, float, float] = (-0.376557, 0.0, 0.0, 0.926393)
    depth_crop_radius_m: float = 0.4
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_3_wrist": ArvCameraConfig(name="gripper_bfs_23595719", ip="192.168.1.138", fps=30, width=224, height=224),
            "cam_4_wrist": ArvCameraConfig(name="gripper_bfs_23595720", ip="192.168.1.139", fps=30, width=224, height=224),
            "cam_2_scene": FramosCameraConfig(name="workspace_framos_d71", ip="192.168.0.116", serial_number="6CD146030D71", fps=30, width=224, height=224),
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

        camera_names = [str(getattr(camera, "name", "")) for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")
