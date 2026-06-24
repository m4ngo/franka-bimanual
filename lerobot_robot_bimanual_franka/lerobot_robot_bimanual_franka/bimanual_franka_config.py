from dataclasses import dataclass, field
from enum import Enum

from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore
from lerobot_camera_framos import FramosCameraConfig  # type: ignore

_VALID_ARMS: tuple[str, ...] = ("l", "r")


class ControlMode(str, Enum):
    JOINT_POS = "JOINT_POS"  # joint position setpoints → joint velocity PD
    EE_POS    = "EE_POS"     # absolute EE pose setpoints → VOsc joint velocity
    EE_DELTA  = "EE_DELTA"   # EE delta commands → accumulated goal pose → VOsc joint velocity


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
    control_mode: ControlMode
    active_arms: tuple[str, ...] = _VALID_ARMS
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": ArvCameraConfig(name="gripper_bfs_23595723", ip="192.168.0.142", fps=30, width=224, height=224),
            "cam_2": FramosCameraConfig(name="workspace_framos_d71", ip="192.168.0.116", serial_number="6CD146030D71", fps=30, width=224, height=224),
            "cam_3": ArvCameraConfig(name="gripper_bfs_23595719", ip="192.168.1.138", fps=30, width=224, height=224),
            "cam_4": ArvCameraConfig(name="gripper_bfs_23595720", ip="192.168.1.139", fps=30, width=224, height=224),
            "cam_5": ArvCameraConfig(name="gripper_bfs_23595724", ip="192.168.1.143", fps=30, width=224, height=224),
            "cam_6": FramosCameraConfig(name="workspace_framos_d63", ip="192.168.1.102", serial_number="6CD146030D63", fps=30, width=224, height=224),
        }
    )
    depth: bool = True
    depth_cam: str = "cam_2_scene"
    world_in_robot_translation_m: tuple[float, float, float] = (0.669, 0.003, 0.120)
    world_in_robot_quat_wxyz: tuple[float, float, float, float] = (-0.376557, 0.0, 0.0, 0.926393)
    depth_crop_radius_m: float = 0.4

    # VOsc (Velocity-Space OSC) control parameters
    # Position delta per step when action component = 1.0 (EE_DELTA only)
    osc_output_max_pos: float = 0.05
    # Rotation delta (rad) per step when action magnitude = 1.0 (EE_DELTA only)
    osc_output_max_rot: float = 0.5
    # Base task-space velocity gain (1/s). Final kp = osc_kp_base * 10^kp_action
    osc_kp_base: float = 2.0
    # Nullspace joint-attraction gain (1/s). Keep low; high values cause oscillation
    # when joints drift from q0 during EE control.
    osc_kp_null: float = 0.5
    # Damping ratio; kd = 2 * sqrt(kp) * osc_damping_ratio (1.0 = critically damped)
    osc_damping_ratio: float = 1.0

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if not self.active_arms:
            raise ValueError("active_arms must contain at least one arm: 'l' and/or 'r'.")

        invalid = [arm for arm in self.active_arms if arm not in _VALID_ARMS]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers: {invalid}. Allowed: {_VALID_ARMS}.")

        self.active_arms = tuple(dict.fromkeys(self.active_arms))

        camera_names = [str(getattr(camera, "name", "")) for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")
