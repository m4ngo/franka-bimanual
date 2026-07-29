from dataclasses import dataclass, field
from enum import Enum

from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore
from lerobot_camera_framos import FramosCameraConfig  # type: ignore

_VALID_ARMS: tuple[str, ...] = ("l", "r")


class ControlMode(str, Enum):
    JOINT_POS = "JOINT_POS"  # joint position setpoints → joint velocity PD
    EE_POS    = "EE_POS"     # absolute EE pose setpoints → Cartesian velocity PD
    EE_DELTA  = "EE_DELTA"   # EE delta commands applied directly as Cartesian velocity


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
    # Gripper RPyC port -- a DIFFERENT process from the torque server. pylibfranka's
    # Gripper bindings do not release the GIL across blocking calls, so a gripper
    # command issued in the torque server's process stalls the 1 kHz RT thread
    # (measured: 100 ms for a mere read_once) and libfranka aborts the motion with
    # communication_constraints_violation. Set equal to the arm port to go back to
    # sharing one process.
    l_gripper_port: int = 18822
    r_gripper_port: int = 18823
    active_arms: tuple[str, ...] = _VALID_ARMS
    # Coulomb friction feedforward, in [0, 1]. Cancels a plant term the sim does
    # not have, so non-zero is CLOSER to osc.py's motion. 0.9 clears the
    # measurement's ~10% spread; 1.0 overshoots (pitch overshot to 126%).
    friction_kc: float = 0.9
    # Sim-to-real scaling on the EE_DELTA action, applied to the position delta
    # and the axis-angle rotation delta. 1.0 = exactly what the policy emits.
    ee_translation_fudge: float = 1.0
    ee_rotation_fudge: float = 1.0
    # See SingleArmFrankaConfig. Measured on the right arm only.
    kp_ori_scale: tuple[float, float, float] = (5.0, 5.0, 10.0)
    kp_pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # osc_pose.json sets uncouple_pos_ori=true, which applies Lambda_pos/Lambda_ori
    # to the two halves of the wrench separately. That leaves the arm's own
    # translation<->rotation inertia coupling in the loop AND scales the moment by
    # the ~0.002 kg m^2 wrist inertia, so orientation commands land under breakaway
    # friction. False applies Lambda_full to the whole wrench: response is exactly
    # the commanded acceleration, cross-coupling is zero, and every axis gets more
    # torque at the same gains. True is sim parity; False is what this arm needs.
    uncouple_pos_ori: bool = False
    use_noise: bool = False
    noise_pos_scale: float = 0.005   # metres, added to position output each step
    noise_rot_scale: float = 0.01   # radians (axis-angle), added to rotation output each step
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
    depth_cam: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_2": FramosCameraConfig(enable_color=False, name="workspace_framos_d71", ip="192.168.0.116", serial_number="6CD146030D71", fps=30, width=224, height=224),
        }
    )
    world_in_robot_translation_m: tuple[float, float, float] = (0.669, 0.003, 0.120)
    world_in_robot_quat_wxyz: tuple[float, float, float, float] = (-0.376557, 0.0, 0.0, 0.926393)
    depth_crop_radius_m: float = 0.4

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
