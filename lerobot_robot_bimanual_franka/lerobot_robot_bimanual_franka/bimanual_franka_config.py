"""Bimanual FR3 follower config. All defaults come from config/*.yaml."""

from dataclasses import dataclass, field
from enum import Enum

import franka_config as fc  # type: ignore
from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig

from .rig_config import profile_arm_fields, profile_cameras, profile_depth_cameras

PROFILE = "bimanual_franka"

_VALID_ARMS: tuple[str, ...] = tuple(fc.profile(PROFILE).arms)


class ControlMode(str, Enum):
    JOINT_POS = "JOINT_POS"  # joint position setpoints → joint velocity PD
    EE_POS    = "EE_POS"     # absolute EE pose setpoints → Cartesian velocity PD
    EE_DELTA  = "EE_DELTA"   # EE delta commands applied directly as Cartesian velocity


def _arm_field(key: str, suffix: str):
    return field(default_factory=lambda: profile_arm_fields(PROFILE)[f"{key}_{suffix}"])


@RobotConfig.register_subclass("bimanual_franka")
@dataclass
class BimanualFrankaConfig(RobotConfig):
    l_server_ip: str = _arm_field("l", "server_ip")
    l_robot_ip: str = _arm_field("l", "robot_ip")
    l_gripper_ip: str = _arm_field("l", "gripper_ip")
    l_port: int = _arm_field("l", "port")
    # Gripper RPyC port — a different process from the torque server, and a
    # different port per arm. See SingleArmFrankaConfig.r_gripper_port.
    l_gripper_port: int = _arm_field("l", "gripper_port")
    r_server_ip: str = _arm_field("r", "server_ip")
    r_robot_ip: str = _arm_field("r", "robot_ip")
    r_gripper_ip: str = _arm_field("r", "gripper_ip")
    r_port: int = _arm_field("r", "port")
    r_gripper_port: int = _arm_field("r", "gripper_port")
    control_mode: ControlMode = field(
        default_factory=lambda: ControlMode(fc.profile(PROFILE).control_mode)
    )
    active_arms: tuple[str, ...] = _VALID_ARMS
    # Every knob below is a per-rig hardware trim and lives in ONE place,
    # config/control.yaml's `tuning:` block. A literal default here would win
    # over the yaml and make it decoration -- see config/README.md.
    friction_kc: float = field(default_factory=lambda: fc.control("tuning.friction_kc"))
    friction_kc_joint: tuple[float, ...] = field(
        default_factory=lambda: tuple(fc.control("tuning.friction_kc_joint"))
    )
    ee_translation_fudge: float = field(
        default_factory=lambda: fc.control("tuning.ee_translation_fudge")
    )
    ee_rotation_fudge: float = field(
        default_factory=lambda: fc.control("tuning.ee_rotation_fudge")
    )
    kp_ori_scale: tuple[float, float, float] = field(
        default_factory=lambda: tuple(fc.control("tuning.kp_ori_scale"))
    )
    kp_pos_scale: tuple[float, float, float] = field(
        default_factory=lambda: tuple(fc.control("tuning.kp_pos_scale"))
    )
    kd_ori_scale: tuple[float, float, float] = field(
        default_factory=lambda: tuple(fc.control("tuning.kd_ori_scale"))
    )
    kd_pos_scale: tuple[float, float, float] = field(
        default_factory=lambda: tuple(fc.control("tuning.kd_pos_scale"))
    )
    use_noise: bool = False
    noise_pos_scale: float = field(default_factory=lambda: fc.control("noise.pos_scale_m"))
    noise_rot_scale: float = field(default_factory=lambda: fc.control("noise.rot_scale_rad"))
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: profile_cameras(PROFILE)
    )
    depth: bool = field(default_factory=lambda: fc.profile(PROFILE).depth)
    depth_cam: dict[str, CameraConfig] = field(
        default_factory=lambda: profile_depth_cameras(PROFILE)
    )
    depth_crop_radius_m: float = field(
        default_factory=lambda: fc.control("observation.depth_crop_radius_m")
    )
    #: Arm key whose EE centres the depth crop.
    depth_center_arm: str = field(
        default_factory=lambda: fc.profile(PROFILE).depth_center_arm
    )
    #: Rig profile name — resolves per-arm base poses out of config/world.yaml.
    rig_profile: str = PROFILE

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        valid = tuple(fc.profile(self.rig_profile).arms)
        if not self.active_arms:
            raise ValueError(f"active_arms must contain at least one of {valid}.")

        invalid = [arm for arm in self.active_arms if arm not in valid]
        if invalid:
            raise ValueError(f"Invalid active arm identifiers: {invalid}. Allowed: {valid}.")

        self.active_arms = tuple(dict.fromkeys(self.active_arms))

        camera_names = [str(getattr(camera, "name", "")) for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")

    def arm_name(self, arm_key: str) -> str:
        """Physical arm ("left"/"right") behind an exposed key prefix."""
        return fc.profile(self.rig_profile).arms[arm_key]

    def base_in_world(self, arm_key: str):
        """`franka_config.Pose` of one arm's base in the world frame."""
        return fc.robot_base_in_world(self.arm_name(arm_key))
