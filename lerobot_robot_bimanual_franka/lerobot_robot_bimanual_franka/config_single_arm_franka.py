"""Single-arm FR3 follower config.

The exposed key prefix is `r_`, but which PHYSICAL arm that maps to is
declared by the `single_arm_franka` profile in config/rig.yaml (currently the
LEFT FR3 on luigi). Keeping the key prefix decoupled from the arm identity is
what lets previously recorded `r_*` datasets stay readable.
"""

from dataclasses import dataclass, field

import franka_config as fc  # type: ignore
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from .bimanual_franka_config import ControlMode
from .rig_config import profile_arm_fields, profile_cameras, profile_depth_cameras

PROFILE = "single_arm_franka"

_VALID_ARMS: tuple[str, ...] = tuple(fc.profile(PROFILE).arms)


def _arm_field(key: str, suffix: str):
    return field(default_factory=lambda: profile_arm_fields(PROFILE)[f"{key}_{suffix}"])


@RobotConfig.register_subclass("single_arm_franka")
@dataclass
class SingleArmFrankaConfig(RobotConfig):
    control_mode: ControlMode = field(
        default_factory=lambda: ControlMode(fc.profile(PROFILE).control_mode)
    )
    r_server_ip: str = _arm_field("r", "server_ip")
    r_robot_ip: str = _arm_field("r", "robot_ip")
    r_gripper_ip: str = _arm_field("r", "gripper_ip")
    r_port: int = _arm_field("r", "port")
    active_arms: tuple[str, ...] = _VALID_ARMS
    use_noise: bool = False
    noise_pos_scale: float = field(default_factory=lambda: fc.control("noise.pos_scale_m"))
    noise_rot_scale: float = field(default_factory=lambda: fc.control("noise.rot_scale_rad"))
    depth: bool = field(default_factory=lambda: fc.profile(PROFILE).depth)
    depth_cam: dict[str, CameraConfig] = field(
        default_factory=lambda: profile_depth_cameras(PROFILE)
    )
    depth_crop_radius_m: float = field(
        default_factory=lambda: fc.control("observation.depth_crop_radius_m")
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: profile_cameras(PROFILE)
    )
    depth_center_arm: str = field(
        default_factory=lambda: fc.profile(PROFILE).depth_center_arm
    )
    rig_profile: str = PROFILE

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        valid = tuple(fc.profile(self.rig_profile).arms)
        if not self.active_arms:
            raise ValueError(f"active_arms must contain {valid}.")

        invalid = [arm for arm in self.active_arms if arm not in valid]
        if invalid:
            raise ValueError(
                f"Invalid active arm identifiers for single_arm_franka: {invalid}. "
                f"Allowed: {valid}."
            )

        self.active_arms = valid

        camera_names = [str(getattr(camera, "name", "")) for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")

    def arm_name(self, arm_key: str) -> str:
        """Physical arm ("left"/"right") behind an exposed key prefix."""
        return fc.profile(self.rig_profile).arms[arm_key]

    def base_in_world(self, arm_key: str):
        """`franka_config.Pose` of one arm's base in the world frame."""
        return fc.robot_base_in_world(self.arm_name(arm_key))
