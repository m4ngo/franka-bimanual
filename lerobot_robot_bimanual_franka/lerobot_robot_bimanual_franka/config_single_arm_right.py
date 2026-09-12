"""Single-arm FR3 follower config for the PHYSICAL RIGHT arm.

Identical to SingleArmFrankaConfig apart from the profile-derived fields, so both
rigs expose the same `r_` keys and record interchangeable datasets. The arm
identity lives in the `single_arm_right` profile in config/rig.yaml, never here.
"""

from dataclasses import dataclass, field

import franka_config as fc  # type: ignore
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from .bimanual_franka_config import ControlMode
from .config_single_arm_franka import SingleArmFrankaConfig
from .rig_config import profile_arm_fields, profile_cameras, profile_depth_cameras

PROFILE = "single_arm_right"


def _arm_field(key: str, suffix: str):
    return field(default_factory=lambda: profile_arm_fields(PROFILE)[f"{key}_{suffix}"])


@RobotConfig.register_subclass("single_arm_right")
@dataclass
class SingleArmRightConfig(SingleArmFrankaConfig):
    control_mode: ControlMode = field(
        default_factory=lambda: ControlMode(fc.profile(PROFILE).control_mode)
    )
    r_server_ip: str = _arm_field("r", "server_ip")
    r_robot_ip: str = _arm_field("r", "robot_ip")
    r_gripper_ip: str = _arm_field("r", "gripper_ip")
    r_port: int = _arm_field("r", "port")
    # 18822 on the right arm, not the left's 18823 -- resolved through the profile.
    r_gripper_port: int = _arm_field("r", "gripper_port")
    active_arms: tuple[str, ...] = tuple(fc.profile(PROFILE).arms)
    depth: bool = field(default_factory=lambda: fc.profile(PROFILE).depth)
    depth_cam: dict[str, CameraConfig] = field(
        default_factory=lambda: profile_depth_cameras(PROFILE)
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: profile_cameras(PROFILE)
    )
    depth_center_arm: str = field(
        default_factory=lambda: fc.profile(PROFILE).depth_center_arm
    )
    rig_profile: str = PROFILE
