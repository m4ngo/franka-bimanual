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
    r_server_ip: str = _arm_field("r", "server_ip")
    r_robot_ip: str = _arm_field("r", "robot_ip")
    r_gripper_ip: str = _arm_field("r", "gripper_ip")
    r_port: int = _arm_field("r", "port")
    control_mode: ControlMode = field(
        default_factory=lambda: ControlMode(fc.profile(PROFILE).control_mode)
    )
    active_arms: tuple[str, ...] = _VALID_ARMS
    # Coulomb friction feedforward, in [0, 1]. Cancels a plant term the sim does
    # not have, so non-zero is CLOSER to osc.py's motion. 0.9 clears the
    # measurement's ~10% spread; 1.0 overshoots (pitch overshot to 126%).
    friction_kc: float = 1.0
    # Per-joint multiplier on friction_kc; see SingleArmFrankaConfig.
    friction_kc_joint: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # Sim-to-real scaling on the EE_DELTA action, applied to the position delta
    # and the axis-angle rotation delta. 1.0 = exactly what the policy emits.
    ee_translation_fudge: float = 1.0
    ee_rotation_fudge: float = 1.0
    # See SingleArmFrankaConfig. Measured on the right arm only.
    kp_ori_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    kp_pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Damping-only trim, per block. kp_*_scale holds kp/kd fixed by construction,
    # so it cannot calm an axis that oscillates -- it stiffens it. These multiply
    # the damping ratio instead, which is the only knob that lowers kp/kd, i.e.
    # slows and damps the axis. Raise these when an axis vibrates; 1.0 is sim.
    kd_ori_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    kd_pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # osc_pose.json sets uncouple_pos_ori=true, which applies Lambda_pos/Lambda_ori
    # to the two halves of the wrench separately. That leaves the arm's own
    # translation<->rotation inertia coupling in the loop AND scales the moment by
    # the ~0.002 kg m^2 wrist inertia, so orientation commands land under breakaway
    # friction. False applies Lambda_full to the whole wrench: response is exactly
    # the commanded acceleration, cross-coupling is zero, and every axis gets more
    # torque at the same gains. True is sim parity; False is what this arm needs.
    uncouple_pos_ori: bool = True
    # Damped-least-squares floor on lambda_full, active only when
    # uncouple_pos_ori is False. Caps lambda_full at 1/mu^2: cond(J) rises
    # 9->58 as the arm raises and an undamped lambda_full took joint 4 past
    # its 69.6 Nm clamp, which shakes. Costs ~24% of the commanded torque at
    # well-conditioned poses, so lower it if translation goes weak.
    lambda_dls_mu: float = 0.025
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
