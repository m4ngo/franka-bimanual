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
    # Gripper RPyC port -- a DIFFERENT process from the torque server. pylibfranka's
    # Gripper bindings do not release the GIL across blocking calls, so a gripper
    # command issued in the torque server's process stalls the 1 kHz RT thread
    # (measured: 100 ms for a mere read_once) and libfranka aborts the motion with
    # communication_constraints_violation. Set an arm's gripper_rpyc_port equal to
    # its rpyc_port in arms.yaml to go back to sharing one process.
    #
    # Resolved through the profile, NOT hardcoded: this rig exposes the LEFT arm
    # under `r_`, whose gripper server is on 18823, not the right arm's 18822.
    r_gripper_port: int = _arm_field("r", "gripper_port")
    active_arms: tuple[str, ...] = _VALID_ARMS
    # Every knob below is a per-rig hardware trim and lives in ONE place,
    # config/control.yaml's `tuning:` block. A literal default here would win
    # over the yaml and make it decoration -- see config/README.md.
    friction_kc: float = field(default_factory=lambda: fc.control("tuning.friction_kc"))
    # Per-joint assist trim, SPLIT BY ROTATION DIRECTION -- 14 numbers. `_pos`
    # applies where the commanded torque on that joint is positive. Breakaway on
    # this arm is directional; measure with
    # `scripts/measure_joint_friction.py --directional`.
    friction_kc_joint_pos: tuple[float, ...] = field(
        default_factory=lambda: tuple(fc.control("tuning.friction_kc_joint_pos"))
    )
    friction_kc_joint_neg: tuple[float, ...] = field(
        default_factory=lambda: tuple(fc.control("tuning.friction_kc_joint_neg"))
    )
    ee_translation_fudge: float = field(
        default_factory=lambda: fc.control("tuning.ee_translation_fudge")
    )
    ee_rotation_fudge: float = field(
        default_factory=lambda: fc.control("tuning.ee_rotation_fudge")
    )
    # Orientation stiffness trim: lambda_ori is 0.028/0.031/0.0019 kg m^2 against
    # robosuite's 0.18-0.58, so a sim-gain wrist moment lands under breakaway.
    # Measure with scripts/osc_check/check_osc_axes.py.
    kp_ori_scale: tuple[float, float, float] = field(
        default_factory=lambda: tuple(fc.control("tuning.kp_ori_scale"))
    )
    # Translation: prefer 1.0. Unlike lambda_ori, lambda_pos rotates with the
    # arm (lambda_pos_XX 0.74 kg folded, 4.2 extended), so a base-frame constant
    # tuned where X is light over-drives it elsewhere -- 4.0 reached 84 Nm
    # against the 69.6 Nm clamp at full reach. Prefer friction_kc, pose-independent.
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
