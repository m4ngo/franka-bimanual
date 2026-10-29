"""Single-arm FR3 follower config.

The exposed key prefix is always `r_`; which PHYSICAL arm and cameras it maps
to is the `rig_profile` field, a profile in config/rig.yaml. Every field left
at None is filled from that profile in `__post_init__`, so
`--robot.rig_profile` alone moves the arm and cameras together, and an explicit
field override still wins. Keeping the key prefix decoupled from the arm
identity is what lets previously recorded `r_*` datasets stay readable.
"""

from dataclasses import dataclass, field

import franka_config as fc  # type: ignore
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from .bimanual_franka_config import ControlMode
from .rig_config import profile_arm_fields, profile_cameras, profile_depth_cameras

# The only key prefix this config's fields are named for.
ARM_KEY = "r"


def single_arm_profiles() -> tuple[str, ...]:
    """Rig profiles this config can drive: one arm, exposed under ARM_KEY."""
    return tuple(n for n in fc.profile_names() if fc.profile(n).keys == (ARM_KEY,))


@RobotConfig.register_subclass("single_arm_franka")
@dataclass
class SingleArmFrankaConfig(RobotConfig):
    rig_profile: str = field(default_factory=fc.default_single_arm_profile)
    control_mode: ControlMode | None = None
    r_server_ip: str | None = None
    r_robot_ip: str | None = None
    r_gripper_ip: str | None = None
    r_port: int | None = None
    # Gripper RPyC port -- a DIFFERENT process from the torque server. pylibfranka's
    # Gripper bindings do not release the GIL across blocking calls, so a gripper
    # command issued in the torque server's process stalls the 1 kHz RT thread
    # (measured: 100 ms for a mere read_once) and libfranka aborts the motion with
    # communication_constraints_violation. Set an arm's gripper_rpyc_port equal to
    # its rpyc_port in arms.yaml to go back to sharing one process.
    #
    # Resolved through the profile, NOT hardcoded: it differs per arm (18822
    # right, 18823 left) and the `r_` prefix does not say which arm this is.
    r_gripper_port: int | None = None
    active_arms: tuple[str, ...] | None = None
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
    # Gaussian goal noise, std dev per axis per step; 0.0 is off. See action_noise.
    noise_pos_scale: float = 0.0
    noise_rot_scale: float = 0.0
    noise_joint_scale: float = 0.0
    depth: bool | None = None
    depth_cam: dict[str, CameraConfig] | None = None
    depth_crop_radius_m: float = field(
        default_factory=lambda: fc.control("observation.depth_crop_radius_m")
    )
    cameras: dict[str, CameraConfig] | None = None
    depth_center_arm: str | None = None

    def __post_init__(self):
        profile = fc.profile(self.rig_profile)
        if profile.keys != (ARM_KEY,):
            raise ValueError(
                f"rig {self.rig_profile!r} exposes keys {profile.keys}; this config drives "
                f"one arm under {ARM_KEY!r}. Single-arm rigs: {single_arm_profiles()}."
            )
        for name, value in profile_arm_fields(self.rig_profile).items():
            if getattr(self, name) is None:
                setattr(self, name, value)
        if self.control_mode is None:
            self.control_mode = ControlMode(profile.control_mode)
        if self.depth is None:
            self.depth = profile.depth
        if self.depth_cam is None:
            self.depth_cam = profile_depth_cameras(self.rig_profile)
        if self.cameras is None:
            self.cameras = profile_cameras(self.rig_profile)
        if self.depth_center_arm is None:
            self.depth_center_arm = profile.depth_center_arm

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        valid = profile.keys
        if self.active_arms is None:
            self.active_arms = valid
        if not self.active_arms:
            raise ValueError(f"active_arms must contain {valid}.")

        invalid = [arm for arm in self.active_arms if arm not in valid]
        if invalid:
            raise ValueError(
                f"Invalid active arm identifiers for {self.rig_profile}: {invalid}. "
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
