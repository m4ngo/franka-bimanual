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
    # communication_constraints_violation. Set equal to the arm port to go back to
    # sharing one process.
    r_gripper_port: int = 18822
    active_arms: tuple[str, ...] = _VALID_ARMS
    # See BimanualFrankaConfig.friction_kc.
    friction_kc: float = 0.9
    # Per-joint multiplier on friction_kc; see SingleArmFrankaConfig.
    friction_kc_joint: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 12.0)
    # Sim-to-real scaling on the EE_DELTA action, applied to the position delta
    # and the axis-angle rotation delta. 1.0 = exactly what the policy emits.
    ee_translation_fudge: float = field(
        default_factory=lambda: fc.control("fudge.ee_translation")
    )
    ee_rotation_fudge: float = field(
        default_factory=lambda: fc.control("fudge.ee_rotation")
    )
    # Per-axis OSC gain scales, capped at 10 by KP_LIMITS. Stiffness only: the
    # damping ratio is derived as sqrt(scale), so these buy friction rejection
    # and not speed. Measure with scripts/check_osc_axes.py; 1.0 is sim.
    # Orientation: lambda_ori is 0.028/0.031/0.0019 kg m^2 against robosuite's
    # 0.18-0.58, so a sim-gain wrist moment lands under breakaway.
    kp_ori_scale: tuple[float, float, float] = (0.8, 0.8, 0.8)
    # Translation: prefer 1.0. Unlike lambda_ori, lambda_pos rotates with the
    # arm (lambda_pos_XX 0.74 kg folded, 4.2 extended), so a base-frame constant
    # tuned where X is light over-drives it elsewhere -- 4.0 reached 84 Nm
    # against the 69.6 Nm clamp at full reach. Prefer friction_kc, pose-independent.
    kp_pos_scale: tuple[float, float, float] = (1.5, 1.5, 1.5)
    # Damping-only trim, per block. kp_*_scale holds kp/kd fixed by construction,
    # so it cannot calm an axis that oscillates -- it stiffens it. These multiply
    # the damping ratio instead, which is the only knob that lowers kp/kd, i.e.
    # slows and damps the axis. Raise these when an axis vibrates; 1.0 is sim.
    kd_ori_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    kd_pos_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    # True is osc_pose.json's setting and applies Lambda_pos/Lambda_ori to the two
    # halves of the wrench separately, i.e. DROPS the coupling terms: it sizes the
    # translation force as if rotation were free, then fights the rotation that
    # push causes. On this arm that leaves joint 4 at 0.25x breakaway on +X.
    # False applies Lambda_full to the whole wrench -- the exact operational-space
    # form -- which is what lets X move: 43-71% -> ~100% of command, and
    # pose-independent. Requires two guards, both on by default in the control
    # loop: LAMBDA_DLS_MU (the coupled 6x6 goes singular) and the dropped
    # orientation->force block (that term turns an orientation error the wrist
    # cannot clear into a standing translational push -- 215 mm of walk measured).
    uncouple_pos_ori: bool = False
    # Damped-least-squares floor on lambda_full, active only when
    # uncouple_pos_ori is False. Caps lambda_full at 1/mu^2: cond(J) rises
    # 9->58 as the arm raises and an undamped lambda_full took joint 4 past
    # its 69.6 Nm clamp, which shakes. Costs ~24% of the commanded torque at
    # well-conditioned poses, so lower it if translation goes weak.
    lambda_dls_mu: float = 0.025
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
