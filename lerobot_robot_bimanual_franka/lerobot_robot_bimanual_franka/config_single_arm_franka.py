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
    # Gripper RPyC port -- a DIFFERENT process from the torque server. pylibfranka's
    # Gripper bindings do not release the GIL across blocking calls, so a gripper
    # command issued in the torque server's process stalls the 1 kHz RT thread
    # (measured: 100 ms for a mere read_once) and libfranka aborts the motion with
    # communication_constraints_violation. Set equal to the arm port to go back to
    # sharing one process.
    r_gripper_port: int = 18822
    active_arms: tuple[str, ...] = ("r",)
    # See BimanualFrankaConfig.friction_kc.
    friction_kc: float = 0.9
    # Sim-to-real scaling on the EE_DELTA action, applied to the position delta
    # and the axis-angle rotation delta. 1.0 = exactly what the policy emits.
    ee_translation_fudge: float = 1.0
    ee_rotation_fudge: float = 1.0
    # Per-axis OSC gain scales, capped at 10 by KP_LIMITS. Stiffness only: the
    # damping ratio is derived as sqrt(scale), so these buy friction rejection
    # and not speed. Measure with scripts/check_osc_axes.py; 1.0 is sim.
    # Orientation: lambda_ori is 0.028/0.031/0.0019 kg m^2 against robosuite's
    # 0.18-0.58, so a sim-gain wrist moment lands under breakaway.
    kp_ori_scale: tuple[float, float, float] = (1.0, 1.0, 2.5)
    # Translation: prefer 1.0. Unlike lambda_ori, lambda_pos rotates with the
    # arm (lambda_pos_XX 0.74 kg folded, 4.2 extended), so a base-frame constant
    # tuned where X is light over-drives it elsewhere -- 4.0 reached 84 Nm
    # against the 69.6 Nm clamp at full reach. Prefer friction_kc, pose-independent.
    kp_pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # osc_pose.json sets uncouple_pos_ori=true, which applies Lambda_pos/Lambda_ori
    # to the two halves of the wrench separately. That leaves the arm's own
    # translation<->rotation inertia coupling in the loop AND scales the moment by
    # the ~0.002 kg m^2 wrist inertia, so orientation commands land under breakaway
    # friction. False applies Lambda_full to the whole wrench: response is exactly
    # the commanded acceleration, cross-coupling is zero, and every axis gets more
    # torque at the same gains. True is sim parity; False is what this arm needs.
    uncouple_pos_ori: bool = True
    use_noise: bool = False
    noise_pos_scale: float = 0.01   # metres, added to position output each step
    noise_rot_scale: float = 0.075    # radians (axis-angle), added to rotation output each step
    depth: bool = True
    depth_cam: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_2_scene": FramosCameraConfig(enable_color=False, name="workspace_framos_d71", ip="192.168.0.116", serial_number="6CD146030D71", fps=30, width=224, height=224,
                                                intrinsic_matrix = (
                                                    (946.73319511, 0.0, 632.15541524),
                                                    (0.0, 963.49477373, 368.33009756),
                                                    (0.0, 0.0, 1.0),
                                                ),
                                                distortion_coeffs = (
                                                    -7.46601288e-02,
                                                    2.27627524e+00,
                                                    -2.34761926e-03,
                                                    2.86842857e-03,
                                                    -1.06307592e+01
                                                ),
                                                r_cam_in_world  = (
                                                    (-0.93549331, -0.02391077, 0.35253446),
                                                    (-0.21260197, 0.83499221, -0.50753169),
                                                    (-0.28222806, -0.54974202, -0.7862131),
                                                ),
                                                t_cam_in_world = (-0.33514749, 0.63967298, 0.912236053)
                                            ),
            "cam_6_scene": FramosCameraConfig(name="workspace_framos_d63", ip="192.168.1.102", serial_number="6CD146030D63", fps=30, width=224, height=224),
        }
    )
    world_in_robot_translation_m: tuple[float, float, float] = (0.669, 0.003, 0.120)
    world_in_robot_quat_wxyz: tuple[float, float, float, float] = (-0.376557, 0.0, 0.0, 0.926393)
    depth_crop_radius_m: float = 0.3
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_3_wrist": ArvCameraConfig(name="gripper_bfs_23595719", ip="192.168.1.138", fps=30, width=224, height=224),
            "cam_4_wrist": ArvCameraConfig(name="gripper_bfs_23595720", ip="192.168.1.139", fps=30, width=224, height=224),
            "cam_6_scene": FramosCameraConfig(name="workspace_framos_d63", ip="192.168.1.102", serial_number="6CD146030D63", fps=30, width=224, height=224),
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
