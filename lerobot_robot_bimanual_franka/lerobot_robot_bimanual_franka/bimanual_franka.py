import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import cast, Optional

import franka_config as fc
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from lerobot_camera_arv import ArvCamera, ArvCameraConfig
from lerobot_camera_framos import FramosCamera, FramosCameraConfig
from scipy.spatial.transform import Rotation

from .bimanual_franka_config import BimanualFrankaConfig, ControlMode
from .franka_gripper import FrankaGripper
from .franka_fk import franka_fk
from .franka_process import NUM_JOINTS, KinematicSnapshot, MultiRobotWrapper
from .safety import ActionSafetyScreen
from .wsg import WSG
from .osc_torque_controller import (
    DAMPING_EXP_SCALE,
    DEFAULT_DAMPING_RATIO,
    DEFAULT_JOINT_KD,
    DEFAULT_JOINT_KP,
    DEFAULT_KP,
    JOINT_TORQUE_LIMITS,
    KP_EXP_SCALE,
    clip_delta,
    resolve_gains,
)
from .franka_jacobian import zero_jacobian  # the new analytic module

# Every constant below comes from config/control.yaml — either read here, or
# re-exported from osc_torque_controller, which resolves the same `torque:` block
# (see torque_config.py, which is what also gets those values onto the NUC).
IMAGE_CHANNELS = fc.control("observation.image_channels")
_CAMERA_READ_TIMEOUT_MS: float = fc.control("observation.camera_read_timeout_ms")
_CONNECT_TIMEOUT_S = fc.control("franka.connect_timeout_s")
_DEPTH_POINT_COUNT = fc.control("observation.depth_point_count")
_GRIP_ACCUM_SPEED = fc.control("gripper_accum_speed")
# Age past which get_observation()'s kin snapshot is re-read instead of reused.
# EE_DELTA anchors its goal on the measured pose, so a stale anchor silently
# subtracts whatever the arm travelled in between from the commanded delta.
_KIN_CACHE_MAX_AGE_S = fc.control("observation.kin_cache_max_age_s")

# Parity knobs on the incoming delta action; 1.0 = exactly what the policy emits.
# Fallbacks only -- the robot config resolves the same `tuning:` block.
_EE_TRANSLATION_FUDGE_FACTOR = fc.control("tuning.ee_translation_fudge")
_EE_ROTATION_FUDGE_FACTOR = fc.control("tuning.ee_rotation_fudge")

# Exponential action->gain remap, matching the sim wrapper the policies were
# trained against (utils/envs/libero.py: exp_scale = limit_max / default).
# kp_gain/kd_gain are the multipliers; OSC_BASE_KP/OSC_BASE_DAMPING_RATIO are
# the robosuite defaults they multiply. All four are DERIVED from
# torque.osc.{default_kp,kp_limits,default_damping_ratio,damping_ratio_limits}.
_KP_GAIN_BASE = KP_EXP_SCALE
_KD_GAIN_BASE = DAMPING_EXP_SCALE
OSC_BASE_KP = DEFAULT_KP
OSC_BASE_DAMPING_RATIO = DEFAULT_DAMPING_RATIO

# Joint-space impedance, used by JOINT_POS and home(); no sim counterpart to match.
JOINT_IMPEDANCE_KP = DEFAULT_JOINT_KP
HOME_IMPEDANCE_KP = DEFAULT_JOINT_KP
HOME_IMPEDANCE_KD = DEFAULT_JOINT_KD
HOME_MAX_QDOT = fc.control("homing.max_qdot_rad_s")      # ramp rate of the commanded home goal
HOME_SETTLE_QDOT = fc.control("homing.settle_qdot_rad_s")  # home() is not done until this still
HOME_LEAD_MARGIN = fc.control("homing.lead_margin")      # keeps the stall clamp off the ramp
# Fraction of each joint's torque clamp that homing may spend on kp*lead + kd*qdot
# together. Both terms scale with the speed, so this caps the speed per joint:
# the wrist (kd 30 against a 12 Nm clamp) cannot be damped at HOME_MAX_QDOT at
# all, and a saturated joint stops tracking the ramp and never converges.
HOME_TAU_FRACTION = fc.control("homing.tau_fraction")

JOINT_FEATURE_KEYS: tuple[str, ...] = (*(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)), "gripper")
EE_FEATURE_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")

_CAMERA_CTORS: dict[type, type] = {FramosCameraConfig: FramosCamera, ArvCameraConfig: ArvCamera}

logger = logging.getLogger(__name__)


def _make_camera(cfg: CameraConfig) -> Camera:
    cls = _CAMERA_CTORS.get(type(cfg))
    if cls is None:
        raise TypeError(f"Unsupported camera config: {type(cfg).__name__}")
    return cls(cfg)


class BimanualFranka(Robot):
    config_class = BimanualFrankaConfig
    name = "bimanual_franka"

    def __init__(self, config: BimanualFrankaConfig):
        super().__init__(config)
        self.config = config
        self.control_mode = config.control_mode
        self.active_arms = config.active_arms
        self.cameras: dict[str, Camera] = {n: _make_camera(c) for n, c in config.cameras.items()}
        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG | FrankaGripper] = {
            arm: self._make_gripper(arm) for arm in self.active_arms
        }
        # Robot base expressed in world (config/world.yaml): p_world = R @ p_base + t.
        # No inversion — the pose is already base-in-world, which is the direction
        # every consumer (safety brake, depth crop, viz, sysid) needs.
        self._base_in_world_by_arm = {
            arm: config.base_in_world(arm) for arm in self.active_arms
        }
        # The worktable brake compares world-frame heights, so it needs each
        # arm's base pose rather than one shared base-frame threshold, plus each
        # arm's EE collision sphere (grippers differ in size between arms).
        self._ee_sphere_by_arm = {
            arm: fc.ee_sphere(config.arm_name(arm)) for arm in self.active_arms
        }
        self.safety = ActionSafetyScreen(self._base_in_world_by_arm, self._ee_sphere_by_arm)
        # Populated by get_observation, consumed by next send_action to skip a redundant RPyC round-trip.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None
        self._cached_kin_ts: float = 0.0
        self._kin_cache_stale = 0
        self._kp_gain = 0.0
        self._kd_gain = 0.0
        self._gripper_accum: dict[str, float] = {arm: 1.0 for arm in self.active_arms}
        self._camera_pool = ThreadPoolExecutor(max_workers=max(len(self.cameras) + 1, 1))
        self._use_depth = bool(getattr(config, "depth", False))
        self._depth_cam: dict[str, Optional[Camera]] = dict()
        for s, cam in config.depth_cam.items():
            if s in self.cameras.keys():
                self._depth_cam[s] = None
                continue
            self._depth_cam[s] = _make_camera(cam)
        # Half-extent of the world-axis-aligned box crop (sim collect convention).
        self._depth_crop_radius_m = float(getattr(config, "depth_crop_radius_m", 0.4))

        depth_center = getattr(config, "depth_center_arm", self.active_arms[0])
        # Fall back when the profile's depth-centre arm isn't among active_arms.
        self._depth_center_arm = depth_center if depth_center in self.active_arms else self.active_arms[0]
        base_pose = self._base_in_world_by_arm[self._depth_center_arm]
        self._base_in_world = base_pose
        self._r_robot_in_world = base_pose.rotation
        self._t_robot_in_world = base_pose.translation
        # Residual offsets added on top of action commands via cache_delta().
        self.delta_pos = np.zeros(3)
        self.delta_rot = np.zeros(3)
        # Cropped and subsampled point cloud from the depth camera, cached each
        # get_observation() call.  None until the first observation is read.
        self._last_full_point_cloud: np.ndarray | None = None
        # robosuite's OSC nullspace reference (initial_joint); seeded in connect(),
        # re-anchored by home().
        self._home_q: dict[str, np.ndarray] = {}
        # Persistent EE_DELTA goal orientation, per arm. osc.py's set_goal only
        # rewrites goal_ori when the rotation delta is nonzero, so between
        # rotation commands this holds an absolute orientation -- that is what
        # makes the EE keep its orientation while translating. Reset by
        # _reset_osc_goal_ori(), robosuite's reset_goal().
        self._osc_goal_ori: dict[str, Rotation] = {}
        self._kp_ori_scale = np.asarray(getattr(config, "kp_ori_scale", 1.0), dtype=np.float64)
        self._kp_pos_scale = np.asarray(getattr(config, "kp_pos_scale", 1.0), dtype=np.float64)
        self._kd_ori_scale = np.asarray(getattr(config, "kd_ori_scale", 1.0), dtype=np.float64)
        self._kd_pos_scale = np.asarray(getattr(config, "kd_pos_scale", 1.0), dtype=np.float64)
        # Sim-to-real delta scaling, settable live so a sweep can search them.
        # Applied to the axis-angle rotation delta, NOT the quaternion: scaling
        # all four quaternion components uniformly is undone by normalisation.
        self._trans_fudge = float(getattr(config, "ee_translation_fudge", _EE_TRANSLATION_FUDGE_FACTOR))
        self._rot_fudge = float(getattr(config, "ee_rotation_fudge", _EE_ROTATION_FUDGE_FACTOR))


    def _make_gripper(self, arm: str) -> WSG | FrankaGripper:
        gripper_ip = getattr(self.config, f"{arm}_gripper_ip")
        if gripper_ip == getattr(self.config, f"{arm}_robot_ip"):
            return FrankaGripper(
                name=arm,
                server_ip=getattr(self.config, f"{arm}_server_ip"),
                robot_ip=getattr(self.config, f"{arm}_robot_ip"),
                # No fallback to {arm}_port: both configs resolve this from
                # arms.yaml, and silently using the ARM's port instead sends
                # gripper commands to the torque server, which just refuses
                # the connection somewhere far from the cause.
                port=getattr(self.config, f"{arm}_gripper_port"),
                do_print=False,
            )
        return WSG(name=arm, TCP_IP=gripper_ip, do_print=False)

    def _arm_features(self, keys: tuple[str, ...]) -> dict[str, type]:
        return {f"{arm}_{key}": float for arm in self.active_arms for key in keys}

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        out: dict[str, tuple[int, int, int]] = {}
        for n, cam in self.cameras.items():
            if cam.height is None or cam.width is None:
                raise RuntimeError(f"Camera '{n}' does not report height/width")
            out[n] = (int(cam.height), int(cam.width), IMAGE_CHANNELS)
        return out

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        # The depth cloud is not in here: it is an array, reached through
        # last_full_point_cloud, not 6144 scalar observation entries.
        return {**self._arm_features(JOINT_FEATURE_KEYS), **self._camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        keys = JOINT_FEATURE_KEYS if self.control_mode == ControlMode.JOINT_POS else EE_FEATURE_KEYS
        d = self._arm_features(keys)
        d["kp"] = float
        d["kd"] = float
        return d

    @property
    def is_connected(self) -> bool:
        return self.robot_manager.num_alive == len(self.active_arms)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        use_ee = self.control_mode != ControlMode.JOINT_POS
        try:
            for n, cam in self.cameras.items():
                try:
                    cam.connect()
                except Exception as e:
                    logger.warning("Camera %s failed to connect: %s", n, e)
            if self._use_depth and len(self._depth_cam) > 0:
                for s, cam in self._depth_cam.items():
                    try:
                        if cam is None:
                            continue
                        cam.connect()
                    except Exception as e:
                        logger.warning("Camera %s failed to connect: %s", s, e)
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    getattr(self.config, f"{arm}_server_ip"),
                    getattr(self.config, f"{arm}_robot_ip"),
                    getattr(self.config, f"{arm}_port"),
                    use_ee_delta=use_ee,
                )
                snap = self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
                # Seed the nullspace reference before any goal is pushed, the way
                # robosuite's Controller.__init__ captures initial_joint.
                self._home_q[arm] = np.asarray(snap[0], dtype=np.float64).copy()
                self._osc_goal_ori[arm] = Rotation.from_quat(np.asarray(snap[4], dtype=np.float64))
            # ALWAYS push, even at the default. Server sessions are keyed by
            # robot_ip and outlive the client, so an assist set by an earlier
            # script (a probe run with --friction-kc, say) otherwise silently
            # persists into the next run, and a sysid sweep would measure a
            # controller nobody configured.
            self.robot_manager.set_tuning_all(friction_kc=self.friction_kc)
            for arm in self.active_arms:
                self.grippers[arm].home()
        except Exception:
            self.robot_manager.shutdown()
            raise

    def disconnect(self) -> None:
        self._camera_pool.shutdown(wait=False)
        self._cached_kin_state = None
        for cam in self.cameras.values():
            cam.disconnect()
        self.robot_manager.shutdown()
        for g in self.grippers.values():
            g.close()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Arm state FIRST: the depth crop needs ee_world at submit time, and that
        # is what lets each camera's cloud compute chain onto its own frame
        # instead of waiting for every camera in the rig to finish reading.
        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        kin = {arm: self._patch_jacobian(snap) for arm, snap in kin.items()}
        self._cached_kin_state = kin
        self._cached_kin_ts = time.perf_counter()
        ee_world = self._ee_world_center(kin)

        depth_cams: dict[str, Camera] = {}
        if self._use_depth:
            for name, cam in self._depth_cam.items():
                c = self.cameras.get(name) if cam is None else cam
                if c is None:
                    raise KeyError(f"Depth camera {name!r} not found in cameras")
                depth_cams[name] = c
            if not depth_cams:
                raise KeyError(f"Depth camera {self._depth_cam!r} not found in cameras")
        cloud_points = _DEPTH_POINT_COUNT // max(len(depth_cams), 1)

        def _read(cam: Camera, with_cloud: bool):
            img = cam.async_read(_CAMERA_READ_TIMEOUT_MS)
            if not with_cloud:
                return img, None
            return img, cam.get_cropped_point_cloud(
                ee_world, self._depth_crop_radius_m, cloud_points
            )

        # A camera in both dicts is the same object, so it is read once and its
        # cloud is derived from that same frame.
        futs = {
            name: self._camera_pool.submit(_read, cam, name in depth_cams)
            for name, cam in {**self.cameras, **depth_cams}.items()
        }

        obs: RobotObservation = {}

        for arm in self.active_arms:
            for i, qi in enumerate(kin[arm][0]):
                obs[f"{arm}_joint_{i + 1}"] = float(qi)
            pos = self.grippers[arm].position
            max_mm = self.grippers[arm].GRIPPER_TRUE_MAX_MM
            obs[f"{arm}_gripper"] = (0 if pos is None else pos) / max_mm

        clouds: dict[str, np.ndarray] = {}
        for name, fut in futs.items():
            try:
                img, cloud = fut.result()
            except Exception as e:
                logger.warning("Camera %s read failed: %s", name, e)
                img, cloud = None, None
            if name in self.cameras:
                if img is None:
                    blank = getattr(self.cameras[name], "blank_frame", None)
                    img = (blank() if callable(blank)
                           else np.zeros(self._camera_features[name], dtype=np.uint8))
                obs[name] = img
            if name in depth_cams:
                clouds[name] = (cloud if cloud is not None
                                else np.zeros((cloud_points, 3), dtype=np.float32))

        if self._use_depth:
            # Concatenated in _depth_cam order so the cloud's per-camera layout is
            # stable across calls. Exposed as an array via last_full_point_cloud,
            # never as scalar obs entries -- flattening it to 6144 float keys and
            # rebuilding it cost ~1.1 ms/step for nothing.
            self._last_full_point_cloud = np.concatenate(
                [clouds[name] for name in depth_cams], axis=0
            )
        return obs

    def _ee_world_center(self, kin: dict[str, KinematicSnapshot]) -> np.ndarray:
        arm = self._depth_center_arm
        return self._base_in_world.apply(np.asarray(kin[arm][3], dtype=np.float64))

    @property
    def base_in_world(self):
        """`franka_config.Pose` mapping robot-base coordinates into world."""
        return self._base_in_world

    def _sample_depth_points(self, verts: np.ndarray, center: np.ndarray) -> np.ndarray:
        points = np.asarray(verts, dtype=np.float64)
        if points.size == 0:
            return np.zeros((_DEPTH_POINT_COUNT, 3), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"expected (N, 3) or (N, 3+C) point cloud, got shape {points.shape}"
            )
        points = points[:, :3]

        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] == 0:
            return np.zeros((_DEPTH_POINT_COUNT, 3), dtype=np.float32)

        deltas = points - center.reshape(1, 3)
        dist2 = np.einsum("ij,ij->i", deltas, deltas)
        cropped = points[dist2 <= (self._depth_crop_radius_m ** 2)]
        total = cropped.shape[0]

        if total == 0:
            sampled = np.zeros((_DEPTH_POINT_COUNT, 3), dtype=np.float64)
        else:
            # idx = np.linspace(0, cropped.shape[0] - 1, _DEPTH_POINT_COUNT, dtype=np.int64)
            idx = np.random.choice(np.arange(0,total), size=_DEPTH_POINT_COUNT, replace=False)
            sampled = cropped[idx]
        # else:
        #     reps = (_DEPTH_POINT_COUNT + cropped.shape[0] - 1) // cropped.shape[0]
        #     sampled = np.tile(cropped, (reps, 1))[:_DEPTH_POINT_COUNT]

        return np.asarray(sampled, dtype=np.float32)

    @property
    def friction_kc(self) -> np.ndarray:
        """Coulomb assist scale as (2, 7): the scalar times the per-joint trim,
        split by rotation direction. Row 0 applies where the commanded torque on
        that joint is positive, row 1 where it is negative -- breakaway on this
        arm is directional, so one symmetric gain either under-assists the hard
        direction or over-assists the easy one."""
        scale = float(getattr(self.config, "friction_kc", 0.0))
        default = (1.0,) * NUM_JOINTS
        return scale * np.stack([
            np.asarray(getattr(self.config, "friction_kc_joint_pos", default), dtype=np.float64),
            np.asarray(getattr(self.config, "friction_kc_joint_neg", default), dtype=np.float64),
        ])

    def set_friction_kc(self, kc) -> None:
        """Push a new assist scale to the running control loops.

        Scalar or (7,) sets both directions; (2, 7) sets them independently.
        """
        kc = np.asarray(kc, dtype=np.float64)
        if kc.size != 2 * NUM_JOINTS:
            kc = np.broadcast_to(kc, (NUM_JOINTS,))
            kc = np.stack([kc, kc])
        self.robot_manager.set_tuning_all(friction_kc=kc.reshape(2, NUM_JOINTS))

    @property
    def kp_gain(self) -> float:
        return self._kp_gain
    
    @property
    def kd_gain(self) -> float:
        return self._kd_gain
    
    @property
    def kin(self) -> Optional[dict[str, KinematicSnapshot]]:
        return self._cached_kin_state
    
    @staticmethod
    def _patch_jacobian(snap: KinematicSnapshot) -> KinematicSnapshot:
        """Replace the driver's Jacobian with the analytic one, anchored on the
        measured EE position. Originally a workaround for franky's all-zero
        zero_jacobian; kept under pylibfranka because this is the convention the
        downstream proprio/twist consumers (residual_wrapper) were validated
        against. pylibfranka's Model.zero_jacobian is all-zero on this build too,
        so the server-side OSC law computes this same analytic Jacobian from the
        same q and measured EE point."""
        q, dq, J, ee_pos, ee_quat_xyzw, ee_twist = snap
        q = np.asarray(q, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        J_real = zero_jacobian(q, ee_pos_base=ee_pos)
        return (q, dq, J_real, ee_pos, ee_quat_xyzw, ee_twist)

    def send_action(self, action: RobotAction, ignore_action: bool = False) -> RobotAction:
        """Push this policy step's goal to the arms' 1 kHz torque loops.

        Mirrors robosuite's split: this is ``set_goal`` (once per policy step),
        while ``run_controller`` runs server-side every tick. All three control
        modes land on torque -- EE_DELTA/EE_POS via OSC, JOINT_POS via joint
        impedance -- since pylibfranka exposes no Cartesian-velocity interface.
        """
        kin = self._cached_kin_state
        if kin is not None and time.perf_counter() - self._cached_kin_ts > _KIN_CACHE_MAX_AGE_S:
            self._kin_cache_stale += 1
            kin = None
        if kin is None:
            kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = None
        kin = {arm: self._patch_jacobian(snap) for arm, snap in kin.items()}
        self._kp_gain = _KP_GAIN_BASE ** float(np.clip(action["kp"], -1.0, 1.0))
        self._kd_gain = _KD_GAIN_BASE ** float(np.clip(action["kd"], -1.0, 1.0))

        for arm in self.active_arms:
            self._gripper_accum[arm] = np.clip(self._gripper_accum[arm] + (action[f"{arm}_gripper"]) * _GRIP_ACCUM_SPEED, -1.0, 1.0)
            self.grippers[arm].move(
                (self._gripper_accum[arm] + 1.0) / 2.0 * self.grippers[arm].GRIPPER_TRUE_MAX_MM,
                blocking=False,
            )

        if self.control_mode == ControlMode.JOINT_POS:
            # Not screened: the worktable floor is a bound on an EE goal pose,
            # and a joint-position command has none. JOINT_POS is GELLO teleop
            # with an operator in the loop.
            goals = {arm: self._joint_goal(action, arm) for arm in self.active_arms}
            self.robot_manager.move_joint_goal_batch(
                {a: (g, self.kp_gain, self.kd_gain) for a, g in goals.items()}
            )
            return action

        kp, kd = resolve_gains(action["kp"], action["kd"],
                               self._kp_ori_scale, self._kd_ori_scale,
                               kp_pos_scale=self._kp_pos_scale,
                               kd_pos_scale=self._kd_pos_scale)

        if self.control_mode == ControlMode.EE_DELTA:
            goals = {
                arm: self._osc_goal_delta(
                    arm, action, kin[arm], self.delta_pos, self.delta_rot,
                    self.config.use_noise, self.config.noise_pos_scale, self.config.noise_rot_scale,
                )
                for arm in self.active_arms
            }
        else:
            goals = {
                arm: self._osc_goal_absolute(
                    arm, action, kin[arm], self.delta_pos, self.delta_rot, ignore_action
                )
                for arm in self.active_arms
            }

        goals = self.safety.shape_goal(goals)
        self.robot_manager.move_osc_goal_batch(
            {a: (pos, quat, kp, kd, self._home_q.get(a)) for a, (pos, quat) in goals.items()}
        )
        return action

    def home(
        self,
        home_q_left: np.ndarray | None,
        home_q_right: np.ndarray | None,
        gripper_norm: float = fc.control("homing.gripper_norm"),
        max_time_s: float = fc.control("homing.max_time_s"),
        tol_rad: float = fc.control("homing.tol_rad"),
        fps: int = fc.control_fps(),
        *,
        home_fps: int | None = None,
        **_unused,
    ) -> bool:
        """Drive both arms to a saved home configuration.

        ``home_q_*`` is the desired ``q`` in every control mode: homing always
        runs server-side joint impedance, the only law that reaches a joint
        configuration directly. The commanded goal leads the measured ``q`` by at
        most ``max_lead`` rather than jumping to the target, which bounds the
        approach speed at ``HOME_MAX_QDOT`` and lets it taper to zero on arrival
        instead of overshooting.

        On success the target also becomes the OSC nullspace reference.
        Convergence is judged in joint space against ``tol_rad``; ``**_unused``
        swallows the EE-space tolerances older call sites still pass.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        candidates = {"l": home_q_left, "r": home_q_right}
        targets_q = {
            arm: np.asarray(q, dtype=np.float64)
            for arm, q in candidates.items()
            if q is not None and arm in self.active_arms
        }
        if not targets_q:
            return True

        for arm in targets_q:
            self.grippers[arm].move(gripper_norm * self.grippers[arm].GRIPPER_TRUE_MAX_MM, blocking=False)

        use_ee_homing = False # self.control_mode != ControlMode.JOINT_POS
        rate_hz = float(home_fps if home_fps is not None else fc.home_fps())
        period_s = 1.0 / rate_hz
        deadline = time.perf_counter() + max_time_s
        names = list(targets_q)
        # Joint impedance settles at v = (kp/kd) * error, so this lead caps speed.
        # Joint impedance settles at v = (kp/kd)*error per joint, so the fastest
        # joint sets the lead that keeps every joint under HOME_MAX_QDOT.
        # RAMP the goal; re-deriving it from the measured q each tick sawtooths
        # the error by v/rate, a 25% torque ripple felt as vibration.
        # kp*lead + kd*qdot = qdot*kd*(1 + margin), so the budget fixes the speed.
        qdot_max = np.minimum(
            HOME_MAX_QDOT,
            HOME_TAU_FRACTION * np.asarray(JOINT_TORQUE_LIMITS)
            / (HOME_IMPEDANCE_KD * (1.0 + HOME_LEAD_MARGIN)))
        step = qdot_max / rate_hz
        # Stall guard, not the speed limit (the ramp is). Sustaining HOME_MAX_QDOT
        # needs exactly HOME_MAX_QDOT/(kp/kd) of lead, so this must sit above it
        # with margin or it binds every tick and the sawtooth comes back.
        max_lead = HOME_LEAD_MARGIN * qdot_max * HOME_IMPEDANCE_KD / HOME_IMPEDANCE_KP
        ramp = {arm: np.asarray(snap[0], dtype=np.float64).copy()
                for arm, snap in self.robot_manager.current_kinematic_state_batch(names).items()}

        while True:
            tick_start = time.perf_counter()
            kin = self.robot_manager.current_kinematic_state_batch(names)

            commanded = {}
            for arm in names:
                lead = ramp[arm] + np.clip(targets_q[arm] - ramp[arm], -step, step)
                # Re-anchors only once the arm has fallen behind.
                ramp[arm] = kin[arm][0] + np.clip(lead - kin[arm][0], -max_lead, max_lead)
                commanded[arm] = ramp[arm]
            # Not screened: a home configuration is a saved, known-safe q, and
            # the worktable floor only bounds an EE goal pose.
            self.robot_manager.move_joint_goal_batch(
                {a: (g, 1.0, 1.0) for a, g in commanded.items()}
            )

            max_err = max(float(np.max(np.abs(targets_q[arm] - kin[arm][0]))) for arm in names)
            # Exit at rest, not merely in position: returning mid-motion leaves
            # the arm coasting into whatever the caller does next.
            max_qdot = max(float(np.max(np.abs(kin[arm][1]))) for arm in names)
            if max_err < tol_rad and max_qdot < HOME_SETTLE_QDOT:
                self._cached_kin_state = None
                for arm in names:
                    self._home_q[arm] = targets_q[arm].copy()
                self._reset_osc_goal_ori(kin)
                return True
            if tick_start >= deadline:
                self._cached_kin_state = None
                logger.warning("home(): timeout after %.2fs, max joint error %.4f rad", max_time_s, max_err)
                self._reset_osc_goal_ori(kin)
                return False

            elapsed = time.perf_counter() - tick_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    def _reset_osc_goal_ori(self, kin: dict[str, KinematicSnapshot]) -> None:
        """robosuite reset_goal(): park the held orientation on the current pose,
        so post-home EE_DELTA offsets are relative to where the arm actually is."""
        for arm, snap in kin.items():
            self._osc_goal_ori[arm] = Rotation.from_quat(np.asarray(snap[4], dtype=np.float64))

    @staticmethod
    def _joint_goal(action: RobotAction, arm: str) -> np.ndarray:
        return np.fromiter(
            (action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)),
            dtype=np.float64, count=NUM_JOINTS,
        )

    @staticmethod
    def _delta_rotvec(dquat_xyzw: np.ndarray) -> np.ndarray:
        """Delta quaternion (xyzw) -> axis-angle, the representation osc.py's
        set_goal_orientation takes. A degenerate all-zero quat means no rotation."""
        norm = float(np.linalg.norm(dquat_xyzw))
        if norm < 1e-9:
            return np.zeros(3)
        return Rotation.from_quat(dquat_xyzw / norm).as_rotvec()

    def _osc_goal_delta(
        self,
        arm: str,
        action: RobotAction,
        snap: KinematicSnapshot,
        dpos_cached: np.ndarray,
        drot_cached: np.ndarray,
        use_noise: bool,
        noise_pos_scale: float,
        noise_rot_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """osc.py set_goal() with use_delta=True.

        The goal is rebuilt from the *current* EE pose every policy step, never
        accumulated onto the previous goal -- that is what makes a released
        joystick or a zero-delta policy step hold position instead of drifting.
        Residual offsets from cache_delta() and the config noise are summed in
        axis-angle space, matching how osc.py carries the orientation delta.
        """
        _, _, _, ee_pos, ee_quat_xyzw, _ = snap

        dpos = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("x", "y", "z")), dtype=np.float64, count=3
        ) + np.asarray(dpos_cached, dtype=np.float64)
        drot = self._delta_rotvec(
            np.fromiter((action[f"{arm}_{ax}"] for ax in ("qx", "qy", "qz", "qw")),
                        dtype=np.float64, count=4)
        ) + np.asarray(drot_cached, dtype=np.float64)

        if use_noise:
            dpos = dpos + np.random.normal(0.0, noise_pos_scale, 3)
            drot = drot + Rotation.from_euler(
                "xyz", np.random.normal(0.0, noise_rot_scale, 3)
            ).as_rotvec()

        # Clip to the envelope a policy could have emitted, THEN apply the
        # hardware fudge. The other order lets clip_delta eat the fudge -- at
        # tf=3 a 0.05 m delta became 0.15 m and was clipped straight back to
        # 0.05, so any fudge above 1.0 (2.0 for rotation) was a silent no-op.
        dpos, drot = clip_delta(dpos, drot)
        dpos = dpos * self._trans_fudge
        drot = drot * self._rot_fudge

        # osc.py updates goal_ori ONLY when the rotation delta is nonzero, and
        # tests it with math.isclose(elem, 0.0) -- exact zero. Re-anchoring it
        # unconditionally makes the orientation error identically zero on a
        # pure-translation command, so nothing holds the EE's orientation and it
        # tumbles as the arm translates. goal_pos, in contrast, IS rebuilt from
        # the current pose every step.
        if arm not in self._osc_goal_ori or np.any(drot != 0.0):
            self._osc_goal_ori[arm] = Rotation.from_rotvec(drot) * Rotation.from_quat(ee_quat_xyzw)
        return np.asarray(ee_pos, dtype=np.float64) + dpos, self._osc_goal_ori[arm].as_quat()

    def _osc_goal_absolute(
        self,
        arm: str,
        action: RobotAction,
        snap: KinematicSnapshot,
        dpos_cached: np.ndarray,
        drot_cached: np.ndarray,
        ignore_action: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """EE_POS: the action already carries an absolute pose, so it becomes the
        OSC goal directly. ``ignore_action`` parks the goal on the current pose so
        only the cache_delta() residual moves the arm."""
        _, _, _, ee_pos, ee_quat_xyzw, _ = snap
        if ignore_action:
            goal_pos = np.asarray(ee_pos, dtype=np.float64)
            goal_rot = Rotation.from_quat(ee_quat_xyzw)
        else:
            target = np.fromiter(
                (action[f"{arm}_{ax}"] for ax in EE_AXIS_KEYS),
                dtype=np.float64, count=len(EE_AXIS_KEYS),
            )
            goal_pos = target[:3]
            goal_rot = Rotation.from_quat(target[3:] / max(float(np.linalg.norm(target[3:])), 1e-12))

        goal_rot = Rotation.from_rotvec(np.asarray(drot_cached, dtype=np.float64)) * goal_rot
        return goal_pos + np.asarray(dpos_cached, dtype=np.float64), goal_rot.as_quat()

    def cache_delta(self, dpos: np.ndarray, drot: np.ndarray) -> None:
        self.delta_pos = dpos
        self.delta_rot = drot

    @property
    def last_full_point_cloud(self) -> np.ndarray | None:
        """Cropped and subsampled world-space point cloud from the depth camera.

        Updated every get_observation() call when depth is enabled.
        Shape: (N, 3) float32 in world-frame metres, or None before the first observation.
        """
        return self._last_full_point_cloud

