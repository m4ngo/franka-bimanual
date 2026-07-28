import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import cast, Optional

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
    DEFAULT_JOINT_KD,
    DEFAULT_JOINT_KP,
    DEFAULT_KP,
    KP_EXP_SCALE,
    DAMPING_EXP_SCALE,
    clip_delta,
    resolve_gains,
)
from .franka_jacobian import zero_jacobian  # the new analytic module

IMAGE_CHANNELS = 3
_CAMERA_READ_TIMEOUT_MS: float = 5.0
_CONNECT_TIMEOUT_S = 10.0
_DEPTH_POINT_COUNT = 2048

# Exponential action->gain remap, matching the sim wrapper the policies were
# trained against (utils/envs/libero.py: exp_scale = limit_max / default).
# kp_gain/kd_gain are the multipliers; OSC_BASE_KP/OSC_BASE_DAMPING_RATIO are
# the robosuite defaults they multiply.
_KP_GAIN_BASE = KP_EXP_SCALE
_KD_GAIN_BASE = DAMPING_EXP_SCALE
OSC_BASE_KP = DEFAULT_KP
OSC_BASE_DAMPING_RATIO = 1.0

# Joint-space impedance, used by JOINT_POS and home(); no sim counterpart to match.
JOINT_IMPEDANCE_KP = DEFAULT_JOINT_KP
HOME_IMPEDANCE_KP = DEFAULT_JOINT_KP
HOME_IMPEDANCE_KD = DEFAULT_JOINT_KD
HOME_MAX_QDOT = 0.6  # rad/s, ramp rate of the commanded home goal

_GRIP_ACCUM_SPEED = 1.0

# Parity knobs on the incoming delta action; 1.0 = exactly what the policy emits.
_EE_TRANSLATION_FUDGE_FACTOR = 1.0
_EE_ROTATION_FUDGE_FACTOR = 1.0

JOINT_FEATURE_KEYS: tuple[str, ...] = (*(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)), "gripper")
EE_FEATURE_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")

_CAMERA_CTORS: dict[type, type] = {FramosCameraConfig: FramosCamera, ArvCameraConfig: ArvCamera}

_DEPTH_POINT_AXES: tuple[str, ...] = ("x", "y", "z")
_DEPTH_FLAT_SIZE: int = _DEPTH_POINT_COUNT * len(_DEPTH_POINT_AXES)  # 6144

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
        self.safety = ActionSafetyScreen()
        # Populated by get_observation, consumed by next send_action to skip a redundant RPyC round-trip.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None
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

        world_in_robot_quat = getattr(config, "world_in_robot_quat_wxyz", (1.0, 0.0, 0.0, 0.0))
        world_in_robot_translation = getattr(config, "world_in_robot_translation_m", (0.0, 0.0, 0.0))
        r_w_in_r = self._quat_wxyz_to_rot(world_in_robot_quat)
        t_w_in_r = np.asarray(world_in_robot_translation, dtype=np.float64)
        # Invert world-in-robot pose to map robot-frame EE positions into world frame.
        self._r_robot_in_world = r_w_in_r.T
        self._t_robot_in_world = -self._r_robot_in_world @ t_w_in_r
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
                port=getattr(self.config, f"{arm}_gripper_port",
                             getattr(self.config, f"{arm}_port")),
                do_print=False,
            )
        return WSG(name=arm, TCP_IP=gripper_ip, do_print=False)

    def _arm_features(self, keys: tuple[str, ...]) -> dict[str, type]:
        return {f"{arm}_{key}": float for arm in self.active_arms for key in keys}

    def _depth_features(self) -> dict[str, type]:
        return {f"depth_{i}": float for i in range(_DEPTH_FLAT_SIZE)}

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
        if self._use_depth:
            return {**self._arm_features(JOINT_FEATURE_KEYS), **self._camera_features, **self._depth_features()}
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
            # ALWAYS push, even at defaults. Server sessions are keyed by robot_ip
            # and outlive the client, so tuning set by an earlier script (a probe
            # run with --friction-kc, say) otherwise silently persists into the
            # next run. The config must be the single source of truth or a sysid
            # sweep can measure a controller nobody configured.
            self.robot_manager.set_tuning_all(
                friction_kc=float(getattr(self.config, "friction_kc", 0.0)),
                uncouple_pos_ori=bool(getattr(self.config, "uncouple_pos_ori", True)),
            )
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

        cam_futs = {
            n: self._camera_pool.submit(cam.async_read, _CAMERA_READ_TIMEOUT_MS)
            for n, cam in self.cameras.items()
        }

        standalone_depth_cam = []
        depth_color_fut = []
        if self._use_depth:
            for s, cam in self._depth_cam.items():
                if cam is None:
                    continue
                standalone_depth_cam.append(cam)
                depth_color_fut.append(self._camera_pool.submit(cam.async_read, _CAMERA_READ_TIMEOUT_MS))

        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        kin = {arm: self._patch_jacobian(snap) for arm, snap in kin.items()}
        self._cached_kin_state = kin
        ee_world = self._ee_world_center(kin)

        obs: RobotObservation = {}

        for arm in self.active_arms:
            for i, qi in enumerate(kin[arm][0]):
                obs[f"{arm}_joint_{i + 1}"] = float(qi)
            pos = self.grippers[arm].position
            max_mm = self.grippers[arm].GRIPPER_TRUE_MAX_MM
            obs[f"{arm}_gripper"] = (0 if pos is None else pos) / max_mm

        for n, fut in cam_futs.items():
            try:
                obs[n] = fut.result()
            except Exception as e:
                logger.warning("Camera %s read failed: %s", n, e)
                blank = getattr(self.cameras[n], "blank_frame", None)
                obs[n] = blank() if callable(blank) else np.zeros(self._camera_features[n], dtype=np.uint8)

        if len(standalone_depth_cam) > 0:
            try:
                for fut in depth_color_fut:
                    fut.result()  # prime the buffer; result unused, not part of obs
            except Exception as e:
                logger.warning("Standalone depth camera color read failed: %s", e)

        if self._use_depth:
            depth_cams = []
            for s, cam in self._depth_cam.items():
                if cam is None:
                    depth_cams.append(self.cameras.get(s))
                else:
                    depth_cams.append(cam)
            if len(depth_cams) <= 0:
                raise KeyError(f"Depth camera {self._depth_cam!r} not found in cameras")

            depth_futs = []
            clouds: list[np.ndarray] = []
            for c in depth_cams:
                depth_futs.append(self._camera_pool.submit(
                    c.get_cropped_point_cloud, ee_world, self._depth_crop_radius_m, _DEPTH_POINT_COUNT // len(depth_cams)
                ))
                
            for fut in depth_futs:
                clouds.append(fut.result())

            self._last_full_point_cloud = np.concatenate(clouds,axis=0) # self._sample_depth_points(np.concatenate(clouds,axis=0), ee_world)
            # print(self._last_full_point_cloud)
            flat = self._last_full_point_cloud.reshape(-1).astype(np.float64)
            # print(flat.shape)
            obs.update(zip((f"depth_{i}" for i in range(_DEPTH_FLAT_SIZE)), flat.tolist()))
        return obs

    def _ee_world_center(self, kin: dict[str, KinematicSnapshot]) -> np.ndarray:
        arm = "r" if "r" in self.active_arms else self.active_arms[0]
        ee_robot = np.asarray(kin[arm][3], dtype=np.float64)
        return self._r_robot_in_world @ ee_robot + self._t_robot_in_world

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

    @staticmethod
    def _quat_wxyz_to_rot(q: tuple[float, float, float, float]) -> np.ndarray:
        w, x, y, z = q
        n = float(np.sqrt(w * w + x * x + y * y + z * z))
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    
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
        against. The server-side OSC law uses libfranka's own Jacobian, which is
        self-consistent with the O_T_EE it forms the pose error from."""
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
        kin = self._cached_kin_state or self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
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
            goals = self.safety.shape_joint_goal(
                {arm: self._joint_goal(action, arm) for arm in self.active_arms},
                kin,
                float(np.max(JOINT_IMPEDANCE_KP)) * self.kp_gain,
                self.kd_gain,
            )
            self.robot_manager.move_joint_goal_batch(
                {a: (g, self.kp_gain, self.kd_gain) for a, g in goals.items()}
            )
            return action

        kp, kd = resolve_gains(action["kp"], action["kd"], self._kp_ori_scale)

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

        goals = self.safety.shape_goal(goals, kin, kp, kd)
        self.robot_manager.move_osc_goal_batch(
            {a: (pos, quat, kp, kd, self._home_q.get(a)) for a, (pos, quat) in goals.items()}
        )
        return action

    def home(
        self,
        home_q_left: np.ndarray | None,
        home_q_right: np.ndarray | None,
        gripper_norm: float = 1.0,
        max_time_s: float = 5.0,
        tol_rad: float = 0.05,
        fps: int = 30,
        *,
        home_fps: int | None = None,
        tol_pos_m: float = 0.025,
        tol_rot_rad: float | None = None,
    ) -> bool:
        """Drive both arms to a saved home configuration.

        ``home_q_*`` is the desired ``q`` in every control mode: homing always
        runs server-side joint impedance, the only law that reaches a joint
        configuration directly. The commanded goal leads the measured ``q`` by at
        most ``max_lead`` rather than jumping to the target, which bounds the
        approach speed at ``HOME_MAX_QDOT`` and lets it taper to zero on arrival
        instead of overshooting.

        On success the target also becomes the OSC nullspace reference.
        ``tol_pos_m`` / ``tol_rot_rad`` are accepted for call-site compatibility
        and unused; convergence is judged in joint space against ``tol_rad``.
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

        rate_hz = float(home_fps if home_fps is not None else fps)
        period_s = 1.0 / rate_hz
        deadline = time.perf_counter() + max_time_s
        names = list(targets_q)
        # Joint impedance settles at v = (kp/kd) * error, so this lead caps speed.
        # Joint impedance settles at v = (kp/kd)*error per joint, so the fastest
        # joint sets the lead that keeps every joint under HOME_MAX_QDOT.
        max_lead = HOME_MAX_QDOT / float(np.max(HOME_IMPEDANCE_KP / HOME_IMPEDANCE_KD))

        while True:
            tick_start = time.perf_counter()
            kin = self.robot_manager.current_kinematic_state_batch(names)

            commanded = {
                arm: kin[arm][0] + np.clip(targets_q[arm] - kin[arm][0], -max_lead, max_lead)
                for arm in names
            }
            goals = self.safety.shape_joint_goal(commanded, kin, float(np.max(HOME_IMPEDANCE_KP)))
            self.robot_manager.move_joint_goal_batch(
                {a: (g, 1.0, 1.0) for a, g in goals.items()}
            )

            max_err = max(float(np.max(np.abs(targets_q[arm] - kin[arm][0]))) for arm in names)
            if max_err < tol_rad:
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

