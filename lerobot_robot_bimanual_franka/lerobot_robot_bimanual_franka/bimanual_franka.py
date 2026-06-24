import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from lerobot_camera_arv import ArvCamera, ArvCameraConfig
from lerobot_camera_framos import FramosCamera, FramosCameraConfig

from .bimanual_franka_config import BimanualFrankaConfig, ControlMode
from .franka_gripper import FrankaGripper
from .franka_fk import franka_fk
from .franka_process import NUM_JOINTS, KinematicSnapshot, MultiRobotWrapper
from .safety import ActionSafetyScreen
from .wsg import WSG

IMAGE_CHANNELS = 3
_CAMERA_READ_TIMEOUT_MS: float = 5.0
_CONNECT_TIMEOUT_S = 10.0
_DEPTH_POINT_COUNT = 2048

# Joint-mode PD gains used in home() (rad/s per rad error)
JOINT_PD_KP = 2.0
JOINT_PD_KD = 0.1

JOINT_FEATURE_KEYS: tuple[str, ...] = (*(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)), "gripper")
EE_FEATURE_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")

_CAMERA_CTORS: dict[type, type] = {FramosCameraConfig: FramosCamera, ArvCameraConfig: ArvCamera}

_DEPTH_POINT_AXES: tuple[str, ...] = ("x", "y", "z")
_DEPTH_FLAT_SIZE: int = _DEPTH_POINT_COUNT * len(_DEPTH_POINT_AXES)  # 6144
_FULL_PCD_CROP_RADIUS_M: float = 0.5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math helpers (OSC-style)
# ---------------------------------------------------------------------------

def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (xyzw) → 3×3 rotation matrix."""
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 1.0 / n
    return np.array([
        [1.0 - 2*s*(y*y + z*z),  2*s*(x*y - z*w),        2*s*(x*z + y*w)],
        [2*s*(x*y + z*w),        1.0 - 2*s*(x*x + z*z),  2*s*(y*z - x*w)],
        [2*s*(x*z - y*w),        2*s*(y*z + x*w),        1.0 - 2*s*(x*x + y*y)],
    ], dtype=np.float64)


def _orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Axis-angle orientation error from 3×3 rotation matrices.

    Matches robosuite OSC orientation_error: 0.5 * sum(cross(rc_i, rd_i)).
    """
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0]) +
        np.cross(current[:, 1], desired[:, 1]) +
        np.cross(current[:, 2], desired[:, 2])
    )


def _quat_xyzw_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (xyzw) → axis-angle (3D vector, magnitude = angle in rad)."""
    v, w = q[:3], float(np.clip(q[3], -1.0, 1.0))
    v_norm = float(np.linalg.norm(v))
    return 2.0 * v if v_norm < 1e-9 else (v / v_norm) * (2.0 * np.arctan2(v_norm, w))


def _axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3D, magnitude = angle) → 3×3 rotation matrix."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    s, c = np.sin(angle), np.cos(angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)


def _osc_matrices(
    J: np.ndarray,
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute dynamically-consistent pseudoinverses and null-space projector.

    Returns (J_bar_pos, J_bar_ori, J_bar_full, N) where:
      J_bar_* = M⁻¹ J*ᵀ λ*  (dynamically-consistent pseudoinverse)
      N       = I − J̄·J      (null-space projector)
    """
    J_pos, J_ori = J[:3], J[3:]
    M_inv = np.linalg.inv(M)

    lambda_full = np.linalg.pinv(J @ M_inv @ J.T)
    lambda_pos  = np.linalg.pinv(J_pos @ M_inv @ J_pos.T)
    lambda_ori  = np.linalg.pinv(J_ori @ M_inv @ J_ori.T)

    J_bar_full = M_inv @ J.T     @ lambda_full   # (7, 6)
    J_bar_pos  = M_inv @ J_pos.T @ lambda_pos    # (7, 3)
    J_bar_ori  = M_inv @ J_ori.T @ lambda_ori    # (7, 3)

    N = np.eye(NUM_JOINTS) - J_bar_full @ J      # (7, 7)
    return J_bar_pos, J_bar_ori, J_bar_full, N


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

        # Populated by get_observation, consumed by next send_action.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None
        self._camera_pool = ThreadPoolExecutor(max_workers=max(len(self.cameras) + 1, 1))
        self._use_depth = bool(getattr(config, "depth", False))
        self._depth_cam = str(getattr(config, "depth_cam", ""))
        self._depth_crop_radius_m = float(getattr(config, "depth_crop_radius_m", 0.4))

        world_in_robot_quat = getattr(config, "world_in_robot_quat_wxyz", (1.0, 0.0, 0.0, 0.0))
        world_in_robot_translation = getattr(config, "world_in_robot_translation_m", (0.0, 0.0, 0.0))
        r_w_in_r = self._quat_wxyz_to_rot(world_in_robot_quat)
        t_w_in_r = np.asarray(world_in_robot_translation, dtype=np.float64)
        self._r_robot_in_world = r_w_in_r.T
        self._t_robot_in_world = -self._r_robot_in_world @ t_w_in_r

        # Residual offsets added on top of action commands via cache_delta().
        self.delta_pos = np.zeros(3)
        self.delta_rot = np.zeros(3)

        self._last_full_point_cloud: np.ndarray | None = None

        # VOsc per-arm state (initialized at connect time)
        self._goal_ori_mat: dict[str, np.ndarray] = {}   # held goal orientation (rotation matrix)
        self._q0: dict[str, np.ndarray] = {}             # nullspace reference joint config

        # OSC gains from config
        self._kp_base = float(config.osc_kp_base)
        self._kp_null = float(config.osc_kp_null)
        self._damping_ratio = float(config.osc_damping_ratio)
        self._out_max_pos = float(config.osc_output_max_pos)
        self._out_max_rot = float(config.osc_output_max_rot)

    def _make_gripper(self, arm: str) -> WSG | FrankaGripper:
        gripper_ip = getattr(self.config, f"{arm}_gripper_ip")
        if gripper_ip == getattr(self.config, f"{arm}_robot_ip"):
            return FrankaGripper(
                name=arm,
                server_ip=getattr(self.config, f"{arm}_server_ip"),
                robot_ip=getattr(self.config, f"{arm}_robot_ip"),
                port=getattr(self.config, f"{arm}_port"),
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
        try:
            for n, cam in self.cameras.items():
                try:
                    cam.connect()
                except Exception as e:
                    logger.warning("Camera %s failed to connect: %s", n, e)
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    getattr(self.config, f"{arm}_server_ip"),
                    getattr(self.config, f"{arm}_robot_ip"),
                    getattr(self.config, f"{arm}_port"),
                )
                snap = self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
                # Initialize VOsc goal orientation and nullspace reference
                _, _, _, _, _, ee_rot_xyzw, _ = snap
                self._goal_ori_mat[arm] = _quat_xyzw_to_mat(np.asarray(ee_rot_xyzw, dtype=np.float64))
                self._q0[arm] = np.asarray(snap[0], dtype=np.float64).copy()
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
        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = kin

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

        if self._use_depth:
            depth_cam = self.cameras.get(self._depth_cam)
            if depth_cam is None:
                raise KeyError(f"Depth camera {self._depth_cam!r} not found in cameras")
            depth_fut = self._camera_pool.submit(getattr(depth_cam, "get_depth"))
            full_pcd_fut = self._camera_pool.submit(getattr(depth_cam, "get_full_point_cloud"))
            verts = depth_fut.result()
            full_pcd = full_pcd_fut.result()
            if len(full_pcd) > 0:
                xyz = full_pcd[:, :3]
                dist2 = np.einsum("ij,ij->i", xyz, xyz)
                full_pcd = full_pcd[dist2 <= (_FULL_PCD_CROP_RADIUS_M ** 2)]
            self._last_full_point_cloud = full_pcd
            ee_world = self._ee_world_center(kin)
            flat = self._sample_depth_points(verts, ee_world).reshape(-1)
            for i, v in enumerate(flat):
                obs[f"depth_{i}"] = float(v)
        return obs

    def _ee_world_center(self, kin: dict[str, KinematicSnapshot]) -> np.ndarray:
        arm = "r" if "r" in self.active_arms else self.active_arms[0]
        ee_robot = np.asarray(kin[arm][4], dtype=np.float64)
        return self._r_robot_in_world @ ee_robot + self._t_robot_in_world

    def _sample_depth_points(self, verts: list[tuple[float, float, float]], center: np.ndarray) -> np.ndarray:
        points = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        if points.size == 0:
            return np.zeros((_DEPTH_POINT_COUNT, 3), dtype=np.float32)

        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] == 0:
            return np.zeros((_DEPTH_POINT_COUNT, 3), dtype=np.float32)

        deltas = points - center.reshape(1, 3)
        dist2 = np.einsum("ij,ij->i", deltas, deltas)
        cropped = points[dist2 <= (self._depth_crop_radius_m ** 2)]

        if cropped.shape[0] == 0:
            sampled = np.zeros((_DEPTH_POINT_COUNT, 3), dtype=np.float64)
        elif cropped.shape[0] >= _DEPTH_POINT_COUNT:
            idx = np.linspace(0, cropped.shape[0] - 1, _DEPTH_POINT_COUNT, dtype=np.int64)
            sampled = cropped[idx]
        else:
            reps = (_DEPTH_POINT_COUNT + cropped.shape[0] - 1) // cropped.shape[0]
            sampled = np.tile(cropped, (reps, 1))[:_DEPTH_POINT_COUNT]

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

    def send_action(self, action: RobotAction, ignore_action: bool = False) -> RobotAction:
        kin = self._cached_kin_state or self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = None

        # kp: base * 10^(kp_action). kd: 2*sqrt(kp)*damping_ratio (OSC critical-damping formula).
        kp = self._kp_base * (10.0 ** float(np.clip(action["kp"], -1.0, 1.0)))
        kd = 2.0 * float(np.sqrt(kp)) * self._damping_ratio
        # kd action scalar is ignored (reserved).

        for arm in self.active_arms:
            self.grippers[arm].move(
                np.clip(action[f"{arm}_gripper"], 0.0, 1.0) * self.grippers[arm].GRIPPER_TRUE_MAX_MM,
                blocking=False,
            )

        if self.control_mode == ControlMode.EE_DELTA:
            cmds = self.safety.shape_joint(
                {arm: self._ee_delta_osc(kp, kd, action, arm, kin[arm],
                                         self._goal_ori_mat, self._q0,
                                         self._kp_null, self._out_max_pos, self._out_max_rot,
                                         self.delta_pos, self.delta_rot)
                 for arm in self.active_arms},
                kin,
            )
        elif self.control_mode == ControlMode.EE_POS:
            cmds = self.safety.shape_joint(
                {arm: self._ee_pos_osc(kp, kd, action, arm, kin[arm],
                                       self._goal_ori_mat, self._q0, self._kp_null,
                                       self.delta_pos, self.delta_rot, ignore_action)
                 for arm in self.active_arms},
                kin,
            )
        else:
            cmds = self.safety.shape_joint(
                {arm: self._joint_pd(kp, kd, action, arm, kin[arm]) for arm in self.active_arms},
                kin,
            )
        self.robot_manager.move_joint_velocity_batch({a: c.tolist() for a, c in cmds.items()})
        return action

    @staticmethod
    def _ee_delta_osc(
        kp: float,
        kd: float,
        action: RobotAction,
        arm: str,
        snap: KinematicSnapshot,
        goal_ori_mat: dict[str, np.ndarray],
        q0: dict[str, np.ndarray],
        kp_null: float,
        out_max_pos: float,
        out_max_rot: float,
        dpos: np.ndarray,
        drot: np.ndarray,
    ) -> np.ndarray:
        """VOsc EE_DELTA: scale action, update goal ori (OSC-style), compute joint velocity.

        Position goal is reset each step to current + scaled_delta (OSC EE_DELTA convention).
        Orientation goal is only updated when the rotation delta is non-zero; otherwise the
        last goal is held, providing active orientation stabilization at zero input.
        """
        q, dq, J, M, ee_pos, ee_rot_xyzw, ee_twist = snap
        q   = np.asarray(q,   dtype=np.float64)
        dq  = np.asarray(dq,  dtype=np.float64)
        J   = np.asarray(J,   dtype=np.float64)
        M   = np.asarray(M,   dtype=np.float64)
        ee_pos   = np.asarray(ee_pos,   dtype=np.float64)
        ee_twist = np.asarray(ee_twist, dtype=np.float64)

        # --- Scale inputs from [-1, 1] to physical units ---
        raw_dpos = np.array([action[f"{arm}_x"], action[f"{arm}_y"], action[f"{arm}_z"]], dtype=np.float64)
        raw_dq   = np.array([action[f"{arm}_qx"], action[f"{arm}_qy"], action[f"{arm}_qz"], action[f"{arm}_qw"]], dtype=np.float64)
        raw_dq  /= max(float(np.linalg.norm(raw_dq)), 1e-12)

        scaled_dpos = raw_dpos * out_max_pos + dpos
        # Scale: max quaternion axis-angle magnitude is π → maps to out_max_rot
        raw_aa = _quat_xyzw_to_axis_angle(raw_dq)
        scaled_drot = raw_aa * (out_max_rot / np.pi) + drot

        # --- OSC goal update ---
        # Position: always reset to current + delta (no accumulation across steps)
        goal_pos = ee_pos + scaled_dpos

        # Orientation: only update when the total delta is non-zero (OSC convention —
        # hold last goal at zero input, enabling active orientation stabilization).
        if float(np.linalg.norm(scaled_drot)) > 1e-9:
            goal_ori_mat[arm] = _axis_angle_to_mat(scaled_drot) @ _quat_xyzw_to_mat(
                np.asarray(ee_rot_xyzw, dtype=np.float64)
            )

        return _vosc_joint_velocity(
            goal_pos, goal_ori_mat[arm], q, dq, J, M, ee_pos, ee_twist,
            np.asarray(ee_rot_xyzw, dtype=np.float64),
            kp, kd, kp_null, q0[arm],
        )

    @staticmethod
    def _ee_pos_osc(
        kp: float,
        kd: float,
        action: RobotAction,
        arm: str,
        snap: KinematicSnapshot,
        goal_ori_mat: dict[str, np.ndarray],
        q0: dict[str, np.ndarray],
        kp_null: float,
        dpos: np.ndarray,
        drot: np.ndarray,
        ignore_action: bool,
    ) -> np.ndarray:
        """VOsc EE_POS: set goal directly from absolute EE pose action."""
        q, dq, J, M, ee_pos, ee_rot_xyzw, ee_twist = snap
        q   = np.asarray(q,   dtype=np.float64)
        dq  = np.asarray(dq,  dtype=np.float64)
        J   = np.asarray(J,   dtype=np.float64)
        M   = np.asarray(M,   dtype=np.float64)
        ee_pos   = np.asarray(ee_pos,   dtype=np.float64)
        ee_twist = np.asarray(ee_twist, dtype=np.float64)

        if ignore_action:
            goal_pos = np.asarray(ee_pos, dtype=np.float64).copy()
            goal_ori_mat[arm] = _quat_xyzw_to_mat(np.asarray(ee_rot_xyzw, dtype=np.float64))
        else:
            target_raw = np.fromiter(
                (action[f"{arm}_{ax}"] for ax in EE_AXIS_KEYS),
                dtype=np.float64, count=len(EE_AXIS_KEYS),
            )
            target_raw[3:] /= max(float(np.linalg.norm(target_raw[3:])), 1e-12)
            goal_pos = target_raw[:3] + dpos
            goal_ori_mat[arm] = _axis_angle_to_mat(drot) @ _quat_xyzw_to_mat(target_raw[3:])

        return _vosc_joint_velocity(
            goal_pos, goal_ori_mat[arm], q, dq, J, M, ee_pos, ee_twist,
            np.asarray(ee_rot_xyzw, dtype=np.float64),
            kp, kd, kp_null, q0[arm],
        )

    @staticmethod
    def _joint_pd(kp: float, kd: float, action: RobotAction, arm: str, snap: KinematicSnapshot) -> np.ndarray:
        target = np.fromiter(
            (action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)),
            dtype=np.float64, count=NUM_JOINTS,
        )
        q, dq = snap[0], snap[1]
        return kp * (target - np.asarray(q)) - kd * np.asarray(dq)

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

        In EE_POS and EE_DELTA modes, FK converts joint targets to EE setpoints and
        homing runs VOsc Cartesian PD. In JOINT_POS mode, joint-velocity PD is used.
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

        use_ee_homing = self.control_mode != ControlMode.JOINT_POS
        rate_hz = float(home_fps if home_fps is not None else (max(fps, 60) if use_ee_homing else fps))
        period_s = 1.0 / rate_hz
        deadline = time.perf_counter() + max_time_s
        names = list(targets_q)
        rot_tol = float(tol_rot_rad if tol_rot_rad is not None else tol_rad)

        if use_ee_homing:
            targets_ee: dict[str, np.ndarray] = {}
            for arm, q in targets_q.items():
                p, qu = franka_fk(q)
                qu = np.asarray(qu, dtype=np.float64)
                qu /= max(float(np.linalg.norm(qu)), 1e-12)
                targets_ee[arm] = np.concatenate((p, qu))
            while True:
                tick_start = time.perf_counter()
                kin = self.robot_manager.current_kinematic_state_batch(names)

                cmds_raw: dict[str, np.ndarray] = {}
                for arm in names:
                    snap = kin[arm]
                    q, dq, J, M, ee_pos, ee_rot_xyzw, ee_twist = snap
                    target = targets_ee[arm]
                    goal_pos = target[:3]
                    goal_ori = _quat_xyzw_to_mat(target[3:])
                    cmds_raw[arm] = _vosc_joint_velocity(
                        goal_pos, goal_ori,
                        np.asarray(q, dtype=np.float64), np.asarray(dq, dtype=np.float64),
                        np.asarray(J, dtype=np.float64), np.asarray(M, dtype=np.float64),
                        np.asarray(ee_pos, dtype=np.float64), np.asarray(ee_twist, dtype=np.float64),
                        np.asarray(ee_rot_xyzw, dtype=np.float64),
                        self._kp_base, 2.0 * np.sqrt(self._kp_base) * self._damping_ratio,
                        self._kp_null, self._q0.get(arm, np.asarray(q, dtype=np.float64)),
                    )
                cmds = self.safety.shape_joint(cmds_raw, kin)
                self.robot_manager.move_joint_velocity_batch({a: c.tolist() for a, c in cmds.items()})

                errs = [self._ee_pose_errors(targets_ee[arm], kin[arm]) for arm in names]
                max_pos = max(float(np.linalg.norm(pe)) for pe, _ in errs)
                max_rot = max(float(np.linalg.norm(re)) for _, re in errs)

                if max_pos < tol_pos_m and max_rot < rot_tol:
                    self._cached_kin_state = None
                    return True
                if tick_start >= deadline:
                    self._cached_kin_state = None
                    logger.warning(
                        "home(): EE timeout after %.2fs (pos err %.4f m, rot err %.4f rad)",
                        max_time_s, max_pos, max_rot,
                    )
                    return False

                elapsed = time.perf_counter() - tick_start
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)

        while True:
            tick_start = time.perf_counter()
            kin = self.robot_manager.current_kinematic_state_batch(names)

            cmds_raw = {
                arm: JOINT_PD_KP * (targets_q[arm] - np.asarray(kin[arm][0], dtype=np.float64))
                     - JOINT_PD_KD * np.asarray(kin[arm][1], dtype=np.float64)
                for arm in names
            }
            cmds = self.safety.shape_joint(cmds_raw, kin)
            self.robot_manager.move_joint_velocity_batch({a: c.tolist() for a, c in cmds.items()})

            max_err = max(float(np.max(np.abs(targets_q[arm] - np.asarray(kin[arm][0])))) for arm in names)
            if max_err < tol_rad:
                self._cached_kin_state = None
                return True
            if tick_start >= deadline:
                self._cached_kin_state = None
                logger.warning("home(): timeout after %.2fs, max joint error %.4f rad", max_time_s, max_err)
                return False

            elapsed = time.perf_counter() - tick_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    def cache_delta(self, dpos: np.ndarray, drot: np.ndarray) -> None:
        self.delta_pos = dpos
        self.delta_rot = drot

    @property
    def last_full_point_cloud(self) -> np.ndarray | None:
        return self._last_full_point_cloud

    @staticmethod
    def _ee_pose_errors(target: np.ndarray, snap: KinematicSnapshot) -> tuple[np.ndarray, np.ndarray]:
        """Position error (m) and axis-angle orientation error (rad) vs target (7: x,y,z,qx,qy,qz,qw)."""
        _, _, _, _, pos, rot, _ = snap
        pos_error = target[:3] - np.asarray(pos, dtype=np.float64)

        target_q = target[3:].copy()
        target_q /= max(float(np.linalg.norm(target_q)), 1e-12)
        curr_q = np.asarray(rot, dtype=np.float64).copy()
        curr_q /= max(float(np.linalg.norm(curr_q)), 1e-12)

        x1, y1, z1, w1 = target_q
        x2, y2, z2, w2 = -curr_q[0], -curr_q[1], -curr_q[2], curr_q[3]
        q_err = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float64)
        q_err /= max(float(np.linalg.norm(q_err)), 1e-12)
        if q_err[3] < 0.0:
            q_err = -q_err

        v = q_err[:3]
        v_norm = float(np.linalg.norm(v))
        rot_error = (2.0 * v if v_norm < 1e-9
                     else (v / v_norm) * (2.0 * np.arctan2(v_norm, float(np.clip(q_err[3], -1.0, 1.0)))))
        return pos_error, rot_error


# ---------------------------------------------------------------------------
# VOsc core (module-level for readability — avoids repeating in each static method)
# ---------------------------------------------------------------------------

def _vosc_joint_velocity(
    goal_pos: np.ndarray,
    goal_ori_mat: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    J: np.ndarray,
    M: np.ndarray,
    ee_pos: np.ndarray,
    ee_twist: np.ndarray,
    ee_rot_xyzw: np.ndarray,
    kp: float,
    kd: float,
    kp_null: float,
    q0: np.ndarray,
) -> np.ndarray:
    """Velocity-Space OSC (VOsc): maps task-space PD error to joint velocity.

    Matches robosuite OSC run_controller with uncouple_pos_ori=True, but outputs
    joint velocity instead of torque:

      q̇ = J̄_pos·F + J̄_ori·T + N·(kp_null·(q₀−q) − kd_null·q̇)

    where J̄_* = M⁻¹ J*ᵀ λ*  (dynamically-consistent pseudoinverse)
    and   λ*  = pinv(J* M⁻¹ J*ᵀ)

    Franky's impedance controller handles gravity internally, so no gravity
    compensation term is needed here.
    """
    ee_ori_mat = _quat_xyzw_to_mat(ee_rot_xyzw)

    # Task-space PD (identical to OSC desired_force / desired_torque)
    pos_err = goal_pos - ee_pos
    ori_err = _orientation_error(goal_ori_mat, ee_ori_mat)

    F = kp * pos_err - kd * ee_twist[:3]
    T = kp * ori_err - kd * ee_twist[3:]

    # Lambda matrices and dynamically-consistent pseudoinverses
    J_bar_pos, J_bar_ori, J_bar_full, N = _osc_matrices(J, M)

    # Uncoupled task-space → joint velocity (OSC uncouple_pos_ori=True analog)
    q_dot = J_bar_pos @ F + J_bar_ori @ T

    # Nullspace: attract joints toward q0 without disturbing EE
    kd_null = 2.0 * float(np.sqrt(kp_null))
    q_dot += N @ (kp_null * (q0 - q) - kd_null * dq)

    return q_dot
