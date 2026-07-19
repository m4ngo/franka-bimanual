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
from .osc_torque_controller import OSCTorqueController
from .franka_jacobian import zero_jacobian  # the new analytic module

IMAGE_CHANNELS = 3
_CAMERA_READ_TIMEOUT_MS: float = 5.0
_CONNECT_TIMEOUT_S = 10.0
_DEPTH_POINT_COUNT = 2048

JOINT_PD_KP, JOINT_PD_KD = 2.0, 0.1
EE_PD_KP, EE_PD_KD = 3.5, 0.2
_KP_GAIN_BASE = 10.0
_KD_GAIN_BASE = 1.0
OSC_BASE_KP = 5.0
_GRIP_ACCUM_SPEED = 0.5

_EE_TRANSLATION_FUDGE_FACTOR = 1.2
_EE_ROTATION_FUDGE_FACTOR = 0.9

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
        self._osc_vel: dict[str, OSCTorqueController] = {
            arm: OSCTorqueController(num_joints=NUM_JOINTS) for arm in self.active_arms
        }
        self._home_q: dict[str, np.ndarray] = {} # nullspace target; set in home()


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
                self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
            for arm in self.active_arms:
                self.grippers[arm].home()

                kin = self.robot_manager.current_kinematic_state(arm)
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
        """franky's zero_jacobian returns all-zero on this build; recompute
        the geometric Jacobian analytically from q, anchored on the measured
        EE position so it stays consistent with the pose error used elsewhere."""
        q, dq, J, ee_pos, ee_quat_xyzw, ee_twist = snap
        q = np.asarray(q, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        J_real = zero_jacobian(q, ee_pos_base=ee_pos)
        return (q, dq, J_real, ee_pos, ee_quat_xyzw, ee_twist)

    def send_action(self, action: RobotAction, ignore_action: bool = False) -> RobotAction:
        kin = self._cached_kin_state or self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = None
        kin = {arm: self._patch_jacobian(snap) for arm, snap in kin.items()}
        dyn = self.robot_manager.current_dynamics_batch(list(self.active_arms))
        self._kp_gain = _KP_GAIN_BASE ** np.clip(action["kp"], -1.0, 1.0)
        self._kd_gain = _KD_GAIN_BASE ** (np.clip(action["kd"], -1.0, 1.0) * 2 * np.sqrt(self.kp_gain))

        for arm in self.active_arms:
            self._gripper_accum[arm] = np.clip(self._gripper_accum[arm] + (action[f"{arm}_gripper"]) * _GRIP_ACCUM_SPEED, -1.0, 1.0)
            self.grippers[arm].move(
                (self._gripper_accum[arm] + 1.0) / 2.0 * self.grippers[arm].GRIPPER_TRUE_MAX_MM,
                blocking=False,
            )

        if self.control_mode == ControlMode.EE_DELTA:

            # print(action["r_z"])
            # unnormalize actions
            for arm in self.active_arms:
                for ax in ("x", "y", "z"):
                    action[f"{arm}_{ax}"] *= _EE_TRANSLATION_FUDGE_FACTOR
                for ax in ("qx", "qy", "qz", "qw"):
                    action[f"{arm}_{ax}"] *= _EE_ROTATION_FUDGE_FACTOR

            # print(action)
            # cmds = self.safety.shape_ee(
            #     {arm: self._ee_delta(self.kp_gain, self.kd_gain, action, arm, kin[arm], self.delta_pos, self.delta_rot, self.config.use_noise, self.config.noise_pos_scale, self.config.noise_rot_scale)
            #      for arm in self.active_arms}, kin
            # )
            cmds = {arm: self._tau_ee_delta(arm, action, kin[arm], dyn[arm], ...) for arm in self.active_arms}
            self.robot_manager.move_joint_torque_batch({a: c.tolist() for a, c in cmds.items()})
        elif self.control_mode == ControlMode.EE_POS:
            cmds = self.safety.shape_ee(
                {arm: self._ee_pd(self.kp_gain, self.kd_gain, action, arm, kin[arm], self.delta_pos, self.delta_rot, ignore_action)
                 for arm in self.active_arms}, kin
            )
            self.robot_manager.move_ee_delta_batch({a: c.tolist() for a, c in cmds.items()})
        else:
            cmds = self.safety.shape_joint(
                {arm: self._joint_pd(self.kp_gain, self.kd_gain, action, arm, kin[arm]) for arm in self.active_arms}, kin
            )
            self.robot_manager.move_joint_velocity_batch({a: c.tolist() for a, c in cmds.items()})
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

        Joint targets (``home_q_*``) are always interpreted as the desired ``q``.
        In EE_POS and EE_DELTA modes, FK converts them to EE setpoints and homing
        runs Cartesian-velocity PD. In JOINT_POS mode, joint-velocity PD is used.

        Control rate: ``home_fps`` if set; else EE modes use ``max(fps, 60)``, joint
        mode uses ``fps``.
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

                cmds_raw = {arm: self._ee_velocity_toward_pose(1.0, 1.0, targets_ee[arm], kin[arm]) for arm in names}
                cmds = self.safety.shape_ee(cmds_raw, kin)
                self.robot_manager.move_ee_delta_batch({a: c.tolist() for a, c in cmds.items()})

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
                arm: JOINT_PD_KP * (targets_q[arm] - kin[arm][0]) - JOINT_PD_KD * kin[arm][1]
                for arm in names
            }
            cmds = self.safety.shape_joint(cmds_raw, kin)
            self.robot_manager.move_joint_velocity_batch({a: c.tolist() for a, c in cmds.items()})

            max_err = max(float(np.max(np.abs(targets_q[arm] - kin[arm][0]))) for arm in names)
            if max_err < tol_rad:
                self._cached_kin_state = None
                for arm in names:
                    self._home_q[arm] = targets_q[arm].copy()
                return True
            if tick_start >= deadline:
                self._cached_kin_state = None
                logger.warning("home(): timeout after %.2fs, max joint error %.4f rad", max_time_s, max_err)
                return False

            elapsed = time.perf_counter() - tick_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    def _tau_ee_delta(
        self,
        arm: str,
        action,  # RobotAction
        snap,  # KinematicSnapshot = (q, dq, J, ee_pos, ee_quat_xyzw, ee_twist)
        dpos_cached: np.ndarray,
        drot_cached: np.ndarray,
        use_noise: bool,
        noise_pos_scale: float,
        noise_rot_scale: float,
    ) -> np.ndarray:
        """Replacement for the old `_ee_delta`. Integrates the incoming delta
        action into a persistent per-arm goal pose (mirroring robosuite OSC's
        set_goal() under use_delta=True), then servos the current EE toward that
        goal with OSCVelocityController -- rather than treating the raw delta as
        a one-shot velocity command every tick.
        """

        if use_noise:
            pos_noise = np.random.normal(0.0, noise_pos_scale, 3)
            rot_noise = Rotation.from_euler("xyz", np.random.normal(0.0, noise_rot_scale, 3)).as_quat()
        else:
            pos_noise = 0
            rot_noise = 0


        q, dq_, J, ee_pos, ee_quat_xyzw, ee_twist = snap
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_quat_xyzw = np.asarray(ee_quat_xyzw, dtype=np.float64)
        action_dpos = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("x", "y", "z")),
            dtype=np.float64, count=3,
        ) + pos_noise
        action_dquat_xyzw = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("qx", "qy", "qz", "qw")),
            dtype=np.float64, count=4,
        ) + rot_noise

        # ---- integrate delta into the persistent goal (position: simple sum;
        #      orientation: compose as delta * goal) ----
        goal_pos = ee_pos + action_dpos + dpos_cached

        dq = action_dquat_xyzw / max(float(np.linalg.norm(action_dquat_xyzw)), 1e-12)
        gx, gy, gz, gw = ee_quat_xyzw
        dx, dy, dz, dw = dq
        new_quat = np.array([
            dw * gx + dx * gw + dy * gz - dz * gy,
            dw * gy - dx * gz + dy * gw + dz * gx,
            dw * gz + dx * gy - dy * gx + dz * gw,
            dw * gw - dx * gx - dy * gy - dz * gz,
        ])
        goal_quat_xyzw = new_quat / max(float(np.linalg.norm(new_quat)), 1e-12)

        if drot_cached is not None and np.any(drot_cached):
            angle = float(np.linalg.norm(drot_cached))
            if angle > 1e-9:
                axis = drot_cached / angle
                s, c = np.sin(angle / 2.0), np.cos(angle / 2.0)
                bx, by, bz, bw = axis[0] * s, axis[1] * s, axis[2] * s, c
                gx, gy, gz, gw = goal_quat_xyzw
                composed = np.array([
                    bw * gx + bx * gw + by * gz - bz * gy,
                    bw * gy - bx * gz + by * gw + bz * gx,
                    bw * gz + bx * gy - by * gx + bz * gw,
                    bw * gw - bx * gx - by * gy - bz * gz,
                ])
                goal_quat_xyzw = composed / max(float(np.linalg.norm(composed)), 1e-12)

        # ---- servo current EE toward the (now-updated) goal ----

        q, dq_, J, ee_pos, ee_quat_xyzw, ee_twist = snap
        mass_matrix, coriolis = self._last_dynamics[arm]  # see point 4
        return self._osc_tau[arm].compute_tau(
            goal_pos=goal_pos,
            goal_quat_xyzw=goal_quat_xyzw,
            ee_pos=ee_pos,
            ee_quat_xyzw=ee_quat_xyzw,
            ee_twist=np.asarray(ee_twist, dtype=np.float64),
            J=np.asarray(J, dtype=np.float64),
            q=np.asarray(q, dtype=np.float64),
            dq=np.asarray(dq_, dtype=np.float64),
            mass_matrix=mass_matrix,
            coriolis=coriolis,
            q_nullspace_target=self._home_q.get(arm),
            kp=self.kp_gain * OSC_BASE_KP,
            rot_fudge=_EE_ROTATION_FUDGE_FACTOR,
        )

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

    @staticmethod
    def _ee_pose_errors(target: np.ndarray, snap: KinematicSnapshot) -> tuple[np.ndarray, np.ndarray]:
        """Position error (m) and axis-angle orientation error (rad) vs ``target`` (7: x,y,z,qx,qy,qz,qw)."""
        _, _, _, pos, rot, _ = snap
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
        rot_error = 2.0 * v if v_norm < 1e-9 else (v / v_norm) * (2.0 * np.arctan2(v_norm, float(np.clip(q_err[3], -1.0, 1.0))))

        return pos_error, rot_error

    @staticmethod
    def _ee_velocity_toward_pose(
        kp_gain: float,
        kd_gain: float,
        target: np.ndarray,
        snap: KinematicSnapshot,
        dpos: np.ndarray | None = None,
        drot: np.ndarray | None = None,
        ignore_action: bool = False,
    ) -> np.ndarray:
        pos_error, rot_error = BimanualFranka._ee_pose_errors(target, snap)
        if ignore_action:
            pos_error = np.zeros_like(pos_error)
            rot_error = np.zeros_like(rot_error)
        if dpos is not None:
            pos_error += dpos
        if drot is not None:
            rot_error += drot
        _, _, _, _, _, twist = snap
        return (EE_PD_KP * kp_gain) * np.concatenate((pos_error, rot_error)) - (EE_PD_KD * kd_gain) * np.asarray(twist, dtype=np.float64)

    @staticmethod
    def _joint_pd(kp_gain: float, kd_gain: float, action: RobotAction, arm: str, snap: KinematicSnapshot) -> np.ndarray:
        target = np.fromiter(
            (action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)),
            dtype=np.float64, count=NUM_JOINTS,
        )
        q, dq = snap[0], snap[1]
        return (JOINT_PD_KP * kp_gain) * (target - q) - (JOINT_PD_KD * kd_gain) * dq

    @staticmethod
    def _ee_pd(
        kp_gain: float,
        kd_gain: float,
        action: RobotAction,
        arm: str,
        snap: KinematicSnapshot,
        dpos: np.ndarray | None = None,
        drot: np.ndarray | None = None,
        ignore_action: bool = False,
    ) -> np.ndarray:
        target = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in EE_AXIS_KEYS),
            dtype=np.float64, count=len(EE_AXIS_KEYS),
        )
        return BimanualFranka._ee_velocity_toward_pose(kp_gain, kd_gain, target, snap, dpos, drot, ignore_action)

    @staticmethod
    def _ee_delta(
        kp_gain: float,
        kd_gain: float,
        action: RobotAction,
        arm: str,
        snap: KinematicSnapshot,
        dpos: np.ndarray | None = None,
        drot: np.ndarray | None = None,
        use_noise: bool = False,
        noise_pos_scale: float = 0.01,
        noise_rot_scale: float = 0.03,
    ) -> np.ndarray:
        """Apply EE deltas from action directly as a velocity command.

        Position axes (x, y, z) are position deltas in metres. Rotation axes
        (qx, qy, qz, qw) encode a delta rotation as a unit quaternion (xyzw),
        converted to axis-angle here. Cached deltas (dpos, drot) are added on top.
        """
        action_dpos = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("x", "y", "z")),
            dtype=np.float64, count=3,
        )
        dq = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("qx", "qy", "qz", "qw")),
            dtype=np.float64, count=4,
        )
        v = dq[:3]
        v_norm = float(np.linalg.norm(v))
        action_drot = 2.0 * v if v_norm < 1e-9 else (v / v_norm) * (2.0 * np.arctan2(v_norm, float(np.clip(dq[3], -1.0, 1.0))))

        if use_noise:
            pos_noise = np.random.normal(0.0, noise_pos_scale, 3)
            rot_noise = Rotation.from_euler("xyz", np.random.normal(0.0, noise_rot_scale, 3)).as_rotvec()
        else:
            pos_noise = 0
            rot_noise = 0

        total_dpos = action_dpos + (dpos if dpos is not None else np.zeros(3, dtype=np.float64)) + pos_noise
        total_drot = (action_drot + (drot if drot is not None else np.zeros(3, dtype=np.float64))) + rot_noise

        _, _, _, _, _, twist = snap
        return (EE_PD_KP * kp_gain) * np.concatenate((total_dpos, total_drot)) - (EE_PD_KD * kd_gain) * np.asarray(twist, dtype=np.float64)
