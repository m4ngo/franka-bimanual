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

from .bimanual_franka_config import BimanualFrankaConfig
from .franka_process import NUM_JOINTS, KinematicSnapshot, MultiRobotWrapper
from .safety import ActionSafetyScreen
from .wsg import WSG

IMAGE_CHANNELS = 3
_CAMERA_READ_TIMEOUT_MS: float = 5.0
_CONNECT_TIMEOUT_S = 10.0

JOINT_PD_KP, JOINT_PD_KD = 2.0, 0.1
EE_PD_KP, EE_PD_KD = 2.0, 0.1

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
        self.use_ee_pos = config.use_ee_pos
        self.active_arms = config.active_arms
        self.cameras: dict[str, Camera] = {n: _make_camera(c) for n, c in config.cameras.items()}
        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG] = {
            arm: WSG(name=arm, TCP_IP=getattr(config, f"{arm}_gripper_ip"), do_print=False)
            for arm in self.active_arms
        }
        self.safety = ActionSafetyScreen()
        # Populated by get_observation, consumed by next send_action to skip a redundant RPyC round-trip.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None
        self._camera_pool = ThreadPoolExecutor(max_workers=max(len(self.cameras), 1))

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
        return {**self._arm_features(JOINT_FEATURE_KEYS), **self._camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        return self._arm_features(EE_FEATURE_KEYS if self.use_ee_pos else JOINT_FEATURE_KEYS)

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
                    use_ee_delta=self.use_ee_pos,
                )
                self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
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
            obs[f"{arm}_gripper"] = (0 if pos is None else pos) / WSG.GRIPPER_TRUE_MAX_MM

        for n, fut in cam_futs.items():
            try:
                obs[n] = fut.result()
            except Exception as e:
                logger.warning("Camera %s read failed: %s", n, e)
                blank = getattr(self.cameras[n], "blank_frame", None)
                obs[n] = blank() if callable(blank) else np.zeros(self._camera_features[n], dtype=np.uint8)
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        kin = self._cached_kin_state or self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = None

        for arm in self.active_arms:
            self.grippers[arm].move(action[f"{arm}_gripper"] * WSG.GRIPPER_TRUE_MAX_MM, blocking=False)

        if self.use_ee_pos:
            cmds = self.safety.shape_ee(
                {arm: self._ee_pd(action, arm, kin[arm]) for arm in self.active_arms}, kin
            )
            self.robot_manager.move_ee_delta_batch({a: c.tolist() for a, c in cmds.items()})
        else:
            cmds = self.safety.shape_joint(
                {arm: self._joint_pd(action, arm, kin[arm]) for arm in self.active_arms}, kin
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
    ) -> bool:
        """Drive both arms to a joint-space home pose via the joint-velocity PD.

        Bypasses `use_ee_pos` so this works regardless of the configured action
        mode. Runs a closed-loop PD at `fps` Hz until each active arm's max
        joint error is below `tol_rad`, or `max_time_s` elapses. Gripper
        command is fire-and-forget.

        Returns True on convergence, False on timeout.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        candidates = {"l": home_q_left, "r": home_q_right}
        targets = {
            arm: np.asarray(q, dtype=np.float64)
            for arm, q in candidates.items()
            if q is not None and arm in self.active_arms
        }
        if not targets:
            return True

        for arm in targets:
            self.grippers[arm].move(gripper_norm * WSG.GRIPPER_TRUE_MAX_MM, blocking=False)

        period_s = 1.0 / fps
        deadline = time.perf_counter() + max_time_s
        names = list(targets)
        while True:
            tick_start = time.perf_counter()
            kin = self.robot_manager.current_kinematic_state_batch(names)

            cmds_raw = {
                arm: JOINT_PD_KP * (targets[arm] - kin[arm][0]) - JOINT_PD_KD * kin[arm][1]
                for arm in names
            }
            cmds = self.safety.shape_joint(cmds_raw, kin)
            self.robot_manager.move_joint_velocity_batch({a: c.tolist() for a, c in cmds.items()})

            max_err = max(
                float(np.max(np.abs(targets[arm] - kin[arm][0]))) for arm in names
            )
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

    @staticmethod
    def _joint_pd(action: RobotAction, arm: str, snap: KinematicSnapshot) -> np.ndarray:
        target = np.fromiter(
            (action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)),
            dtype=np.float64, count=NUM_JOINTS,
        )
        q, dq = snap[0], snap[1]
        return JOINT_PD_KP * (target - q) - JOINT_PD_KD * dq

    @staticmethod
    def _ee_pd(action: RobotAction, arm: str, snap: KinematicSnapshot) -> np.ndarray:
        target = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in EE_AXIS_KEYS),
            dtype=np.float64, count=len(EE_AXIS_KEYS),
        )
        _, _, _, pos, rot, twist = snap

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
        if v_norm < 1e-9:
            rot_error = 2.0 * v
        else:
            rot_error = (v / v_norm) * (2.0 * np.arctan2(v_norm, float(np.clip(q_err[3], -1.0, 1.0))))

        return EE_PD_KP * np.concatenate((pos_error, rot_error)) - EE_PD_KD * np.asarray(twist, dtype=np.float64)
