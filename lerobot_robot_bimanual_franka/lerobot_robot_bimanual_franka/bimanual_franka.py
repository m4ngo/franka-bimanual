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
from .franka_process import KinematicSnapshot, MultiRobotWrapper
from .safety import ActionSafetyScreen
from .wsg import WSG

NUM_JOINTS = 7
IMAGE_CHANNELS = 3

# Short timeout for async camera reads; falls back to cached frame rather than blocking.
_CAMERA_READ_TIMEOUT_MS: float = 5.0

# PD gains for joint-position tracking (parent-side, in rad/s output).
JOINT_PD_KP = 2.0
JOINT_PD_KD = 0.1

JOINT_FEATURE_KEYS: tuple[str, ...] = (*(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)), "gripper")
EE_FEATURE_KEYS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")

_PROCESS_STARTUP_S = 1.0
_CONNECT_RETRIES = 3
_CONNECT_TIMEOUT_S = 10.0
_RETRY_SLEEP_S = 1.0

logger = logging.getLogger(__name__)


class BimanualFranka(Robot):
    config_class = BimanualFrankaConfig
    name = "bimanual_franka"

    def __init__(self, config: BimanualFrankaConfig):
        super().__init__(config)
        self.config = config
        self.use_ee_delta = config.use_ee_delta
        self.active_arms = config.active_arms
        self.cameras: dict[str, Camera] = {
            name: self._make_camera(cfg) for name, cfg in config.cameras.items()
        }
        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG] = {
            arm: WSG(name=arm, TCP_IP=self._gripper_ip(arm), do_print=False)
            for arm in self.active_arms
        }
        self.safety = ActionSafetyScreen()
        # Populated by get_observation; consumed by the next send_action to avoid a redundant IPC round-trip.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None
        self._camera_pool = ThreadPoolExecutor(max_workers=max(len(self.cameras), 1))

    def _gripper_ip(self, arm: str) -> str:
        return getattr(self.config, f"{arm}_gripper_ip")

    def _server_ip(self, arm: str) -> str:
        return getattr(self.config, f"{arm}_server_ip")

    def _robot_ip(self, arm: str) -> str:
        return getattr(self.config, f"{arm}_robot_ip")

    def _port(self, arm: str) -> int:
        return getattr(self.config, f"{arm}_port")

    def _arm_features(self, keys: tuple[str, ...]) -> dict[str, type]:
        return {f"{arm}_{key}": float for arm in self.active_arms for key in keys}

    def _make_camera(self, camera: CameraConfig) -> Camera:
        if isinstance(camera, FramosCameraConfig):
            return FramosCamera(camera)
        if isinstance(camera, ArvCameraConfig):
            return ArvCamera(camera)
        raise TypeError(f"Unsupported camera config type: {type(camera).__name__}")

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (cam.height, cam.width, IMAGE_CHANNELS)
            for name, cam in self.cameras.items()
        }

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._arm_features(JOINT_FEATURE_KEYS), **self._camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        return self._arm_features(EE_FEATURE_KEYS if self.use_ee_delta else JOINT_FEATURE_KEYS)

    @property
    def is_connected(self) -> bool:
        return self.robot_manager.num_processes == len(self.active_arms)

    def connect(self, calibrate: bool = True) -> None:
        try:
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    self._server_ip(arm),
                    self._robot_ip(arm),
                    self._port(arm),
                    use_ee_delta=self.use_ee_delta,
                )
            time.sleep(_PROCESS_STARTUP_S)
            for arm in self.active_arms:
                self._probe_arm(arm)
            if not self.is_calibrated and calibrate:
                self.calibrate()
            self.configure()
            for arm in self.active_arms:
                self.grippers[arm].home()
            self._connect_cameras()
        except Exception:
            self.robot_manager.shutdown()
            raise

    def _connect_cameras(self) -> None:
        for name, camera in self.cameras.items():
            try:
                camera.connect()
            except Exception as exc:
                logger.warning("Camera %s failed to connect: %s", name, exc)

    def _probe_arm(self, arm: str) -> None:
        last_error: Exception | None = None
        for _ in range(_CONNECT_RETRIES):
            try:
                self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
                return
            except Exception as e:
                last_error = e
                time.sleep(_RETRY_SLEEP_S)
        raise RuntimeError(
            f"Failed to communicate with robot '{arm}' at {self._robot_ip(arm)}: {last_error}"
        )

    def disconnect(self) -> None:
        self._camera_pool.shutdown(wait=False)
        self._cached_kin_state = None
        for camera in self.cameras.values():
            camera.disconnect()
        self.robot_manager.shutdown()
        for gripper in self.grippers.values():
            gripper.close()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Submit camera reads immediately so they overlap the IPC round-trip below.
        camera_futures = {
            name: self._camera_pool.submit(camera.async_read, _CAMERA_READ_TIMEOUT_MS)
            for name, camera in self.cameras.items()
        }

        kin_state = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = kin_state

        obs: RobotObservation = {}
        for arm in self.active_arms:
            q = kin_state[arm][0]
            for i in range(NUM_JOINTS):
                obs[f"{arm}_joint_{i + 1}"] = float(q[i]) / (2 * np.pi)
            pos = self.grippers[arm].position
            obs[f"{arm}_gripper"] = (0 if pos is None else pos) / WSG.GRIPPER_TRUE_MAX_MM

        for name, fut in camera_futures.items():
            try:
                obs[name] = fut.result()
            except Exception as exc:
                logger.warning("Camera %s read failed: %s", name, exc)
                obs[name] = self.cameras[name].blank_frame()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Forward normalized action to grippers and arms.

        Reuses the kinematic snapshot from the preceding get_observation, or fetches a fresh one.
        Gripper and arm commands are fire-and-forget (non-blocking).
        """
        kin_state = self._cached_kin_state or self.robot_manager.current_kinematic_state_batch(
            list(self.active_arms)
        )
        self._cached_kin_state = None

        # Gripper action is normalized [0, 1]; convert to mm before sending.
        for arm in self.active_arms:
            self.grippers[arm].move(action[f"{arm}_gripper"] * WSG.GRIPPER_TRUE_MAX_MM, blocking=False)

        if self.use_ee_delta:
            twists = {
                arm: np.fromiter(
                    (action[f"{arm}_{ax}"] for ax in EE_AXIS_KEYS),
                    dtype=np.float64,
                    count=len(EE_AXIS_KEYS),
                )
                for arm in self.active_arms
            }
            twists = self.safety.shape_ee(twists, kin_state)
            self.robot_manager.move_ee_delta_batch(
                {arm: twist.tolist() for arm, twist in twists.items()},
                asynchronous=True,
            )
        else:
            velocities = {
                arm: self._joint_pd(action, arm, kin_state[arm])
                for arm in self.active_arms
            }
            velocities = self.safety.shape_joint(velocities, kin_state)
            self.robot_manager.move_joint_velocity_batch(
                {arm: vel.tolist() for arm, vel in velocities.items()},
                asynchronous=True,
            )

        return action

    @staticmethod
    def _joint_pd(action: RobotAction, arm: str, snapshot: KinematicSnapshot) -> np.ndarray:
        # Action joints are normalized by 2π; convert back to radians for PD tracking.
        target = np.array(
            [action[f"{arm}_joint_{i}"] * 2 * np.pi for i in range(1, NUM_JOINTS + 1)]
        )
        q, dq, _, _ = snapshot
        return JOINT_PD_KP * (target - q) - JOINT_PD_KD * dq
