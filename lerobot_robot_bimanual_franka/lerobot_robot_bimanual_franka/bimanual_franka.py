"""Bimanual Franka robot plugin for LeRobot.

Wraps two Franka arms (left / right) plus their Schunk WSG grippers behind the
LeRobot Robot interface. Each arm runs in its own subprocess via
MultiRobotWrapper; grippers communicate over TCP through WSG.
"""

import logging
import time
from functools import cached_property

import numpy as np

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from lerobot_camera_arv import ArvCamera, ArvCameraConfig  # type: ignore

from .bimanual_franka_config import BimanualFrankaConfig
from .franka_process import KinematicSnapshot, MultiRobotWrapper
from .safety import ActionSafetyScreen
from .wsg import WSG

# 7 degrees of freedom per Franka arm.
NUM_JOINTS = 7
IMAGE_CHANNELS = 3

# Joint-velocity PD controller for tracking joint-position targets. Lives in
# the parent process (this file) rather than franka_process so the safety
# screen can inspect/modify the same velocities that get streamed to franky.
JOINT_PD_KP = 2.5
JOINT_PD_KD = 0.1

# Per-arm action/observation feature key suffixes.
JOINT_FEATURE_KEYS: tuple[str, ...] = (
    *(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)),
    "gripper",
)
EE_FEATURE_KEYS: tuple[str, ...] = (
    "x", "y", "z", "roll", "pitch", "yaw", "gripper",
)
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")

# Connection bring-up parameters.
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
        self.cameras: dict[str, ArvCamera] = {
            camera_name: self._make_camera(camera_config)
            for camera_name, camera_config in self.config.cameras.items()
        }

        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG] = {
            arm: WSG(name=arm, TCP_IP=self._gripper_ip(arm), do_print=False)
            for arm in self.active_arms
        }
        self.safety = ActionSafetyScreen()

        # Cache populated by `get_observation` and consumed by the very next
        # `send_action` to avoid issuing a redundant `current_kinematic_state`
        # IPC every loop. Cleared after a single use so a stand-alone
        # `send_action` still fetches a fresh snapshot.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None

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

    def _make_camera(self, camera: ArvCameraConfig) -> ArvCamera:
        return ArvCamera(
            ArvCameraConfig(
                name=camera.name,
                ip=camera.ip,
                width=camera.width,
                height=camera.height,
                fps=camera.fps,
            )
        )

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        return {
            camera_name: (
                self.cameras[camera_name].height,
                self.cameras[camera_name].width,
                IMAGE_CHANNELS,
            )
            for camera_name in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {
            **self._arm_features(JOINT_FEATURE_KEYS),
            **self._camera_features,
        }

    @property
    def action_features(self) -> dict[str, type]:
        return self._arm_features(
            EE_FEATURE_KEYS if self.use_ee_delta else JOINT_FEATURE_KEYS
        )

    @property
    def is_connected(self) -> bool:
        return self.robot_manager.num_processes == len(self.active_arms)

    def connect(self, calibrate: bool = True) -> None:
        """Start arm processes, verify they respond, then home the grippers."""
        try:
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    self._server_ip(arm),
                    self._robot_ip(arm),
                    self._port(arm),
                    use_ee_delta=self.use_ee_delta,
                )

            # Give each subprocess time to initialise its RPC connection.
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
        for camera_name, camera in self.cameras.items():
            try:
                camera.connect()
            except Exception as exc:  # noqa: BLE001 - cameras should not block control
                logger.warning("Camera %s failed to connect: %s", camera_name, exc)

    def _probe_arm(self, arm: str) -> None:
        """Confirm an arm responds to a kinematic-state query, retrying on failure."""
        last_error: Exception | None = None
        for _ in range(_CONNECT_RETRIES):
            try:
                self.robot_manager.current_kinematic_state(
                    arm, timeout_s=_CONNECT_TIMEOUT_S
                )
                return
            except Exception as e:
                last_error = e
                time.sleep(_RETRY_SLEEP_S)

        raise RuntimeError(
            f"Failed to communicate with robot '{arm}' at {self._robot_ip(arm)}: {last_error}"
        )

    def disconnect(self) -> None:
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

    # ------------------------------------------------------------------
    # Hot path: observation + action
    # ------------------------------------------------------------------

    def _fetch_kin_state(self) -> dict[str, KinematicSnapshot]:
        """One parallel IPC round-trip to grab (q, dq, ee, J) for every arm."""
        return self.robot_manager.current_kinematic_state_batch(
            list(self.active_arms)
        )

    def _consume_kin_state(self) -> dict[str, KinematicSnapshot]:
        """Return the cached snapshot if fresh, otherwise fetch one."""
        kin_state = self._cached_kin_state
        self._cached_kin_state = None
        return kin_state if kin_state is not None else self._fetch_kin_state()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        kin_state = self._fetch_kin_state()
        # Stash for send_action. send_action invalidates after consuming.
        self._cached_kin_state = kin_state

        obs: RobotObservation = {}
        for arm in self.active_arms:
            q = kin_state[arm][0]
            for i in range(NUM_JOINTS):
                obs[f"{arm}_joint_{i + 1}"] = float(q[i])
            obs[f"{arm}_gripper"] = self.grippers[arm].position

        for camera_name, camera in self.cameras.items():
            try:
                obs[camera_name] = camera.async_read()
            except Exception as exc:  # noqa: BLE001 - preserve control-path behavior
                logger.warning("Camera %s read failed: %s", camera_name, exc)
                obs[camera_name] = camera.blank_frame()
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Forward gripper + arm commands.

        One IPC round-trip fetches the kinematic snapshot used by both the PD
        controller and the safety screen (or reuses the snapshot stashed by
        the immediately preceding `get_observation`). A second parallel IPC
        ships the velocity command to both arm subprocesses. Gripper sends
        are non-blocking and run in parallel with the arm IPC.
        """
        for arm in self.active_arms:
            self.grippers[arm].move(action[f"{arm}_gripper"], blocking=False)

        kin_state = self._consume_kin_state()

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
            return action

        # Joint mode: PD turns each position target into a velocity, then the
        # safety screen + magnitude clamp shape it before it goes out.
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
    def _joint_pd(
        action: RobotAction, arm: str, snapshot: KinematicSnapshot
    ) -> np.ndarray:
        target = np.fromiter(
            (action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)),
            dtype=np.float64,
            count=NUM_JOINTS,
        )
        q, dq, _, _ = snapshot
        return JOINT_PD_KP * (target - q) - JOINT_PD_KD * dq
