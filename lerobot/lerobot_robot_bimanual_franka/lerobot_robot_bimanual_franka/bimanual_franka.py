"""Bimanual Franka robot plugin for LeRobot.

Wraps two Franka arms (left / right) plus their Schunk WSG grippers behind the
LeRobot Robot interface. Each arm runs in its own subprocess via
MultiRobotWrapper; grippers communicate over TCP through WSG.
"""

import time

import numpy as np

from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from .bimanual_franka_config import BimanualFrankaConfig
from .franka_process import MultiRobotWrapper
from .safety import ActionSafetyScreen
from .wsg import WSG

# 7 degrees of freedom per Franka arm.
NUM_JOINTS = 7

# Joint-velocity PD controller for tracking joint-position targets. Lives in
# the parent process (this file) rather than franka_process so the safety
# screen can inspect/modify the same velocities that get streamed to franky.
JOINT_PD_KP = 2.5
JOINT_PD_KD = 0.1

# Connection bring-up parameters.
_PROCESS_STARTUP_S = 1.0
_CONNECT_RETRIES = 3
_CONNECT_TIMEOUT_S = 10.0
_RETRY_SLEEP_S = 1.0


class BimanualFranka(Robot):
    config_class = BimanualFrankaConfig
    name = "my_cool_robot"

    def __init__(self, config: BimanualFrankaConfig):
        super().__init__(config)
        self.config = config
        self.use_ee_delta = config.use_ee_delta
        self.active_arms = config.active_arms

        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG] = {
            arm: WSG(name=arm, TCP_IP=self._gripper_ip(arm), do_print=False)
            for arm in self.active_arms
        }
        self.safety = ActionSafetyScreen()

    def _gripper_ip(self, arm: str) -> str:
        return getattr(self.config, f"{arm}_gripper_ip")

    def _server_ip(self, arm: str) -> str:
        return getattr(self.config, f"{arm}_server_ip")

    def _robot_ip(self, arm: str) -> str:
        return getattr(self.config, f"{arm}_robot_ip")

    def _port(self, arm: str) -> int:
        return getattr(self.config, f"{arm}_port")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Action feature schema: end-effector deltas or joint angles."""
        if self.use_ee_delta:
            axes = ("x", "y", "z", "roll", "pitch", "yaw")
            return {
                f"{arm}_{key}": float
                for arm in self.active_arms
                for key in (*axes, "gripper")
            }
        return self._motors_ft_joints

    @property
    def _motors_ft_joints(self) -> dict[str, type]:
        """Per-joint action/observation feature schema."""
        return {
            f"{arm}_{key}": float
            for arm in self.active_arms
            for key in (
                *(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)),
                "gripper",
            )
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Camera support is intentionally disabled; reserve the hook for later.
        return {}

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft_joints, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

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

            # Give each subprocess time to initialize its RPC connection.
            time.sleep(_PROCESS_STARTUP_S)
            for arm in self.active_arms:
                self._probe_arm(arm)

            if not self.is_calibrated and calibrate:
                self.calibrate()

            self.configure()
            for arm in self.active_arms:
                self.grippers[arm].home()
        except Exception:
            self.robot_manager.shutdown()
            raise

    def _probe_arm(self, arm: str) -> None:
        """Confirm an arm responds to a joint-state query, retrying on failure."""
        last_error: Exception | None = None
        for _ in range(_CONNECT_RETRIES):
            try:
                self.robot_manager.current_joint_positions(arm, timeout_s=_CONNECT_TIMEOUT_S)
                return
            except Exception as e:
                last_error = e
                time.sleep(_RETRY_SLEEP_S)

        raise RuntimeError(
            f"Failed to communicate with robot '{arm}' at {self._robot_ip(arm)}: {last_error}"
        )

    def disconnect(self) -> None:
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

        obs: RobotObservation = {}
        for arm in self.active_arms:
            joints: np.ndarray = self.robot_manager.current_joint_positions(arm)
            for i, value in enumerate(joints, start=1):
                obs[f"{arm}_joint_{i}"] = value
            obs[f"{arm}_gripper"] = self.grippers[arm].position
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Forward gripper + arm commands. Gripper send is non-blocking.

        For arm motion the parent owns PD shaping (joint mode) and the
        safety screen, so both observe the same velocities that get
        streamed to franky. One IPC round-trip per arm fetches the
        kinematic snapshot used by both PD and the screen.
        """
        for arm in self.active_arms:
            self.grippers[arm].move(action[f"{arm}_gripper"], blocking=False)

        kin_state = self.robot_manager.current_kinematic_state_batch(
            list(self.active_arms)
        )

        if self.use_ee_delta:
            ee_by_arm = {
                arm: np.array(
                    [action[f"{arm}_{axis}"] for axis in ("x", "y", "z", "roll", "pitch", "yaw")],
                    dtype=np.float64,
                )
                for arm in self.active_arms
            }
            ee_by_arm = self.safety.screen_ee_actions(ee_by_arm, kin_state)
            ee_by_arm = {
                arm: self.safety.clamp_ee_twist_magnitude(twist)
                for arm, twist in ee_by_arm.items()
            }
            self.robot_manager.move_ee_delta_batch(
                {arm: twist.tolist() for arm, twist in ee_by_arm.items()},
                asynchronous=True,
            )
            return action

        # Joint mode: PD turns the position target into a velocity, the
        # safety screen shapes that velocity, and the magnitude clamp is
        # the final hardware-safe ceiling.
        joint_velocities_by_arm: dict[str, np.ndarray] = {}
        for arm in self.active_arms:
            target = np.asarray(
                [action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)],
                dtype=np.float64,
            )
            q, dq, _, _ = kin_state[arm]
            joint_velocities_by_arm[arm] = JOINT_PD_KP * (target - q) - JOINT_PD_KD * dq

        joint_velocities_by_arm = self.safety.screen_joint_actions(
            joint_velocities_by_arm, kin_state
        )
        joint_velocities_by_arm = {
            arm: self.safety.clamp_joint_velocity_magnitude(vel)
            for arm, vel in joint_velocities_by_arm.items()
        }

        self.robot_manager.move_joint_velocity_batch(
            {arm: vel.tolist() for arm, vel in joint_velocities_by_arm.items()},
            asynchronous=True,
        )
        return action
