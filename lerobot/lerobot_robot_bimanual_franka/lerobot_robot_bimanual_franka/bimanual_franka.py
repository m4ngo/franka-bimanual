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
from .wsg import WSG

# 7 degrees of freedom per Franka arm.
NUM_JOINTS = 7
# Schunk WSG default travel range, in millimeters. Commanded positions are
# clipped to this range before being forwarded to the gripper.
GRIPPER_MIN_MM = 0
GRIPPER_MAX_MM = 110

# Joint-velocity PD controller for tracking joint-position targets. Lives in
# the parent process (this file) rather than franka_process so the safety
# overlay can inspect/modify the same velocities that get streamed to franky.
JOINT_PD_KP = 2.5
JOINT_PD_KD = 0.2
# Hardware-safe ceiling on the L2 norm of the 7-DoF joint-velocity command,
# enforced AFTER the safety overlay so safety can never push past it.
JOINT_PD_VELOCITY_MAX = 2.0  # rad/s

# Safety parameters. The worktable safety brakes the downward (toward-the-
# table) component of the commanded EE motion. Three layers stack:
#   1. Linear soft brake inside DISTANCE_THRESHOLD: smooth slowdown
#      proportional to closeness, fully zeroed at DISTANCE_MIN.
#   2. Kinematic stopping-distance envelope (always active): caps the
#      commanded downward speed at sqrt(2 * MAX_DECEL * (dist - MIN)) so
#      the arm can always decelerate before reaching the table given the
#      assumed max deceleration. This is what makes fast commands safe.
#   3. Actual-velocity overspeed override: if the *measured* downward speed
#      exceeds the envelope (e.g. due to momentum from a prior fast command),
#      the commanded EE downward velocity is forced to zero so the arm
#      brakes maximally.
# Lateral, upward, and angular motion are always passed through unmodified.
WORKTABLE_HEIGHT = 0.32 # meters
WORKTABLE_DISTANCE_THRESHOLD = 0.02 # meters; how close to the table can you get before the soft brake starts?
WORKTABLE_DISTANCE_MIN = 0.02 # meters; minimum closeness to the table; downward velocity is forced to zero at/past this distance
# Maximum deceleration (m/s^2) we assume the arm can deliver. Used by the
# kinematic envelope: smaller values are more conservative (larger braking
# zone, lower allowed approach speed) and less prone to overshoot.
WORKTABLE_MAX_DECEL = 0.2
# Damping for the joint-space pseudoinverse Jacobian used to translate the
# desired EE-frame brake into a joint-velocity correction. Higher values keep
# the correction tame near singularities at the cost of biasing the EE
# direction.
WORKTABLE_JACOBIAN_DAMPING = 0.0

# left arm's position relative to right arm
RELATIVE_ARM_POSITION_ROTATION = np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
])
BIMANUAL_DISTANCE_THRESHOLD = 0.05 # meters; how close can arms be before starting to slow?
BIMANUAL_DISTANCE_MIN = 0.02 # meters; minimum closeness between arms; repel immediately at/past this distance
BIMANUAL_REPEL_FORCE = 1.0

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
        for arm in self.active_arms:
            self.robot_manager.add_robot(
                arm, self._server_ip(arm), self._robot_ip(arm), self._port(arm)
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

        For arm motion the parent process owns PD shaping (joint mode) and the
        safety overlay so that both observe the same velocities that get
        streamed to franky. One IPC round-trip per arm fetches the kinematic
        snapshot used by both PD and safety.
        """
        for arm in self.active_arms:
            pos = np.clip(action[f"{arm}_gripper"], GRIPPER_MIN_MM, GRIPPER_MAX_MM)
            self.grippers[arm].move(pos, blocking=False)

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
            ee_by_arm = self.enforce_ee_safety(ee_by_arm, kin_state)
            self.robot_manager.move_ee_delta_batch(
                {arm: twist.tolist() for arm, twist in ee_by_arm.items()},
                asynchronous=True,
            )
            return action

        # Joint mode: convert the position target into a velocity via PD,
        # apply the safety overlay, then clamp the magnitude.
        joint_velocities_by_arm: dict[str, np.ndarray] = {}
        for arm in self.active_arms:
            target = np.asarray(
                [action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)],
                dtype=np.float64,
            )
            q, dq, _, _ = kin_state[arm]
            joint_velocities_by_arm[arm] = JOINT_PD_KP * (target - q) - JOINT_PD_KD * dq

        joint_velocities_by_arm = self.enforce_joint_safety(
            joint_velocities_by_arm, kin_state
        )
        joint_velocities_by_arm = {
            arm: self._clamp_joint_velocity(vel)
            for arm, vel in joint_velocities_by_arm.items()
        }

        self.robot_manager.move_joint_velocity_batch(
            {arm: vel.tolist() for arm, vel in joint_velocities_by_arm.items()},
            asynchronous=True,
        )
        return action

    @staticmethod
    def _clamp_joint_velocity(velocity: np.ndarray) -> np.ndarray:
        """Scale the 7-DoF velocity vector down to the hardware-safe norm."""
        norm = float(np.linalg.norm(velocity))
        if norm > JOINT_PD_VELOCITY_MAX:
            return velocity * (JOINT_PD_VELOCITY_MAX / norm)
        return velocity

    # ------------------------------------------------------------------
    # Internal: Safety screening
    # ------------------------------------------------------------------
    def enforce_ee_safety(
        self,
        ee_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Apply worktable / bimanual safety to EE velocity actions.

        Returns a new dict with the same shape; safety checks may scale or
        zero individual components of each per-arm twist.
        """
        ee_by_arm = self._apply_worktable_brake(ee_by_arm, kin_state, is_ee=True)
        # ee_by_arm = self._apply_bimanual_repel(ee_by_arm, kin_state, is_ee=True)  # TODO
        return ee_by_arm

    def enforce_joint_safety(
        self,
        joint_velocities_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Apply worktable / bimanual safety to joint velocity actions.

        Returns a new dict with the same shape; safety checks operate in
        velocity space so they shape exactly what gets streamed to franky.
        """
        joint_velocities_by_arm = self._apply_worktable_brake(
            joint_velocities_by_arm, kin_state, is_ee=False
        )
        # joint_velocities_by_arm = self._apply_bimanual_repel(joint_velocities_by_arm, kin_state, is_ee=False)  # TODO
        return joint_velocities_by_arm

    def _apply_worktable_brake(
        self,
        action_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        is_ee: bool,
    ) -> dict[str, np.ndarray]:
        """Brake the toward-the-table component of the commanded velocity.

        Three layers, applied in order on the EE-frame downward velocity:

            v_ee_z_target = v_ee_z * (1 - soft_factor)         # soft brake
            v_ee_z_target = max(v_ee_z_target, -v_envelope)    # envelope cap
            if -v_actual_z > v_envelope:                       # overspeed
                v_ee_z_target = max(v_ee_z_target, 0.0)        # active brake

        - Soft brake: linear ramp from 0 at WORKTABLE_DISTANCE_THRESHOLD to 1
          at WORKTABLE_DISTANCE_MIN, scaling down the commanded downward
          velocity. Gives smooth UX as the EE approaches the table.
        - Kinematic envelope: hard cap on commanded downward speed equal to
          sqrt(2 * MAX_DECEL * (dist - MIN)). Guarantees the arm can always
          stop before the table assuming the assumed max deceleration.
        - Overspeed override: if the *measured* downward speed (J @ dq from
          the kinematic snapshot) exceeds the envelope, the command is
          forced non-negative so the controller maximally brakes the
          existing momentum. This catches the case of a prior fast command
          carrying the arm into the soft zone too quickly.

        Lateral, upward, and angular motion pass through untouched, so the
        operator can still slide along or pull away from the table near
        the limit.

        - is_ee=True: action is a 6-vector EE twist in the arm's base frame.
          The third entry (linear z) is overwritten with v_ee_z_target.
        - is_ee=False: action is a 7-vector joint velocity. The induced EE
          z-velocity is computed via J @ dq; the matching joint correction
          is found by mapping the EE-frame delta back through the damped
          pseudoinverse Jacobian.
        """
        out: dict[str, np.ndarray] = {}
        for arm, action in action_by_arm.items():
            action = np.asarray(action, dtype=np.float64)
            _, dq, ee_translation, jacobian = kin_state[arm]
            ee_z = float(np.asarray(ee_translation)[2])
            jacobian = np.asarray(jacobian, dtype=np.float64)
            dq = np.asarray(dq, dtype=np.float64)

            v_envelope = self._worktable_velocity_envelope(ee_z)
            soft_factor = self._worktable_brake_factor(ee_z)
            v_actual_z = float(jacobian[2, :] @ dq)
            if is_ee:
                v_commanded_z = float(action[2])
            else:
                v_commanded_z = float(jacobian[2, :] @ action)

            v_target_z = v_commanded_z
            # Layer 1: soft brake on commanded downward motion.
            if v_target_z < 0.0 and soft_factor > 0.0:
                v_target_z = v_target_z * (1.0 - soft_factor)
            # Layer 2: kinematic stopping-distance envelope.
            if v_target_z < -v_envelope:
                v_target_z = -v_envelope
            # Layer 3: overspeed override on measured velocity.
            if -v_actual_z > v_envelope:
                v_target_z = max(v_target_z, 0.0)

            if v_target_z == v_commanded_z:
                out[arm] = action
                continue

            if is_ee:
                action = action.copy()
                action[2] = v_target_z
            else:
                # Minimum-norm joint correction whose forward kinematics is
                # the desired EE z-velocity delta; other EE directions are
                # left ~unchanged thanks to J @ J^+ ≈ I in the z row.
                delta_v_ee_z = v_target_z - v_commanded_z
                delta_v_ee = np.array(
                    [0.0, 0.0, delta_v_ee_z, 0.0, 0.0, 0.0], dtype=np.float64
                )
                delta_dq = self._damped_pinv(jacobian) @ delta_v_ee
                action = action + delta_dq

            out[arm] = action
        return out

    @staticmethod
    def _worktable_brake_factor(ee_z: float) -> float:
        """Soft-brake factor in [0, 1] based on EE height above the worktable.

        - >= WORKTABLE_DISTANCE_THRESHOLD above the table: 0 (no brake).
        - <= WORKTABLE_DISTANCE_MIN above the table (or below): 1 (downward
          velocity fully zeroed by this layer alone).
        - In between: linear ramp.
        """
        dist = ee_z - WORKTABLE_HEIGHT
        if dist >= WORKTABLE_DISTANCE_THRESHOLD:
            return 0.0
        if dist <= WORKTABLE_DISTANCE_MIN:
            return 1.0
        span = WORKTABLE_DISTANCE_THRESHOLD - WORKTABLE_DISTANCE_MIN
        return (WORKTABLE_DISTANCE_THRESHOLD - dist) / span

    @staticmethod
    def _worktable_velocity_envelope(ee_z: float) -> float:
        """Maximum safe downward EE Z-speed (m/s) at the given EE height.

        Derived from the kinematic stopping distance d_stop = v^2 / (2a):
        bounding the commanded downward speed by sqrt(2 * MAX_DECEL * (dist -
        MIN)) guarantees the arm can decelerate to zero before reaching
        WORKTABLE_DISTANCE_MIN of the table, assuming WORKTABLE_MAX_DECEL is
        achievable. Returns 0 at or below DISTANCE_MIN.
        """
        safe_dist = ee_z - WORKTABLE_HEIGHT - WORKTABLE_DISTANCE_MIN
        if safe_dist <= 0.0:
            return 0.0
        return float(np.sqrt(2.0 * WORKTABLE_MAX_DECEL * safe_dist))

    @staticmethod
    def _damped_pinv(
        jacobian: np.ndarray, damping: float = WORKTABLE_JACOBIAN_DAMPING
    ) -> np.ndarray:
        """Damped least-squares pseudoinverse of a 6x7 Jacobian.

        Returns J^T (J J^T + damping^2 I)^-1 (shape 7x6). DLS keeps joint
        motions bounded near kinematic singularities at the cost of biasing
        the EE direction; cheap and robust for safety overlays.
        """
        jjt = jacobian @ jacobian.T
        return jacobian.T @ np.linalg.inv(jjt + (damping ** 2) * np.eye(jjt.shape[0]))

    # def _apply_bimanual_repel(
    #     self,
    #     action_by_arm: dict[str, np.ndarray],
    #     kin_state: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    #     is_ee: bool,
    # ) -> dict[str, np.ndarray]:
    #     """Push the arms apart when their EEs get too close. Returns
    #     modified actions with the same shape as the input. Not yet
    #     implemented.
    #     """
    #     pass
