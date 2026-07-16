"""Action safety screens for the bimanual Franka.

All checks are pure (action, kin_state) → action transforms applied before dispatch.
Currently implements a worktable brake; bimanual arm-repel is not yet implemented.
"""

import numpy as np

from .franka_process import KinematicSnapshot

WORKTABLE_HEIGHT = 0.11                  # m, base-frame Z of the worktable surface
CUSTOM_END_EFFECTOR_Z_EXTENSION = 0.00   # m, extra tool reach below the EE frame
WORKTABLE_DISTANCE_MIN = 0.03            # m, minimum clearance; downward velocity zeroed at/past this
WORKTABLE_MAX_DECEL = 0.5                # m/s², assumed deceleration for the braking envelope
WORKTABLE_VELOCITY_EPS = 1.0e-4          # m/s, ignore commands smaller than this (float noise)

JOINT_VELOCITY_MAX = 2.0    # rad/s, L2-norm ceiling on joint velocity commands
EE_LINEAR_VELOCITY_MAX = 1.0   # m/s
EE_ANGULAR_VELOCITY_MAX = 2.0  # rad/s


def _clamp_joint_velocity(velocity: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(velocity))
    return velocity * (JOINT_VELOCITY_MAX / norm) if norm > JOINT_VELOCITY_MAX else velocity


def _clamp_ee_twist(twist: np.ndarray) -> np.ndarray:
    twist = np.asarray(twist, dtype=np.float64).copy()
    linear, angular = twist[:3], twist[3:]
    lin_norm = float(np.linalg.norm(linear))
    if lin_norm > EE_LINEAR_VELOCITY_MAX:
        linear *= EE_LINEAR_VELOCITY_MAX / lin_norm
    ang_norm = float(np.linalg.norm(angular))
    if ang_norm > EE_ANGULAR_VELOCITY_MAX:
        angular *= EE_ANGULAR_VELOCITY_MAX / ang_norm
    twist[:3] = linear
    twist[3:] = angular
    return twist


class ActionSafetyScreen:
    """Screens arm actions before dispatch, braking toward-table motion."""

    def __init__(self, end_effector_z_extension: float = CUSTOM_END_EFFECTOR_Z_EXTENSION) -> None:
        if end_effector_z_extension < 0.0:
            raise ValueError("end_effector_z_extension must be non-negative.")
        self.end_effector_z_extension = float(end_effector_z_extension)

    def shape_ee(
        self,
        ee_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
    ) -> dict[str, np.ndarray]:
        ee_by_arm = self._apply_worktable_brake(ee_by_arm, kin_state, is_ee=True)
        return {arm: _clamp_ee_twist(twist) for arm, twist in ee_by_arm.items()}

    def shape_joint(
        self,
        joint_velocities_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
    ) -> dict[str, np.ndarray]:
        joint_velocities_by_arm = self._apply_worktable_brake(
            joint_velocities_by_arm, kin_state, is_ee=False
        )
        return {arm: _clamp_joint_velocity(vel) for arm, vel in joint_velocities_by_arm.items()}

    def _apply_worktable_brake(
        self,
        action_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
        is_ee: bool,
    ) -> dict[str, np.ndarray]:
        """Limit downward EE velocity to prevent table contact.

        Layer 1: cap downward speed at sqrt(2 * MAX_DECEL * clearance) so the arm can decelerate
        before reaching WORKTABLE_DISTANCE_MIN.
        Layer 2: if measured downward speed already exceeds the envelope, zero commanded downward vel.

        EE mode: only the z component is modified.
        Joint mode: the whole vector is uniformly scaled to preserve joint-space direction.
        """
        out: dict[str, np.ndarray] = {}
        for arm, action in action_by_arm.items():
            action = np.asarray(action, dtype=np.float64)
            _, dq, jacobian, ee_translation, _, _ = kin_state[arm]
            contact_z = float(np.asarray(ee_translation)[2]) - self.end_effector_z_extension
            jacobian = np.asarray(jacobian, dtype=np.float64)
            dq = np.asarray(dq, dtype=np.float64)

            safe_dist = contact_z - WORKTABLE_HEIGHT - WORKTABLE_DISTANCE_MIN
            v_envelope = 0.0 if safe_dist <= 0.0 else float(np.sqrt(2.0 * WORKTABLE_MAX_DECEL * safe_dist))
            v_actual_z = float(jacobian[2, :] @ dq)
            v_commanded_z = float(action[2]) if is_ee else float(jacobian[2, :] @ action)

            if v_commanded_z >= -WORKTABLE_VELOCITY_EPS:
                out[arm] = action
                continue

            v_target_z = max(v_commanded_z, -v_envelope)
            if -v_actual_z > v_envelope:
                v_target_z = max(v_target_z, 0.0)

            if v_target_z == v_commanded_z:
                out[arm] = action
                continue

            if is_ee:
                action = action.copy()
                action[2] = v_target_z
            else:
                # Uniform scale preserves joint-space direction while braking descent.
                scale = max(0.0, v_target_z / v_commanded_z)
                action = action * scale

            out[arm] = action
        return out
