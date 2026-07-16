"""
Velocity-domain analogue of robosuite's OperationalSpaceController.

Robosuite's OSC computes a 6D task-space wrench from pos/ori PD error, maps it
through operational-space dynamics (lambda = (J M^-1 J^T)^-1) into torques, and
adds a nullspace *torque* term pulling toward a reference joint config. We have
no torque interface here -- BimanualFranka commands joint *velocities* -- so
this controller instead:

  1. Computes a 6D task-space *velocity* command as Kp * pose_error (pure
     proportional -- see note below on why there's no -Kd*twist term here).
  2. Maps desired task-space velocity to joint velocities via the damped-least-
     squares pseudo-inverse of J (6, num_joints).
  3. Adds a nullspace *velocity* bias pulling redundant joints toward a
     reference config, projected through (I - Jbar @ J) so it doesn't disturb
     commanded task-space motion -- the velocity-control equivalent of
     robosuite's nullspace_torques().

Why no -Kd*ee_twist term: torque-domain OSC needs it because it commands a
force on a mass -- an undamped P term there gives an undamped mass-spring
system, and -Kd*twist is what makes it stable. Here qdot IS the control input
(no mass in the loop), so pure P is already first-order and stable by
construction; adding a raw-twist derivative term back in doesn't add physical
damping, it just destabilizes the discrete-time loop (verified: the resulting
difference equation has eigenvalues outside the unit circle at ordinary
control rates). ee_twist is still accepted by compute_qdot for interface
parity in case a feedforward term is added later, but it isn't used.

Variable impedance mirrors robosuite's impedance_mode="variable_kp": kp is
settable per-call, kd is always re-derived as critically damped (or at the
given damping ratio) for callers that want to read/log it, even though the
current law doesn't consume kd directly.
"""

from __future__ import annotations

import numpy as np

# Velocity-domain PD gains (not torque-domain), same order of magnitude as the
# EE_PD_KP/EE_PD_KD already used elsewhere in bimanual_franka.py.
DEFAULT_KP = 3.5
DEFAULT_DAMPING_RATIO = 1.0

# Nullspace joint-velocity gain (acts only through the projected nullspace,
# so it never fights the commanded task-space motion).
DEFAULT_NULLSPACE_KP = 1.0

# Damped-least-squares singularity damping term used in the Jacobian pinv.
# Must stay well below the smallest singular value J normally has away from
# true kinematic singularities, or it introduces steady-state tracking error
# during ordinary motion (not just near singularities). 0.01 only meaningfully
# engages when a singular value drops below ~0.01, which is a real singularity.
DEFAULT_DLS_DAMPING = 0.01

# Final safety clip on the resulting joint velocity command, rad/s.
DEFAULT_MAX_QDOT = 2.5


def quat_xyzw_conjugate(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quat_xyzw_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """q1 * q2, both (x, y, z, w)."""
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def orientation_error_from_quats(goal_quat_xyzw: np.ndarray, current_quat_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle (rad, 3-vector) rotation taking current -> goal.

    Same role as robosuite's orientation_error(goal_mat, current_mat), computed
    from quaternions since that's what KinematicSnapshot carries.
    """
    g = goal_quat_xyzw / max(float(np.linalg.norm(goal_quat_xyzw)), 1e-12)
    c = current_quat_xyzw / max(float(np.linalg.norm(current_quat_xyzw)), 1e-12)

    q_err = quat_xyzw_multiply(g, quat_xyzw_conjugate(c))
    q_err /= max(float(np.linalg.norm(q_err)), 1e-12)
    if q_err[3] < 0.0:
        # Shortest-path equivalent rotation (positive scalar part).
        q_err = -q_err

    v = q_err[:3]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-9:
        return 2.0 * v
    angle = 2.0 * np.arctan2(v_norm, float(np.clip(q_err[3], -1.0, 1.0)))
    return (v / v_norm) * angle


class OSCVelocityController:
    """Per-arm velocity-domain OSC. One instance per arm (holds gain state)."""

    def __init__(
        self,
        num_joints: int,
        kp: np.ndarray | float = DEFAULT_KP,
        damping_ratio: float = DEFAULT_DAMPING_RATIO,
        nullspace_kp: float = DEFAULT_NULLSPACE_KP,
        dls_damping: float = DEFAULT_DLS_DAMPING,
        max_qdot: float = DEFAULT_MAX_QDOT,
    ) -> None:
        self.num_joints = num_joints
        self.damping_ratio = float(damping_ratio)
        self.nullspace_kp = float(nullspace_kp)
        self.dls_damping = float(dls_damping)
        self.max_qdot = float(max_qdot)
        self.set_impedance(kp, damping_ratio)

    @staticmethod
    def _as_six(kp: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(kp, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(6, float(arr))
        assert arr.shape == (6,), f"kp must be scalar or shape (6,), got {arr.shape}"
        return arr

    def reset_derivative_state(self) -> None:
        """No-op placeholder, kept for interface parity with reset_goal()-style
        calls at connect()/home() time (useful if filtered derivatives are added later)."""
        pass

    def set_impedance(self, kp: np.ndarray | float, damping_ratio: float | None = None) -> None:
        """Update kp (and optionally damping_ratio) for variable-impedance control.

        Mirrors OSC's impedance_mode="variable_kp": kp is set per-call from the
        action, kd is always re-derived as critically damped -- kd is never set
        directly.
        """
        self.kp = self._as_six(kp)
        if damping_ratio is not None:
            self.damping_ratio = float(damping_ratio)
        self.kd = 2.0 * np.sqrt(self.kp) * self.damping_ratio

    def compute_qdot(
        self,
        goal_pos: np.ndarray,
        goal_quat_xyzw: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat_xyzw: np.ndarray,
        ee_twist: np.ndarray,
        J: np.ndarray,
        q: np.ndarray,
        rot_fudge: float = 1.0,
        q_nullspace_target: np.ndarray | None = None,
        kp: np.ndarray | float | None = None,
        damping_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Args:
            goal_pos: (3,) desired EE position, same frame as ee_pos
            goal_quat_xyzw: (4,) desired EE orientation quaternion (x,y,z,w)
            ee_pos: (3,) current EE position
            ee_quat_xyzw: (4,) current EE orientation
            ee_twist: (6,) current EE spatial velocity [linear(3), angular(3)]
            J: (6, num_joints) Jacobian, rows = [linear; angular]
            q: (num_joints,) current joint config, needed for the nullspace term
            q_nullspace_target: (num_joints,) reference config for the redundant-
                DoF nullspace bias (e.g. home pose). If None, no nullspace term.
            kp: optional scalar or (6,) array to set impedance for this call
                (variable impedance). If None, uses the last-set self.kp.
            damping_ratio: optional override, paired with kp. If kp is given and
                this is None, damping_ratio stays at its last-set value (kd is
                still re-derived from the new kp).

        Returns:
            (num_joints,) joint velocity command.
        """
        if kp is not None:
            self.set_impedance(kp, damping_ratio)

        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_twist = np.asarray(ee_twist, dtype=np.float64)
        J = np.asarray(J, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        assert J.shape[0] == 6, f"expected J with 6 task rows, got {J.shape}"

        pos_error = goal_pos - ee_pos
        ori_error = orientation_error_from_quats(goal_quat_xyzw, ee_quat_xyzw) * rot_fudge
        pose_error = np.concatenate([pos_error, ori_error])

        # Pure proportional task-space velocity law. Unlike torque-domain OSC,
        # there is no -kd*ee_twist term here: qdot is the control input itself
        # (not a force on a mass), so the closed loop is already first-order
        # and exponentially stable under P alone. Adding a raw-twist derivative
        # term on top does not add physical damping in this system -- it
        # destabilizes the discrete-time loop instead. ee_twist is accepted as
        # an argument for interface parity / possible future use (e.g. feed-
        # forward compensation) but is intentionally not used in the law below.
        desired_task_vel = self.kp * pose_error

        # Damped-least-squares pseudo-inverse: Jbar = J^T (J J^T + damping^2 I)^-1
        # Robust near singularities, unlike a plain pinv.
        JJt = J @ J.T

        damped = JJt + (self.dls_damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(damped, np.eye(6))
        qdot = J_pinv @ desired_task_vel
        if q_nullspace_target is not None:
            # Nullspace projector N = I - Jbar @ J: a joint-velocity bias passed
            # through N only acts in directions that produce zero task-space
            # motion, so it can't fight the commanded EE velocity above. This is
            # the velocity-control analogue of robosuite's nullspace_torques().
            n = J.shape[1]
            nullspace_proj = np.eye(n) - J_pinv @ J
            q_bias = self.nullspace_kp * (np.asarray(q_nullspace_target, dtype=np.float64) - q)
            qdot = qdot + nullspace_proj @ q_bias

        # Scale the whole vector down to respect max_qdot, rather than clipping
        # each joint independently. Per-joint clipping breaks the proportionality
        # between joints that the pinv solved for, so once any single joint
        # saturates the resulting motion no longer points toward the goal at
        # all -- this can produce a stable limit cycle with zero net progress
        # whenever the goal is far enough to saturate. A uniform scale-down
        # preserves direction and just slows down the approach.
        peak = float(np.max(np.abs(qdot)))
        if peak > self.max_qdot:
            qdot = qdot * (self.max_qdot / peak)
        return qdot
