"""
Torque-domain OSC controller, replacing OSCVelocityController now that
BimanualFranka drives the arm via pylibfranka's start_torque_control()
instead of franky joint-velocity commands.

This mirrors robosuite's OperationalSpaceController.run_controller() almost
exactly (see osc.py): task-space PD wrench -> operational-space projection
via lambda = (J M^-1 J^T)^-1 -> torques = J^T @ F_task + nullspace torques.

Two differences from robosuite's osc.py, both driven by what pylibfranka
gives us:

1. No `+ torque_compensation` (robosuite adds gravity compensation itself,
   since MuJoCo's underlying torque actuators are not gravity-compensated).
   pylibfranka's `start_torque_control()` gravity-compensates internally
   (per the pylibfranka docs: "gravity is automatically compensated"), so
   adding `g` again here would double-compensate. Coriolis is *not*
   mentioned as auto-compensated, so we add `-C` explicitly (opposing the
   Coriolis torque, matching the standard robot-dynamics convention
   tau = M*qddot + C + g).

2. Where osc.py reads `self.mass_matrix` / `self.J_full` etc. off `self`
   (populated by the surrounding Controller/sim base class each `update()`),
   here those quantities arrive per-call as arguments, since they're read
   from a single pylibfranka RobotState snapshot on the RPyC server side
   (see pylibfranka_server.py._RobotSession._bundle_state) to keep M, C, g,
   J, and the pose all mutually consistent for one control tick.

Variable impedance mirrors robosuite's impedance_mode="variable_kp": kp is
settable per-call, kd is always re-derived as critically damped (or at the
given damping ratio).
"""

from __future__ import annotations

import numpy as np

# Torque-domain PD gains. NOT the same units/magnitude as the old
# velocity-domain DEFAULT_KP=3.5 in osc_velocity_controller.py -- these are
# task-space stiffness/damping now (N/m, Nm/rad and their *sqrt-derived*
# damping terms), same role as robosuite osc.py's default kp=150.
DEFAULT_KP = 150.0
DEFAULT_DAMPING_RATIO = 1.0

# Nullspace joint torque gain (Nm/rad), analogous role to robosuite's
# nullspace_torques() internal joint-space stiffness.
DEFAULT_NULLSPACE_KP = 10.0
DEFAULT_NULLSPACE_KD = 2.0 * np.sqrt(DEFAULT_NULLSPACE_KP)

# Final safety clip on the resulting joint torque command, Nm. Kept well
# under the FR3 continuous joint torque limits as a last-resort backstop;
# the primary safety envelope is set_collision_behavior() on the server.
DEFAULT_MAX_TAU = 15.0


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
    """Axis-angle (rad, 3-vector) rotation taking current -> goal. Same role
    as robosuite's orientation_error(goal_mat, current_mat)."""
    g = goal_quat_xyzw / max(float(np.linalg.norm(goal_quat_xyzw)), 1e-12)
    c = current_quat_xyzw / max(float(np.linalg.norm(current_quat_xyzw)), 1e-12)

    q_err = quat_xyzw_multiply(g, quat_xyzw_conjugate(c))
    q_err /= max(float(np.linalg.norm(q_err)), 1e-12)
    if q_err[3] < 0.0:
        q_err = -q_err  # shortest-path equivalent rotation

    v = q_err[:3]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-9:
        return 2.0 * v
    angle = 2.0 * np.arctan2(v_norm, float(np.clip(q_err[3], -1.0, 1.0)))
    return (v / v_norm) * angle


def opspace_matrices(mass_matrix: np.ndarray, J_full: np.ndarray, J_pos: np.ndarray, J_ori: np.ndarray):
    """Same decomposition as robosuite.utils.control_utils.opspace_matrices:
    lambda_full = (J M^-1 J^T)^-1 (with pinv fallback near singularities),
    and the corresponding position-only / orientation-only lambdas plus the
    nullspace projector N = I - Jbar @ J."""
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    def _lambda(J):
        lambda_inv = J @ mass_matrix_inv @ J.T
        try:
            return np.linalg.inv(lambda_inv)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lambda_inv)

    lambda_full = _lambda(J_full)
    lambda_pos = _lambda(J_pos)
    lambda_ori = _lambda(J_ori)

    Jbar = mass_matrix_inv @ J_full.T @ lambda_full
    n = mass_matrix.shape[0]
    nullspace_matrix = np.eye(n) - Jbar @ J_full

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


def nullspace_torques(
    mass_matrix: np.ndarray,
    nullspace_matrix: np.ndarray,
    q_target: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    kp: float = DEFAULT_NULLSPACE_KP,
    kd: float = DEFAULT_NULLSPACE_KD,
) -> np.ndarray:
    """Same role as robosuite's nullspace_torques(): a joint-space PD torque
    toward q_target, projected through the nullspace so it doesn't disturb
    the commanded task-space wrench."""
    pd = kp * (q_target - q) - kd * dq
    return nullspace_matrix.T @ (mass_matrix @ pd)


class OSCTorqueController:
    """Per-arm torque-domain OSC. One instance per arm (holds gain state)."""

    def __init__(
        self,
        num_joints: int,
        kp: np.ndarray | float = DEFAULT_KP,
        damping_ratio: float = DEFAULT_DAMPING_RATIO,
        nullspace_kp: float = DEFAULT_NULLSPACE_KP,
        max_tau: float = DEFAULT_MAX_TAU,
        uncouple_pos_ori: bool = True,
    ) -> None:
        self.num_joints = num_joints
        self.damping_ratio = float(damping_ratio)
        self.nullspace_kp = float(nullspace_kp)
        self.nullspace_kd = 2.0 * np.sqrt(self.nullspace_kp)
        self.max_tau = float(max_tau)
        self.uncoupling = bool(uncouple_pos_ori)
        self.set_impedance(kp, damping_ratio)

    @staticmethod
    def _as_six(kp: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(kp, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(6, float(arr))
        assert arr.shape == (6,), f"kp must be scalar or shape (6,), got {arr.shape}"
        return arr

    def set_impedance(self, kp: np.ndarray | float, damping_ratio: float | None = None) -> None:
        """Mirrors robosuite's impedance_mode="variable_kp": kp is set
        per-call, kd is always re-derived as critically damped (or at the
        given damping ratio)."""
        self.kp = self._as_six(kp)
        if damping_ratio is not None:
            self.damping_ratio = float(damping_ratio)
        self.kd = 2.0 * np.sqrt(self.kp) * self.damping_ratio

    def compute_tau(
        self,
        goal_pos: np.ndarray,
        goal_quat_xyzw: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat_xyzw: np.ndarray,
        ee_twist: np.ndarray,
        J: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        mass_matrix: np.ndarray,
        coriolis: np.ndarray,
        rot_fudge: float = 1.0,
        q_nullspace_target: np.ndarray | None = None,
        kp: np.ndarray | float | None = None,
        damping_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Args:
            goal_pos: (3,) desired EE position
            goal_quat_xyzw: (4,) desired EE orientation quaternion (x,y,z,w)
            ee_pos: (3,) current EE position
            ee_quat_xyzw: (4,) current EE orientation
            ee_twist: (6,) current EE spatial velocity [linear(3), angular(3)]
            J: (6, num_joints) Jacobian, rows = [linear; angular]
            q: (num_joints,) current joint config
            dq: (num_joints,) current joint velocities
            mass_matrix: (num_joints, num_joints) joint-space mass matrix,
                from the SAME RobotState snapshot as J/q/dq (see module
                docstring -- these must not be mixed across ticks)
            coriolis: (num_joints,) Coriolis force vector, same snapshot
            q_nullspace_target: (num_joints,) reference config for the
                redundant-DoF nullspace bias (e.g. home pose). If None, no
                nullspace term is added.
            kp/damping_ratio: optional per-call impedance override
                (variable impedance), same semantics as set_impedance().

        Returns:
            (num_joints,) joint torque command, NOT including gravity
            compensation (pylibfranka's torque-control mode adds that
            internally -- see module docstring).
        """
        if kp is not None:
            self.set_impedance(kp, damping_ratio)

        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        ee_twist = np.asarray(ee_twist, dtype=np.float64)
        J = np.asarray(J, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
        coriolis = np.asarray(coriolis, dtype=np.float64)
        assert J.shape[0] == 6, f"expected J with 6 task rows, got {J.shape}"

        pos_error = goal_pos - ee_pos
        ori_error = orientation_error_from_quats(goal_quat_xyzw, ee_quat_xyzw) * rot_fudge

        vel_pos_error = -ee_twist[:3]
        vel_ori_error = -ee_twist[3:]

        # F_r = kp * pos_err + kd * vel_err  (same structure as osc.py)
        desired_force = self.kp[0:3] * pos_error + self.kd[0:3] * vel_pos_error
        desired_torque = self.kp[3:6] * ori_error + self.kd[3:6] * vel_ori_error

        J_pos = J[:3, :]
        J_ori = J[3:, :]
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(mass_matrix, J, J_pos, J_ori)

        if self.uncoupling:
            decoupled_force = lambda_pos @ desired_force
            decoupled_torque = lambda_ori @ desired_torque
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = lambda_full @ desired_wrench

        # Gamma (without gravity/nullspace) = J^T * F - Coriolis.
        # No `+ gravity`: pylibfranka's start_torque_control() gravity-
        # compensates internally, so adding g here would double-compensate.
        tau = J.T @ decoupled_wrench - coriolis

        if q_nullspace_target is not None:
            tau = tau + nullspace_torques(
                mass_matrix,
                nullspace_matrix,
                np.asarray(q_nullspace_target, dtype=np.float64),
                q,
                dq,
                kp=self.nullspace_kp,
                kd=self.nullspace_kd,
            )

        # Uniform scale-down (not per-joint clip) to preserve direction --
        # per-joint clipping breaks the torque proportionality the OSC
        # projection solved for and can produce a stable, non-progressing
        # limit cycle once any single joint saturates.
        peak = float(np.max(np.abs(tau)))
        if peak > self.max_tau:
            tau = tau * (self.max_tau / peak)
        return tau