"""Torque-domain OSC controller — aligned 1:1 with robosuite's
OperationalSpaceController.run_controller() in "variable_kp" impedance mode.

Uses rotation MATRICES for orientation error (not quaternions) to match
robosuite exactly and avoid quaternion-composition-order bugs.

variable_kp semantics (see robosuite OperationalSpaceController.set_goal):
    kp is supplied as an action every tick, clipped to [kp_min, kp_max].
    kd is always derived as kd = 2*sqrt(kp) (critically damped) -- there is
    no independent damping_ratio action in this mode.
"""

from __future__ import annotations
import numpy as np

# Matches osc.py OperationalSpaceController defaults exactly.
DEFAULT_KP = 150.0
DEFAULT_KP_LIMITS = (0.0, 300.0)
DEFAULT_NULLSPACE_KP = 10.0
DEFAULT_NULLSPACE_KD = 2.0 * np.sqrt(DEFAULT_NULLSPACE_KP)
DEFAULT_MAX_TAU = 35.0  # Conservative uniform clamp; robot has no per-joint enforcement,
                         # so this is our only safety margin. Panda datasheet nominal
                         # per-joint limits range ~87 N*m (joints 1-4) down to ~12 N*m
                         # (joints 5-7) -- 35 sits well clear of the weak wrist joints
                         # while being far less choking than the old 15 N*m default.


def quat_xyzw_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw / max(float(np.linalg.norm(q_xyzw)), 1e-12)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x+y*y)],
    ], dtype=np.float64)


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0); w = 0.25 / s
        x = (m21 - m12) * s; y = (m02 - m20) * s; z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22); w = (m21 - m12) / s
        x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22); w = (m02 - m20) / s
        x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11); w = (m10 - m01) / s
        x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / max(float(np.linalg.norm(q)), 1e-12)


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """EXACT robosuite formula: 0.5*(rc1 x rd1 + rc2 x rd2 + rc3 x rd3),
    columns of the rotation matrices. Do NOT replace with a quaternion
    axis-angle error -- different convention, breaks rotation control."""
    rc1, rc2, rc3 = current[0:3, 0], current[0:3, 1], current[0:3, 2]
    rd1, rd2, rd3 = desired[0:3, 0], desired[0:3, 1], desired[0:3, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    """Exact port of control_utils.opspace_matrices -- uses pinv, not inv,
    on the lambda matrices (matches osc.py: inv on mass_matrix, pinv on the
    lambda inversions, for numerical stability against singular Jacobians)."""
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    def _lambda_inv(J):
        return J @ mass_matrix_inv @ J.T

    lambda_full = np.linalg.pinv(_lambda_inv(J_full))
    lambda_pos = np.linalg.pinv(_lambda_inv(J_pos))
    lambda_ori = np.linalg.pinv(_lambda_inv(J_ori))

    Jbar = mass_matrix_inv @ J_full.T @ lambda_full
    nullspace_matrix = np.eye(mass_matrix.shape[0]) - Jbar @ J_full
    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


def nullspace_torques(mass_matrix, nullspace_matrix, initial_joint, joint_pos, joint_vel,
                       joint_kp=DEFAULT_NULLSPACE_KP):
    """Exact port of control_utils.nullspace_torques -- joint_kv is always
    derived as sqrt(joint_kp)*2 here (matches osc.py exactly); this differs
    from critical damping of joint_kp itself (2*sqrt(joint_kp)) only in that
    osc.py uses this specific (slightly off-critical) formula intentionally,
    so we replicate it rather than "fixing" it."""
    joint_kv = np.sqrt(joint_kp) * 2
    pose_torques = mass_matrix @ (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    return nullspace_matrix.T @ pose_torques


class OSCTorqueController:
    """Torque-domain OSC controller, variable_kp impedance mode only.

    kp is a scalar action supplied every compute_tau() call (broadcast to
    all 6 pos/ori DOF, matching robosuite's nums2array(kp, 6) behavior for a
    scalar kp input). kd is always 2*sqrt(kp) -- critically damped, exactly
    as osc.py's variable_kp mode computes it in set_goal().
    """

    def __init__(self, num_joints, kp_limits=DEFAULT_KP_LIMITS, max_tau=DEFAULT_MAX_TAU,
                 uncouple_pos_ori=True, nullspace_kp=DEFAULT_NULLSPACE_KP):
        self.num_joints = num_joints
        self.kp_min, self.kp_max = float(kp_limits[0]), float(kp_limits[1])
        self.max_tau = float(max_tau)
        self.uncoupling = bool(uncouple_pos_ori)
        self.nullspace_kp = float(nullspace_kp)
        self.nullspace_kd = 2.0 * np.sqrt(self.nullspace_kp)

        self.goal_pos: np.ndarray | None = None
        self.goal_ori: np.ndarray | None = None
        self._reference_pos: np.ndarray | None = None
        self._reference_ori: np.ndarray | None = None

    def init_goal(self, ee_pos, ee_quat_xyzw):
        """Call once (e.g. right after home()) to fix the REFERENCE pose that
        all subsequent absolute-offset teleop deltas are applied on top of.
        This reference is NOT updated every tick -- see set_goal_from_offset."""
        self._reference_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        self._reference_ori = quat_xyzw_to_mat(np.asarray(ee_quat_xyzw, dtype=np.float64))
        self.goal_pos = self._reference_pos.copy()
        self.goal_ori = self._reference_ori.copy()

    def set_goal_from_offset(self, offset_dpos: np.ndarray, offset_dquat_xyzw: np.ndarray):
        """Teleop reports offset_dpos/offset_dquat_xyzw as an ABSOLUTE offset
        from teleop's own rest pose (re-sent fresh every tick, decaying back to
        zero/identity on release) -- NOT a per-tick incremental delta. The goal
        must be recomputed fresh from the fixed reference pose every call, never
        composed onto the previous goal -- composing every tick made the goal
        orientation run away, since the same offset got re-applied ~20x/sec
        instead of being a one-time step."""
        if self.goal_pos is None:
            raise RuntimeError("call init_goal() before set_goal_from_offset()")
        self.goal_pos = self._reference_pos + np.asarray(offset_dpos, dtype=np.float64)
        delta_mat = quat_xyzw_to_mat(np.asarray(offset_dquat_xyzw, dtype=np.float64))
        self.goal_ori = delta_mat @ self._reference_ori

    def compute_tau(self, ee_pos, ee_quat_xyzw, ee_twist, J, q, dq, mass_matrix, coriolis,
                     kp, kd, q_nullspace_target=None):
        """kp: scalar (broadcast to 6 DOF) or length-6 array. kd is always
        derived as 2*sqrt(kp), critically damped, matching osc.py's
        variable_kp mode -- there is no separate damping_ratio input."""
        kp_arr = np.asarray(kp, dtype=np.float64)
        kp6 = np.full(6, float(kp_arr)) if kp_arr.ndim == 0 else kp_arr
        kp6 = np.clip(kp6, self.kp_min, self.kp_max)
        if kd is None:
            kd6 = 2.0 * np.sqrt(kp6)
        else:
            kd_arr = np.asarray(kd, dtype=np.float64)
            kd6 = np.full(6, float(kd_arr)) if kd_arr.ndim == 0 else kd_arr

        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        current_ori = quat_xyzw_to_mat(np.asarray(ee_quat_xyzw, dtype=np.float64))
        ee_twist = np.asarray(ee_twist, dtype=np.float64)
        J = np.asarray(J, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64); dq = np.asarray(dq, dtype=np.float64)
        mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
        coriolis = np.asarray(coriolis, dtype=np.float64)

        position_error = self.goal_pos - ee_pos
        ori_error = orientation_error(self.goal_ori, current_ori)

        vel_pos_error = -ee_twist[:3]
        vel_ori_error = -ee_twist[3:]

        desired_force = kp6[0:3] * position_error + kd6[0:3] * vel_pos_error
        desired_torque = kp6[3:6] * ori_error + kd6[3:6] * vel_ori_error

        J_pos, J_ori = J[:3, :], J[3:, :]
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(mass_matrix, J, J_pos, J_ori)

        if self.uncoupling:
            decoupled_wrench = np.concatenate([lambda_pos @ desired_force, lambda_ori @ desired_torque])
        else:
            decoupled_wrench = lambda_full @ np.concatenate([desired_force, desired_torque])

        # No +gravity term: pylibfranka torque-control gravity-compensates
        # internally, unlike osc.py's self.torque_compensation (mujoco does not).
        tau = J.T @ decoupled_wrench - coriolis

        if q_nullspace_target is not None:
            tau = tau + nullspace_torques(
                mass_matrix, nullspace_matrix,
                np.asarray(q_nullspace_target, dtype=np.float64), q, dq,
                joint_kp=self.nullspace_kp,
            )

        peak = float(np.max(np.abs(tau)))
        if peak > self.max_tau:
            tau = tau * (self.max_tau / peak)

        return tau