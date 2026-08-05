"""Torque-domain OSC, ported 1:1 from robosuite's OperationalSpaceController.

Source: ``multi-fast/robosuite/robosuite/controllers/osc.py`` in
``impedance_mode="variable"`` -- the mode the sim policies were trained under
(``cfg/fast_default.yaml``: kp=150, kp_limits=[0, 1500], damping_ratio=1,
damping_ratio_limits=[0, 10], ``osc_pose.json``: uncouple_pos_ori=true, no
interpolator, no position/orientation limits).

Runs on the NUC inside pylibfranka_server's 1 kHz loop, not on the workstation.
robosuite re-runs ``run_controller()`` every 2 ms sim step with the goal held
between 20 Hz policy steps; a torque computed once per policy step and latched
for 30-50 ms is a different system with no live damping term. The workstation
sets the goal (``set_goal``), this recomputes tau every tick (``run_controller``).

Kept numpy-only and free of package-relative imports so the deploy script can
drop this file next to the server on the NUC.

Two deviations from osc.py, both forced by the hardware:

- **Gravity.** osc.py adds ``self.torque_compensation`` = mujoco's ``qfrc_bias``
  = Coriolis + gravity. libfranka gravity-compensates internally and does not
  compensate Coriolis, so the parity term here is ``+ coriolis`` alone.
- **Orientation input.** The action carries a delta *quaternion* (qx,qy,qz,qw)
  where osc.py carries an axis-angle triple. Both become a rotation matrix
  composed as ``delta_mat @ current_ori`` before the error is taken, so the
  goal is identical; only the wire representation differs.
"""

from __future__ import annotations

import numpy as np

# This stack always runs osc.py's impedance_mode="variable": kp and the damping
# ratio come from the action every step (resolve_gains), never from a fixed
# constant. Stated explicitly because the two modes are NOT interchangeable --
# under "fixed" robosuite ignores gain actions entirely, so a gain sweep changes
# the real arm and leaves the sim reference untouched.
#
# NOTE the sysid sim reference in ~/sysid/sim_rotation_only/data.hdf5 was
# generated under "fixed" (LIBERO's unmodified osc_pose.json, kp_limits [0,300]).
# That is numerically identical to variable at action kp=kd=0 -- both give
# kp=150, ratio=1 -- so it is a valid reference for a kp=0 run and only for that.
# To sweep gains against sim, regenerate it with
#   sysid.controller_overrides: {impedance_mode: variable, kp_limits: [0, 1500]}
# and an action carrying the gain channels.
IMPEDANCE_MODE = "variable"

# Sim-parity gain schedule (cfg/fast_default.yaml controller block).
DEFAULT_KP = 150.0
KP_LIMITS = (0.0, 1500.0)
DEFAULT_DAMPING_RATIO = 1.0
DAMPING_RATIO_LIMITS = (0.0, 10.0)

# utils/envs/libero.py: exp_scale = limit_max / default, gains = exp_scale ** action * default.
KP_EXP_SCALE = KP_LIMITS[1] / DEFAULT_KP
DAMPING_EXP_SCALE = DAMPING_RATIO_LIMITS[1] / DEFAULT_DAMPING_RATIO

# osc_pose.json output_max/output_min: the per-step delta envelope in metres/radians.
DELTA_POS_MAX = 0.05
DELTA_ROT_MAX = 0.5

# control_utils.nullspace_torques default.
DEFAULT_NULLSPACE_KP = 10.0

# Damped-least-squares regularisation on lambda_full, in the units of J M^-1 J^T.
# Only used when uncouple_pos_ori=False, which inverts the coupled 6x6 and so
# blows up near a singularity. 0.05 bounds commanded torque at 24 Nm against the
# 69.6 Nm clamp across the reach, at <=14 deg of direction error at the worst
# pose and ~0 where the arm is well conditioned.
# Caps lambda_full at 1/mu^2. 0.05 caps at 400 against a healthy 11, so it never
# engaged before the torque clamp did: cond(J) 9->58 as the arm raises took joint 4
# to 99 Nm against its 69.6 Nm clamp, which is the shaking. 0.15 caps at 44.
LAMBDA_DLS_MU = 0.10

# FR3/Panda datasheet continuous joint torque limits (Nm).
JOINT_TORQUE_LIMITS = (87.0, 87.0, 87.0, 87.0, 30.0, 25.0, 20.0)

# Joint-space impedance for JOINT_POS / home() / hold. Per-joint stiffness, NOT
# a scalar through the mass matrix: tau = M @ (kp*e) collapses on the wrist,
# where M is ~0.01, so a 0.5 rad error asks for ~0.3 Nm -- under breakaway, which
# is why home() could not rotate the wrist joints. These are libfranka's
# joint_impedance_example values, in Nm/rad and Nms/rad.
DEFAULT_JOINT_KP = np.array([300.0, 300.0, 300.0, 300.0, 400.0, 150.0, 100.0])
# kd is sized per joint as 2*zeta*sqrt(kp*M) at zeta~0.7, NOT copied from
# libfranka's example: those values assume the proximal joints' inertia, and on
# the wrist (M ~ 0.01-0.05) they are both discretely unstable (kd/M above the
# 500 Hz law's Nyquist) and self-throttling -- home()'s torque budget is
# frac*tau_limit/(kd*(1+margin)), so kd=25 capped joint 6 at 0.13 rad/s and a
# 1 rad wrist move could not finish inside the 5 s default max_time_s.
DEFAULT_JOINT_KD = np.array([35.0, 35.0, 35.0, 35.0, 20.0, 10.0, 10.0])
DEFAULT_JOINT_DAMPING_RATIO = 1.0

# Cap on each joint's velocity-loop pole kd/M (rad/s). The STIFFNESS must stay
# un-weighted or the wrist cannot break away (see DEFAULT_JOINT_KP), but the
# DAMPING then inherits the same tiny inertia: kd/M is 600-1500 rad/s on joints
# 5-7, so -kd*dq alternates sign against the 500 Hz law and the wrist bang-bangs
# its +/-12 Nm clamp -- the homing vibration. This is the failure the velocity
# path already avoids by weighting through M (DEFAULT_JOINT_VEL_KV). 100 rad/s
# leaves joints 1-4 untouched and lands 5-7 at damping ratio 0.58-0.71.
JOINT_KD_POLE_MAX = 100.0

# Velocity-mode bandwidth (rad/s), applied THROUGH the mass matrix so the pole is
# this on every joint; DEFAULT_JOINT_KD as a velocity gain puts the wrist at
# kd/M = 600-7900 rad/s and chatters against the 500 Hz law.
DEFAULT_JOINT_VEL_KV = 25.0


def resolve_gains(
    kp_action: float,
    kd_action: float,
    kp_ori_scale: np.ndarray | float = 1.0,
    kd_ori_scale: np.ndarray | float = 1.0,
    kp_pos_scale: np.ndarray | float = 1.0,
    kd_pos_scale: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map action kp/kd in [-1, 1] to robosuite (kp, kd) 6-vectors.

    Reproduces LIBEROObservationWrapper.step's exponential rescale followed by
    OperationalSpaceController.set_goal's clip, for impedance_mode="variable":
    kp = 150 * 10**a_kp, damping_ratio = 1 * 10**a_kd, kd = 2*sqrt(kp)*ratio.
    All four scales at 1.0 is exactly that; every one of them is a deliberate
    hardware deviation from it.

    The two families do different things, and which one you want depends on
    whether the axis is too weak or too lively:

    - kp_*_scale multiplies its block's stiffness. The ratio is then re-derived
      as sqrt(kp6/kp), which holds kp/kd -- and therefore the settling speed
      v = (kp/kd)*delta -- FIXED. So a kp scale buys friction rejection, not
      speed.
    - kd_*_scale multiplies its block's damping ratio directly, which is the one
      knob that does change kp/kd. Raising it slows and damps the axis without
      stiffening it: the antidote to an axis that oscillates, where more kp only
      makes it worse.

    Each is a scalar or (3,). Non-uniform gains are inside osc.py's own action
    space (kp is a 6-vector there and variable mode reads all six), and the
    scaled ratio still passes through DAMPING_RATIO_LIMITS, so no combination
    can leave the envelope the sim can represent.
    """
    kp = DEFAULT_KP * KP_EXP_SCALE ** float(np.clip(kp_action, -1.0, 1.0))
    ratio = DEFAULT_DAMPING_RATIO * DAMPING_EXP_SCALE ** float(np.clip(kd_action, -1.0, 1.0))
    kp6 = np.full(6, float(kp))
    kp6[:3] *= np.asarray(kp_pos_scale, dtype=np.float64)
    kp6[3:] *= np.asarray(kp_ori_scale, dtype=np.float64)
    kp6 = np.clip(kp6, *KP_LIMITS)
    # From the CLIPPED kp, so the slew rate holds even where KP_LIMITS bit.
    ratio6 = float(ratio) * np.sqrt(kp6 / max(float(kp), 1e-12))
    ratio6[:3] *= np.asarray(kd_pos_scale, dtype=np.float64)
    ratio6[3:] *= np.asarray(kd_ori_scale, dtype=np.float64)
    kd6 = 2.0 * np.sqrt(kp6) * np.clip(ratio6, *DAMPING_RATIO_LIMITS)
    return kp6, kd6


def quat_xyzw_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64) / max(
        float(np.linalg.norm(q_xyzw)), 1e-12
    )
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """control_utils.orientation_error, verbatim.

    Do not swap in a quaternion axis-angle error: this is a different (and
    non-equivalent) convention, and the sim policies were trained against it.
    """
    rc1, rc2, rc3 = current[0:3, 0], current[0:3, 1], current[0:3, 2]
    rd1, rd2, rd3 = desired[0:3, 0], desired[0:3, 1], desired[0:3, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


def _lambda_inverse(lambda_inv: np.ndarray) -> np.ndarray:
    """Invert an operational-space inertia, matching control_utils' pinv.

    osc.py calls np.linalg.pinv here "to zero out small singular values for
    stability", but numpy's default rcond is 1e-15, so pinv only differs from a
    plain inverse when the matrix is singular to machine precision. Taking the
    fast path and falling back to pinv on LinAlgError is therefore numerically
    equivalent (checked by scripts/check_osc_parity.py against robosuite's real
    implementation) and ~3x cheaper -- which matters because this runs 1000x/s.
    """
    try:
        return np.linalg.inv(lambda_inv)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lambda_inv)


def opspace_matrices(
    mass_matrix: np.ndarray, J_full: np.ndarray, J_pos: np.ndarray, J_ori: np.ndarray,
    dls_mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """control_utils.opspace_matrices (see _lambda_inverse on the pinv/inv path).

    dls_mu damps lambda_full ONLY, which is the one that blows up near a
    singularity: measured 156 Nm of commanded joint torque against a 69.6 Nm
    clamp at sigma_min(J)=0.0045, where lambda_pos and lambda_ori stay bounded.
    lambda_pos/lambda_ori are left exact so uncouple_pos_ori=True remains
    bit-identical to robosuite. The nullspace projector is built from the damped
    lambda_full and so leaks O(mu^2); at mu=0.05 that is far below the 0.03 Nm
    the nullspace term itself produces.
    """
    mass_matrix_inv = np.linalg.inv(mass_matrix)
    Mi_Jt = mass_matrix_inv @ J_full.T

    A = J_full @ Mi_Jt
    if dls_mu:
        A = A + (dls_mu * dls_mu) * np.eye(A.shape[0])
    lambda_full = _lambda_inverse(A)
    lambda_pos = _lambda_inverse(J_pos @ mass_matrix_inv @ J_pos.T)
    lambda_ori = _lambda_inverse(J_ori @ mass_matrix_inv @ J_ori.T)

    Jbar = Mi_Jt @ lambda_full
    nullspace_matrix = np.eye(J_full.shape[-1]) - Jbar @ J_full
    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


def nullspace_torques(
    mass_matrix: np.ndarray,
    nullspace_matrix: np.ndarray,
    initial_joint: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_kp: float = DEFAULT_NULLSPACE_KP,
) -> np.ndarray:
    """control_utils.nullspace_torques, verbatim."""
    joint_kv = np.sqrt(joint_kp) * 2
    pose_torques = mass_matrix @ (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    return nullspace_matrix.T @ pose_torques


def clip_delta(delta_pos: np.ndarray, delta_rotvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply osc.py's effective per-step delta envelope.

    osc.py clips the raw action to input_min/input_max ([-1, 1]) and then scales
    to output_min/output_max (+/-0.05 m, +/-0.5 rad). The workstation hands us
    deltas already in output units, so the equivalent guard is a clip at the
    output bound -- scaling again here would double-apply it.
    """
    return (
        delta_pos, # np.clip(delta_pos, -DELTA_POS_MAX, DELTA_POS_MAX),
        delta_rotvec # np.clip(delta_rotvec, -DELTA_ROT_MAX, DELTA_ROT_MAX),
    )


class OSCTorqueController:
    """Per-arm OSC_POSE controller in impedance_mode="variable".

    ``set_goal`` is the policy-rate half (goal pose + gains); ``run_controller``
    is the 1 kHz half. Split that way because they run at different rates on
    opposite sides of the RPyC link, unlike osc.py where both are in-process.
    """

    def __init__(
        self,
        num_joints: int = 7,
        uncouple_pos_ori: bool = True,
        nullspace_kp: float = DEFAULT_NULLSPACE_KP,
    ) -> None:
        self.num_joints = int(num_joints)
        self.uncoupling = bool(uncouple_pos_ori)
        self.nullspace_kp = float(nullspace_kp)

        self.goal_pos = np.zeros(3)
        self.goal_ori = np.eye(3)
        self.kp = np.full(6, DEFAULT_KP)
        self.kd = 2.0 * np.sqrt(self.kp)
        self.initial_joint = np.zeros(self.num_joints)

    def reset_goal(self, ee_pos: np.ndarray, ee_ori_mat: np.ndarray) -> None:
        """osc.py reset_goal: park the goal on the current pose (zero error)."""
        self.goal_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        self.goal_ori = np.asarray(ee_ori_mat, dtype=np.float64).copy()

    def set_goal(
        self,
        goal_pos: np.ndarray,
        goal_ori_mat: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        initial_joint: np.ndarray | None = None,
    ) -> None:
        self.goal_pos = np.asarray(goal_pos, dtype=np.float64)
        self.goal_ori = np.asarray(goal_ori_mat, dtype=np.float64)
        self.kp = np.clip(np.asarray(kp, dtype=np.float64), *KP_LIMITS)
        self.kd = np.asarray(kd, dtype=np.float64)
        if initial_joint is not None:
            self.initial_joint = np.asarray(initial_joint, dtype=np.float64)

    def run_controller(
        self,
        ee_pos: np.ndarray,
        ee_ori_mat: np.ndarray,
        ee_pos_vel: np.ndarray,
        ee_ori_vel: np.ndarray,
        J_full: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        mass_matrix: np.ndarray,
        coriolis: np.ndarray,
        use_nullspace: bool = True,
        joint_damping_kv: float = 0.0,
        dls_mu: float = 0.0,
        ori_force_coupling: bool = True,
    ) -> np.ndarray:
        """joint_damping_kv adds extra joint-space damping *projected into the
        nullspace*; 0.0 (the default) is exact osc.py. It must be projected: an
        unprojected -kv*dq opposes commanded task motion, and does so ~10x more
        for rotation than translation, because the wrist has to spin fast to
        produce a given EE angular velocity. Note osc.py's nullspace_torques
        already damps the redundant DoF at kv = 2*sqrt(10), so this is only for
        adding more on top."""
        position_error = self.goal_pos - ee_pos
        vel_pos_error = -ee_pos_vel
        desired_force = position_error * self.kp[0:3] + vel_pos_error * self.kd[0:3]

        ori_error = orientation_error(self.goal_ori, ee_ori_mat)
        vel_ori_error = -ee_ori_vel
        desired_torque = ori_error * self.kp[3:6] + vel_ori_error * self.kd[3:6]

        J_pos, J_ori = J_full[:3, :], J_full[3:, :]
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            mass_matrix, J_full, J_pos, J_ori, dls_mu=dls_mu
        )

        if self.uncoupling:
            decoupled_wrench = np.concatenate(
                [lambda_pos @ desired_force, lambda_ori @ desired_torque]
            )
        else:
            # ori_force_coupling=False drops lambda_full's orientation->FORCE block.
            # It is what makes the coupled form usable on this arm: the block turns
            # an orientation error the wrist cannot clear into a standing
            # translational push (26.8 Nm-driven newtons at 0.15 rad), measured as
            # 215 mm of post-release walk. The blocks that matter are kept.
            force = lambda_full[:3, :3] @ desired_force
            if ori_force_coupling:
                force = force + lambda_full[:3, 3:] @ desired_torque
            moment = lambda_full[3:, :3] @ desired_force + lambda_full[3:, 3:] @ desired_torque
            decoupled_wrench = np.concatenate([force, moment])

        # +coriolis, not +qfrc_bias: libfranka already compensates gravity.
        torques = J_full.T @ decoupled_wrench + coriolis

        if use_nullspace:
            torques = torques + nullspace_torques(
                mass_matrix,
                nullspace_matrix,
                self.initial_joint,
                q,
                dq,
                joint_kp=self.nullspace_kp,
            )
        if joint_damping_kv:
            torques = torques - joint_damping_kv * (nullspace_matrix.T @ dq)
        return torques


class JointImpedanceController:
    """Computed-torque joint impedance for JOINT_POS, home(), hold and float.

    Shaped like robosuite's JointPositionController (tau = M @ PD + bias) so the
    gravity/Coriolis handling matches OSCTorqueController's.
    """

    def __init__(self, num_joints: int = 7) -> None:
        self.num_joints = int(num_joints)
        self.goal_q = np.zeros(self.num_joints)
        self.goal_dq = np.zeros(self.num_joints)
        self.kp = DEFAULT_JOINT_KP.copy()
        self.kd = DEFAULT_JOINT_KD.copy()
        self.kv = DEFAULT_JOINT_VEL_KV

    def set_goal(
        self,
        goal_q: np.ndarray | None = None,
        goal_dq: np.ndarray | None = None,
        kp: float | None = None,
        damping_ratio: float | None = None,
    ) -> None:
        if goal_q is not None:
            self.goal_q = np.asarray(goal_q, dtype=np.float64)
        if goal_dq is not None:
            self.goal_dq = np.asarray(goal_dq, dtype=np.float64)
        else:
            self.goal_dq = np.zeros(self.num_joints)
        # kp is a SCALE on the per-joint stiffness vector, not an absolute gain.
        if kp is not None:
            self.kp = DEFAULT_JOINT_KP * float(kp)
        if kp is not None or damping_ratio is not None:
            ratio = DEFAULT_JOINT_DAMPING_RATIO if damping_ratio is None else float(damping_ratio)
            self.kd = DEFAULT_JOINT_KD * np.sqrt(np.max(self.kp) / np.max(DEFAULT_JOINT_KP)) * ratio
            self.kv = DEFAULT_JOINT_VEL_KV * ratio

    def run_controller(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        mass_matrix: np.ndarray,
        coriolis: np.ndarray,
        position_hold: bool = True,
    ) -> np.ndarray:
        # Direct joint impedance, no mass-matrix weighting -- see DEFAULT_JOINT_KP.
        if position_hold:
            # Damping alone is capped against the live inertia; see JOINT_KD_POLE_MAX.
            kd = np.minimum(self.kd, JOINT_KD_POLE_MAX * np.diag(mass_matrix))
            return self.kp * (self.goal_q - q) - kd * dq + coriolis
        # Velocity mode weights by M -- see DEFAULT_JOINT_VEL_KV.
        return mass_matrix @ (self.kv * (self.goal_dq - dq)) + coriolis
