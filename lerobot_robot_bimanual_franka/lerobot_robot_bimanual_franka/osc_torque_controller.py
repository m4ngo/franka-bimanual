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

try:  # package on the workstation, flat next to the server on the NUC
    from .torque_config import torque
except ImportError:
    from torque_config import torque  # type: ignore[no-redef]

# Every constant below comes from config/control.yaml (`torque:`); the rationale
# for each value lives there. Redeploy the NUC after changing one.
#
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
IMPEDANCE_MODE = torque("osc.impedance_mode")

# Sim-parity gain schedule (cfg/fast_default.yaml controller block).
DEFAULT_KP = float(torque("osc.default_kp"))
KP_LIMITS = tuple(float(v) for v in torque("osc.kp_limits"))
DEFAULT_DAMPING_RATIO = float(torque("osc.default_damping_ratio"))
DAMPING_RATIO_LIMITS = tuple(float(v) for v in torque("osc.damping_ratio_limits"))

# utils/envs/libero.py: exp_scale = limit_max / default, gains = exp_scale ** action * default.
# DERIVED, never configured separately -- a second copy silently drifts from sim.
KP_EXP_SCALE = KP_LIMITS[1] / DEFAULT_KP
DAMPING_EXP_SCALE = DAMPING_RATIO_LIMITS[1] / DEFAULT_DAMPING_RATIO

# osc_pose.json output_max/output_min: the per-step delta envelope in metres/radians.
DELTA_POS_MAX = float(torque("delta.pos_max_m"))
DELTA_ROT_MAX = float(torque("delta.rot_max_rad"))

# control_utils.nullspace_torques default.
DEFAULT_NULLSPACE_KP = float(torque("osc.nullspace_kp"))

# osc.py's uncoupling switch, applied to SIM's mass matrix when emulating.
DEFAULT_UNCOUPLE_POS_ORI = bool(torque("osc.uncouple_pos_ori"))

# Singularity conditioning on lambda_full. 0.0 is np.linalg.pinv's own default, i.e.
# exactly the previously shipped behaviour; see config/control.yaml for the measurement
# and for why this is not a limit layer.
LAMBDA_RCOND = float(torque("osc.lambda_rcond"))
# Cancel the rotation->translation leak uncoupling leaves behind. Only read when
# uncouple_pos_ori is true; false = osc.py exactly. See run_controller.
CROSS_COUPLING_COMPENSATION = bool(torque("osc.cross_coupling_compensation"))
LAMBDA_LENGTH_SCALE = float(torque("osc.lambda_length_scale_m"))
# diag(1, 1, 1, L, L, L): makes the 6x6 unit-consistent so one rcond means the same
# thing at every pose. Held as a vector because the scaling is applied by broadcasting.
_LAMBDA_SCALE = np.array([1.0, 1.0, 1.0] + [LAMBDA_LENGTH_SCALE] * 3)

# mujoco ctrlrange on robosuite's Panda actuators -- part of the REFERENCE dynamics,
# not a hardware bound: sim's rotation authority saturates here long before the FR3's
# does, which is why sim's rotation travel per step FALLS with command amplitude
# (0.24 -> 0.125). Applied inside the law, upstream of _enforce_limits, and only when
# emulating. Distinct from limits.joint_torque_nm, which bounds the real joints.
SIM_TORQUE_LIMITS = np.asarray(torque("osc.sim_ctrlrange_nm"), dtype=np.float64)

# mujoco dof_damping on the same joints -- the rest of sim's plant. Checked against
# the live model by tests/test_sim_dynamics.py.
SIM_JOINT_DAMPING = np.asarray(torque("osc.sim_joint_damping_nms_rad"), dtype=np.float64)

# FR3/Panda datasheet continuous joint torque limits (Nm).
JOINT_TORQUE_LIMITS = tuple(float(v) for v in torque("limits.joint_torque_nm"))

# Joint-space impedance for JOINT_POS / home() / hold.
DEFAULT_JOINT_KP = np.asarray(torque("joint_impedance.kp"), dtype=np.float64)
DEFAULT_JOINT_KD = np.asarray(torque("joint_impedance.kd"), dtype=np.float64)
DEFAULT_JOINT_DAMPING_RATIO = float(torque("joint_impedance.damping_ratio"))

# Cap on each joint's velocity-loop pole kd/M (rad/s).
JOINT_KD_POLE_MAX = float(torque("joint_impedance.kd_pole_max_rad_s"))

# Velocity-mode bandwidth (rad/s), applied THROUGH the mass matrix.
DEFAULT_JOINT_VEL_KV = float(torque("joint_impedance.velocity_kv"))


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

    Written out in scalars rather than as three np.cross calls purely for speed:
    np.cross costs ~6.6 us on a 3-vector and this sits on the 500 Hz law's path,
    where the whole tick budget is ~1 ms. Same three products accumulated in the
    same order, so it is bit-identical -- checked at 0.0e+00 against the np.cross
    form over 5000 random rotation pairs. Do not "simplify" it back.
    """
    x = y = z = 0.0
    for i in range(3):                    # robosuite pairs the matrices COLUMN-wise
        c0, c1, c2 = current[0, i], current[1, i], current[2, i]
        d0, d1, d2 = desired[0, i], desired[1, i], desired[2, i]
        x += c1 * d2 - c2 * d1
        y += c2 * d0 - c0 * d2
        z += c0 * d1 - c1 * d0
    return np.array([0.5 * x, 0.5 * y, 0.5 * z])


def _lambda_inverse(lambda_inv: np.ndarray) -> np.ndarray:
    """control_utils' pinv, verbatim.

    This used to take np.linalg.inv with a pinv fallback on LinAlgError, on the
    argument that numpy's default rcond is 1e-15 so the two only differ at exact
    singularity. That argument is wrong in the direction that matters: inv does
    not RAISE on an ill-conditioned matrix, it returns an amplified result, so the
    fallback never fired where it was needed. Near a wrist singularity that is a
    torque spike on real hardware and a zeroed direction in sim -- the opposite
    behaviour, at the pose most likely to fault.
    """
    return np.linalg.pinv(lambda_inv)


def _lambda_inverse_full(lambda_inv: np.ndarray, rcond: float) -> tuple[np.ndarray, int]:
    """`_lambda_inverse` for the 6x6, with the singular-value truncation
    control_utils only claims to do. Returns (lambda_full, directions dropped).

    LAMBDA_RCOND 0.0 is `_lambda_inverse` bit-for-bit, which is what every parity
    test asserts against; see config/control.yaml for why the default is not 0.

    The scaling is not cosmetic. rcond is relative to the largest singular value,
    and J's angular rows carry 1/m against its linear ones, so `J M^-1 J^T` mixes
    kg with kg m^2 and reads cond 77 at home_pose where nothing is singular at
    all. Scaling the angular block by LAMBDA_LENGTH_SCALE first makes one rcond
    mean the same thing at every pose; `S pinv(S A S) S` is exactly `pinv(A)`
    wherever nothing is truncated, since S is diagonal and invertible.
    """
    if rcond <= 0.0:
        return np.linalg.pinv(lambda_inv), 0
    scaled = _LAMBDA_SCALE[:, None] * lambda_inv * _LAMBDA_SCALE[None, :]
    # One SVD, not two: np.linalg.pinv takes its own, and asking for the singular
    # values separately to count the truncation cost ~9 us of the law's ~1 ms.
    # This is pinv spelled out -- V diag(1/s) U^T with the small s zeroed.
    u, sv, vh = np.linalg.svd(scaled)
    keep = sv >= rcond * sv[0]
    dropped = int(np.count_nonzero(~keep))
    inv = (vh.T * np.where(keep, 1.0 / sv, 0.0)) @ u.T
    return _LAMBDA_SCALE[:, None] * inv * _LAMBDA_SCALE[None, :], dropped


def opspace_matrices(
    mass_matrix: np.ndarray, J_full: np.ndarray, J_pos: np.ndarray, J_ori: np.ndarray,
    lambda_rcond: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """control_utils.opspace_matrices, verbatim except for the 6x6's conditioning
    (see `_lambda_inverse_full`; at LAMBDA_RCOND 0.0 this is verbatim).

    The 3x3 blocks are left on the plain pinv on purpose: each inverts a 3x7
    Jacobian's Gram matrix, so it keeps four DOF of redundancy and measures cond
    3-17 over the whole envelope, against up to ~950 for the 6x6. They are also
    single-unit, so there is nothing for a length scale to fix.

    ``lambda_rcond`` defaults to 0.0 -- robosuite exactly -- so a caller that does
    not opt in cannot be silently conditioned. The control loop passes
    ``torque.osc.lambda_rcond`` by name; the parity harness passes nothing.

    The truncation count is returned rather than logged: this runs at 500 Hz on
    the NUC, and the caller accumulates it into a counter the workstation can read.

    Returns one element more than control_utils does -- the coupling block
    ``Jv M^-1 Jw^T`` -- because ``cross_coupling_compensation`` needs it and it is
    free here. Existing positional unpackings of the first four are unaffected.
    """
    mass_matrix_inv = np.linalg.inv(mass_matrix)
    Mi_Jt = mass_matrix_inv @ J_full.T

    lambda_full, dropped = _lambda_inverse_full(J_full @ Mi_Jt, lambda_rcond)
    lambda_pos = _lambda_inverse(J_pos @ mass_matrix_inv @ J_pos.T)
    lambda_ori = _lambda_inverse(J_ori @ mass_matrix_inv @ J_ori.T)
    # Jv M^-1 Jw^T: the translation/rotation coupling block that uncoupling DISCARDS.
    # Returned rather than recomputed by the caller because mass_matrix_inv is already
    # here, and it takes no inverse of its own -- which is the whole reason the
    # compensation below can be bounded where lambda_full is not.
    coupling = J_pos @ mass_matrix_inv @ J_ori.T

    # Jbar carries the same truncation, which is the point: an unbounded nullspace
    # projector is the other half of the near-singular torque spike.
    Jbar = Mi_Jt @ lambda_full
    nullspace_matrix = np.eye(J_full.shape[-1]) - Jbar @ J_full
    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix, dropped, coupling


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
        np.clip(delta_pos, -DELTA_POS_MAX, DELTA_POS_MAX),
        np.clip(delta_rotvec, -DELTA_ROT_MAX, DELTA_ROT_MAX),
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
        uncouple_pos_ori: bool = DEFAULT_UNCOUPLE_POS_ORI,
        nullspace_kp: float = DEFAULT_NULLSPACE_KP,
        lambda_rcond: float = LAMBDA_RCOND,
        cross_coupling_compensation: bool = CROSS_COUPLING_COMPENSATION,
    ) -> None:
        self.num_joints = int(num_joints)
        self.uncoupling = bool(uncouple_pos_ori)
        self.nullspace_kp = float(nullspace_kp)
        # Per-instance for the same reason uncoupling is: the parity harness must be
        # able to pin it at 0.0, or it compares a conditioned law against robosuite's
        # unconditioned one and reports the deviation as a controller regression.
        self.lambda_rcond = float(lambda_rcond)
        # Pinned off in the parity harness for the same reason as lambda_rcond:
        # robosuite has no such term, so a nonzero default would read as a regression.
        self.cross_coupling_compensation = bool(cross_coupling_compensation)

        self.goal_pos = np.zeros(3)
        self.goal_ori = np.eye(3)
        self.kp = np.full(6, DEFAULT_KP)
        self.kd = 2.0 * np.sqrt(self.kp)
        # None, NOT zeros(7): zeros is a real posture -- the arm straight up -- so a
        # missing reference would silently become a nullspace goal pulling every joint
        # there. robosuite takes initial_joint from the reset qpos and never has an
        # unset case; the equivalent fail-open here is no nullspace term at all.
        self.initial_joint: np.ndarray | None = None
        # Last tick's nullspace contribution, split out for the friction assist.
        self._no_nullspace = np.zeros(self.num_joints)
        self.nullspace_torque = self._no_nullspace
        # Law ticks on which sim's ctrlrange clip bound. The whole rotation-overshoot
        # question is whether sim's saturation is reproduced here, and that is not
        # observable downstream: the clip is applied to tau_sim, and what leaves the
        # law is M_real @ qddot, which carries no mark of having been clipped.
        self.sim_clip_ticks = 0
        # Law ticks on which lambda_full's conditioning dropped a direction. Counted
        # for the same reason as sim_clip_ticks: the truncation is invisible downstream
        # -- what leaves the law is a torque, which carries no mark of it -- and a
        # conditioning term nobody can see is the failure mode that got the previous
        # six envelopes deleted. Nonzero here means the arm is near a singularity.
        self.lambda_trunc_ticks = 0

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
        kp_raw = np.asarray(kp, dtype=np.float64)
        self.kp = np.clip(kp_raw, *KP_LIMITS)
        # osc.py derives kd FROM the clipped kp (kd = 2*sqrt(kp)*damping_ratio), so a
        # clip that bites must carry kd with it or the damping ratio silently changes.
        # We are handed kd rather than the ratio, so rescale by sqrt(kp_clipped/kp_raw),
        # which holds the ratio exactly and is a no-op wherever the clip did not bite.
        # resolve_gains already clips before deriving kd; this covers every other caller.
        self.kd = np.asarray(kd, dtype=np.float64) * np.sqrt(
            np.divide(self.kp, kp_raw, out=np.ones_like(kp_raw), where=kp_raw > 0.0)
        )
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
        mass_matrix_sim: np.ndarray | None = None,
        bias_sim: np.ndarray | None = None,
    ) -> np.ndarray:
        """osc.py's run_controller, optionally evaluated on SIM's plant model.

        With ``mass_matrix_sim=None`` this is the law as ported: robosuite's PD,
        robosuite's lambda built from the arm's own M, ``+coriolis`` instead of
        ``+qfrc_bias``. Every existing parity test exercises that path.

        With ``mass_matrix_sim`` supplied it emulates robosuite end to end. The
        reason it must is that osc.py's ``uncouple_pos_ori`` -- true in
        ``osc_pose.json``, in ``data.hdf5`` and in every trained policy -- DISCARDS
        the translation/rotation coupling block, and how much that discards is a
        function of M. At the sysid anchor ``lambda_uncoupled/lambda_full`` on +x is
        0.497 for sim's armature-inflated plant and 0.145 for the FR3. So running
        osc.py's law against the FR3's own M is not osc.py; it is a different
        controller that happens to share the source. Neither uncouple setting can
        fix that, which is why sweeping the tuning block never converged.

        The emulation forms sim's actuator command exactly -- including the
        ``qfrc_bias`` that ``Controller.run_controller`` adds and the ctrlrange clip
        that ``SingleArm.control`` applies on top of it -- turns it into the joint
        acceleration sim would have produced, and realises THAT on the real arm:

            qddot = M_sim^-1 (clip(tau_law + bias_sim) - bias_sim)
            tau   = M_real qddot + coriolis_real

        Joint acceleration, not task acceleration, so sim's nullspace motion is
        reproduced too. Identical to the ported law when the two M agree, which is
        what makes it a strict generalisation rather than a replacement.
        """
        position_error = self.goal_pos - ee_pos
        vel_pos_error = -ee_pos_vel
        desired_force = position_error * self.kp[0:3] + vel_pos_error * self.kd[0:3]

        ori_error = orientation_error(self.goal_ori, ee_ori_mat)
        vel_ori_error = -ee_ori_vel
        desired_torque = ori_error * self.kp[3:6] + vel_ori_error * self.kd[3:6]

        emulate = mass_matrix_sim is not None
        model = np.asarray(mass_matrix_sim) if emulate else mass_matrix

        J_pos, J_ori = J_full[:3, :], J_full[3:, :]
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix, dropped, coupling = (
            opspace_matrices(model, J_full, J_pos, J_ori, lambda_rcond=self.lambda_rcond)
        )
        if dropped:
            self.lambda_trunc_ticks += 1

        if self.uncoupling:
            task_force = desired_force
            if self.cross_coupling_compensation:
                # Cancel the rotation -> translation leak that uncoupling leaves behind,
                # WITHOUT inverting the 6x6. Writing J M^-1 J^T as [[A, B], [B^T, C]]
                # with A = lambda_pos^-1 and C = lambda_ori^-1, the realised task
                # acceleration is (J M^-1 J^T) @ wrench, so:
                #
                #   uncoupled  W = [Lp f ; Lo t]        -> accel_pos = f + B Lo t
                #   here       W = [Lp (f - B Lo t) ; Lo t] -> accel_pos = f   exactly
                #
                # since A Lp = I. That is the SAME position behaviour uncouple_pos_ori
                # =false buys, reached through the two 3x3 inverses (cond 3-38) instead
                # of the 6x6 (cond up to ~950), and B itself is inverted nowhere.
                #
                # It is deliberately ONE-SIDED. Cancelling both directions at once IS
                # the 6x6 inverse -- the symmetric form needs the Schur complement
                # A - B C^-1 B^T, which is precisely the object that goes singular. So
                # translation still disturbs orientation here; goal_ori is latched
                # across steps and holds it with real stiffness, while goal_pos is
                # re-anchored on the measured pose every step and therefore cannot.
                #
                # The price is rotation speed: the compensating force has a moment
                # about the EE, so realised angular acceleration becomes
                # (I - B^T Lp B Lo) t -- a median 0.88 of commanded over the envelope.
                # That is honest physics, not loss: it is largest exactly where holding
                # the EE point through a rotation is genuinely hard.
                task_force = desired_force - coupling @ (lambda_ori @ desired_torque)
            decoupled_wrench = np.concatenate(
                [lambda_pos @ task_force, lambda_ori @ desired_torque]
            )
        else:
            decoupled_wrench = lambda_full @ np.concatenate([desired_force, desired_torque])

        task_torque = J_full.T @ decoupled_wrench
        null_torque = (
            nullspace_torques(
                model, nullspace_matrix, self.initial_joint, q, dq,
                joint_kp=self.nullspace_kp,
            )
            if use_nullspace and self.initial_joint is not None
            else self._no_nullspace
        )

        if not emulate:
            # +coriolis, not +qfrc_bias: libfranka already compensates gravity.
            # Published, not just summed: the friction assist must be able to
            # subtract it again (pylibfranka_control._compute_tau).
            self.nullspace_torque = null_torque
            return task_torque + coriolis + null_torque

        # robosuite clips at the ACTUATOR, i.e. after run_controller has already
        # added torque_compensation. Outside saturation the bias cancels exactly, so
        # this only bites on sim's +/-12 Nm wrist limit -- which is exactly where the
        # large-rotation reference trajectories live.
        b = np.zeros(self.num_joints) if bias_sim is None else np.asarray(bias_sim)
        tau_sim = np.clip(task_torque + null_torque + b, -SIM_TORQUE_LIMITS, SIM_TORQUE_LIMITS)
        # Read the clip off its OUTPUT rather than naming its input: |tau_sim| lands
        # exactly on the limit iff the clip bound. Keeping the line above byte-identical
        # matters -- test_torque_limits_are_the_only_thing_that_rescales_tau whitelists
        # it by exact text, and that exactness is the point of the test.
        if np.any(np.abs(tau_sim) >= SIM_TORQUE_LIMITS):
            self.sim_clip_ticks += 1

        # mujoco's dof_damping is PASSIVE -- it acts on the plant, not through ctrl,
        # so it sits outside the clip. Small (0.1 N m s/rad against tens of Nm) but
        # exact and free. Sim's dof_frictionloss is deliberately NOT modelled here:
        # it is a constraint, not a sign function, and -0.1*sign(dq) would chatter at
        # rest. It is accounted for on the other side instead, by aiming the real
        # friction feedforward at coulomb_nm - frictionloss rather than at zero.
        qddot = np.linalg.solve(model, tau_sim - b - SIM_JOINT_DAMPING * dq)
        # The nullspace share is published unclipped: the friction assist consumes it
        # as an estimate of what is bias rather than command, and splitting a
        # saturated total is not defined.
        self.nullspace_torque = mass_matrix @ np.linalg.solve(model, null_torque)
        return mass_matrix @ qddot + coriolis


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
