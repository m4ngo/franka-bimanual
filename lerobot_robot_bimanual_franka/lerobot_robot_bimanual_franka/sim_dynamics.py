"""robosuite's Panda mass matrix, M_sim(q), computed without mujoco.

The OSC law's wrench is `lambda * (kp*e - kd*edot)` with `lambda` built from the
MASS MATRIX. robosuite's Panda carries a fictitious armature (5/(i+1) N m s^2/rad)
and placeholder link inertias, so its M is 2-10x the FR3's and NOT block-diagonal in
the same way. That matters because `uncouple_pos_ori=True` -- which every sim
reference and every trained policy used -- DISCARDS the translation/rotation coupling
block, and how much that throws away is a function of M. Measured at the sim anchor,
`lambda_uncoupled/lambda_full` on +x is 0.497 in sim and 0.145 on the FR3. Running
robosuite's law against the FR3's own M therefore cannot reproduce robosuite, at any
gain. This module supplies the other M so it can.

Deployed to the NUC alongside the controller: numpy only, no package-relative
imports, no config reads. The constants below ARE robosuite's Panda model (v1.4.0
`models/assets/robots/panda/robot.xml` + `grippers/panda_gripper.xml`), transcribed;
`tests/test_sim_dynamics.py` regenerates them from the live robosuite install and
fails on any drift, so they are checked, not trusted.

Method is the kinetic-energy form, `M = sum_b (m_b Jv_b^T Jv_b + Jw_b^T I_b Jw_b)`,
not CRBA -- it is one einsum, it is obviously correct from `T = 0.5 qdot^T M qdot`,
and it matches robosuite's own `mass_matrix` to 1.2e-14 over 1000 random q spanning
the full joint range.
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 7

# (name, parent index in this list or -1, pos, quat_wxyz, mass, ipos, iquat_wxyz,
#  inertia_diag, joint slot: 0-6 arm revolute, 7/8 gripper slide, -1 fixed)
_BODIES = [
    ("link1", -1, [0.0, 0.0, 0.333], [1.0, 0.0, 0.0, 0.0],
     3.0, [0.0, 0.0, -0.07], [1.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.3], 0),
    ("link2", 0, [0.0, 0.0, 0.0], [0.7071067811865475, -0.7071067811865475, 0.0, 0.0],
     3.0, [0.0, -0.1, 0.0], [1.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.3], 1),
    ("link3", 1, [0.0, -0.316, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
     2.0, [0.04, 0.0, -0.05], [1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2], 2),
    ("link4", 2, [0.0825, 0.0, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
     2.0, [-0.04, 0.05, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2], 3),
    ("link5", 3, [-0.0825, 0.384, 0.0], [0.7071067811865475, -0.7071067811865475, 0.0, 0.0],
     2.0, [0.0, 0.0, -0.15], [1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2], 4),
    ("link6", 4, [0.0, 0.0, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
     1.5, [0.06, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1], 5),
    ("link7", 5, [0.088, 0.0, 0.0], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
     0.5, [0.0, 0.0, 0.08], [1.0, 0.0, 0.0, 0.0], [0.05, 0.05, 0.05], 6),
    ("right_hand", 6, [0.0, 0.0, 0.1065], [0.923785244892942, 0.0, 0.0, -0.38291098354328656],
     0.5, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.05, 0.05, 0.05], -1),
    ("right_gripper", 7, [0.0, 0.0, 0.0], [0.7071067811865475, 0.0, 0.0, -0.7071067811865475],
     0.3, [0.0, 0.0, 0.17], [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
     [0.09, 0.07, 0.05], -1),
    ("leftfinger", 8, [0.0, 0.0, 0.0524], [0.7071067811865475, 0.0, 0.0, 0.7071067811865475],
     0.1, [0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.005], 7),
    ("finger_joint1_tip", 9, [0.0, 0.0085, 0.056], [1.0, 0.0, 0.0, 0.0],
     0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.01, 0.01, 0.01], -1),
    ("rightfinger", 8, [0.0, 0.0, 0.0524], [0.7071067811865475, 0.0, 0.0, 0.7071067811865475],
     0.1, [0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.005], 8),
    ("finger_joint2_tip", 11, [0.0, -0.0085, 0.056], [1.0, 0.0, 0.0, 0.0],
     0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.01, 0.01, 0.01], -1),
]

# mujoco dof_armature: robot_model.py's `armature = 5.0/(i+1)`. Diagonal, so it never
# couples -- but it is 55-78% of each sim joint's inertia and has no FR3 counterpart.
ARMATURE = np.array([5.0 / (i + 1) for i in range(NUM_JOINTS)], dtype=np.float64)

# mujoco dof_frictionloss / dof_damping on the same DOFs. Sim's plant is NOT
# frictionless, so real friction must be compensated DOWN TO these, not to zero.
FRICTIONLOSS_NM = np.full(NUM_JOINTS, 0.1)
DAMPING_NMS_RAD = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01])

# The gripper slide DOFs are frozen at their reset qpos. Over the full finger travel
# M moves by 3.6e-5 relative, so this is below every other error in the stack.
_FINGER_QPOS = (0.020833, -0.020833)
_FINGER_AXIS = np.array([0.0, 1.0, 0.0])


def _cross(a, b):
    """``np.cross`` for (..., 3), without its dispatch overhead.

    Bit-identical -- it is the same three products in the same order -- but
    np.cross spends ~6.3 us on shape negotiation against ~3.0 us here, and this
    module calls it nine times per law tick inside a 1 ms budget.
    """
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def _quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _compile():
    """Fold every rigid sub-chain into the joint frame it hangs off.

    Bodies between two joints never move relative to each other, so the 13 mujoco
    bodies collapse to exactly NUM_JOINTS composite bodies -- one per joint frame,
    with the merged mass, COM and inertia. That makes the per-tick FK 7 steps
    instead of 13 and the einsum 7 bodies wide.
    """
    n = len(_BODIES)
    owner = [-1] * n          # index of the arm joint whose frame this body hangs off
    R_own = [None] * n        # body orientation in that joint frame
    t_own = [None] * n        # body origin in that joint frame
    # Fixed parent->child transform, with the frozen slide folded into the translation.
    fixed = []
    for _, par, pos, quat, _, _, _, _, slot in _BODIES:
        R = _quat2mat(quat)
        t = np.asarray(pos, dtype=np.float64)
        if slot >= NUM_JOINTS:
            t = t + R @ (_FINGER_AXIS * _FINGER_QPOS[slot - NUM_JOINTS])
        fixed.append((R, t))
    for i, (_, par, _, _, _, _, _, _, slot) in enumerate(_BODIES):
        R, t = fixed[i]
        if slot >= 0 and slot < NUM_JOINTS:
            # This body carries an arm joint: it IS the joint frame (pre-rotation;
            # the joint's own +z rotation commutes with the body's inertia only via
            # the FK, so the composite is expressed post-rotation -- see below).
            owner[i], R_own[i], t_own[i] = slot, np.eye(3), np.zeros(3)
        else:
            owner[i] = owner[par]
            R_own[i] = R_own[par] @ R
            t_own[i] = t_own[par] + R_own[par] @ t
    # Merge per owner.
    mass = np.zeros(NUM_JOINTS)
    com = np.zeros((NUM_JOINTS, 3))
    inertia = np.zeros((NUM_JOINTS, 3, 3))
    for i, (_, _, _, _, m, ipos, iquat, idiag, _) in enumerate(_BODIES):
        k = owner[i]
        mass[k] += m
        com[k] += m * (t_own[i] + R_own[i] @ np.asarray(ipos, dtype=np.float64))
    com /= mass[:, None]
    for i, (_, _, _, _, m, ipos, iquat, idiag, _) in enumerate(_BODIES):
        k = owner[i]
        Rb = R_own[i] @ _quat2mat(iquat)
        c = t_own[i] + R_own[i] @ np.asarray(ipos, dtype=np.float64)
        d = c - com[k]
        inertia[k] += Rb @ np.diag(idiag) @ Rb.T + m * (d @ d * np.eye(3) - np.outer(d, d))
    # Joint frame k's own fixed transform from joint frame k-1.
    Rf = np.stack([fixed[i][0] for i in range(n) if 0 <= _BODIES[i][8] < NUM_JOINTS])
    tf = np.stack([fixed[i][1] for i in range(n) if 0 <= _BODIES[i][8] < NUM_JOINTS])
    # anc[b, k] = 1 iff joint k moves composite body b. Serial chain, so lower-triangular.
    anc = np.tril(np.ones((NUM_JOINTS, NUM_JOINTS)))
    return Rf, tf, mass, com, inertia, anc


_RF, _TF, _MASS, _COM_L, _I_L, _ANC = _compile()


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """robosuite's 7x7 Panda mass matrix at ``q``, armature included.

    Frame-invariant, so the base frame convention does not matter; matches
    ``robosuite.controllers.Controller.mass_matrix`` to ~1e-14.
    """
    return _mass_from(_frames(q))


def _mass_from(fr):
    R, t, com, Iw = fr
    z = R[:, :, 2]                                    # joint k axis is its frame's +z
    d = com[:, None, :] - t[None, :, :]               # (B,7,3)
    # .transpose is a view; the einsum that used to do this transpose cost 2.9 us.
    Jv = _cross(z[None, :, :], d).transpose(0, 2, 1) * _ANC[:, None, :]
    Jw = z.T[None, :, :] * _ANC[:, None, :]
    M = np.einsum("b,bjk,bjl->kl", _MASS, Jv, Jv)
    M += np.einsum("bjk,bji,bil->kl", Jw, Iw, Jw)
    M[np.diag_indices(NUM_JOINTS)] += ARMATURE
    return M


def _frames(q):
    """Per-joint-frame world rotation and origin, plus composite COMs/inertias."""
    q = np.asarray(q, dtype=np.float64)
    c, s = np.cos(q), np.sin(q)
    R = np.empty((NUM_JOINTS, 3, 3))
    t = np.empty((NUM_JOINTS, 3))
    Rp, tp = np.eye(3), np.zeros(3)
    for k in range(NUM_JOINTS):
        Rk = Rp @ _RF[k]
        t[k] = tp + Rp @ _TF[k]
        R[k] = Rk @ np.array([[c[k], -s[k], 0.0], [s[k], c[k], 0.0], [0.0, 0.0, 1.0]])
        Rp, tp = R[k], t[k]
    com = t + np.einsum("bij,bj->bi", R, _COM_L)
    Iw = np.einsum("bij,bjk,blk->bil", R, _I_L, R)
    return R, t, com, Iw


def mass_and_bias(q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both, sharing one forward-kinematics pass.

    The 500 Hz law needs M_sim and qfrc_bias at the same q every tick, and the FK
    is the expensive half of each. Calling the two public entry points separately
    does it twice; this does not.
    """
    fr = _frames(q)
    return _mass_from(fr), _bias_from(fr, dq)


def bias(q: np.ndarray, dq: np.ndarray, gravity: float = -9.81) -> np.ndarray:
    """mujoco's ``qfrc_bias`` for the Panda: Coriolis + centrifugal + gravity.

    Needed because robosuite clips at the ACTUATOR, i.e. after ``run_controller``
    has already added ``torque_compensation = qfrc_bias``. Outside saturation the
    term cancels exactly (``clip(x + b) - b == x``), so this only changes anything
    where sim is on its ±12 Nm wrist limit -- which is precisely the large-rotation
    regime the reference trajectories live in.

    World-frame recursive Newton-Euler with ``qddot = 0``; the base is seeded with
    ``-g`` so the same pass yields gravity. Armature contributes nothing here (it
    is a fictitious rotor inertia with no gravitational or velocity-product term),
    matching mujoco, which also excludes it from ``qfrc_bias``.
    """
    return _bias_from(_frames(q), dq, gravity)


def _bias_from(fr, dq, gravity: float = -9.81):
    dq = np.asarray(dq, dtype=np.float64)
    R, o, com, Iw = fr
    z = R[:, :, 2]

    zdq = z * dq[:, None]
    omega = np.cumsum(zdq, axis=0)                                  # (7,3)
    w_prev = np.empty((NUM_JOINTS, 3))
    w_prev[0] = 0.0
    w_prev[1:] = omega[:-1]
    alpha = np.cumsum(_cross(w_prev, zdq), axis=0)

    # Both recursions are cumulative sums, so neither needs a Python loop -- which
    # matters: the looped version cost 312 us per call inside a 500 Hz law that has
    # ~1 ms of slack, and the deadline misses came back as libfranka
    # controller_torque_discontinuity reflexes.
    a_prev = np.zeros((NUM_JOINTS, 3))
    a_prev[1:] = alpha[:-1]
    d = np.empty((NUM_JOINTS, 3))                                   # o[k] - o[k-1]
    d[0] = 0.0                       # np.diff(..., prepend=o[:1]) seeded d[0] at zero
    d[1:] = o[1:] - o[:-1]
    a_o = np.cumsum(_cross(a_prev, d) + _cross(w_prev, _cross(w_prev, d)), axis=0)
    a_o[:, 2] -= gravity

    r = com - o
    a_com = a_o + _cross(alpha, r) + _cross(omega, _cross(omega, r))
    F = _MASS[:, None] * a_com
    N = np.einsum("bij,bj->bi", Iw, alpha) + _cross(
        omega, np.einsum("bij,bj->bi", Iw, omega))

    # Accumulate the subtree wrench about the WORLD origin, where the moment arm no
    # longer depends on k, then shift each joint's moment back onto its own axis:
    #   n_k = sum_{b>=k}[N_b + (com_b - o_k) x F_b] = n_k^O - o_k x f_k
    f = np.cumsum((F)[::-1], axis=0)[::-1]
    n_o = np.cumsum((N + _cross(com, F))[::-1], axis=0)[::-1]
    return np.einsum("kj,kj->k", n_o - _cross(o, f), z)
