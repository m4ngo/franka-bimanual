"""Pure (numpy + scipy) helpers shared by the hardware and the sim rollouts.

Split out of rollout_common so `libero_bridge/rollout.py` can use them: that
loop runs in multi-fast/.venv, which has no lerobot and no franka_config, and a
second copy of an upstream port is how an executor drifts from the one it was
diffed against.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def slowdown_mode(prev_acts, cur_act, future_acts, window: int) -> bool:
    """Port of SAIL/utils/dev_utils.py:get_slowdown_mode_from_model.

    The precision label is the LAST action column. A window of `window` centred
    on the current step; any label over 0.5 means execute this step slowly.
    """
    cur_act = np.asarray(cur_act)
    if window == 1:
        return bool(cur_act[-1] > 0.5)
    future_acts = np.asarray(future_acts)
    segment = window // 2
    left_n = min(len(prev_acts), segment)
    right_n = min(future_acts.shape[0], segment)
    left = np.asarray(prev_acts[-left_n:]) if left_n else np.empty((0,))
    left = left[..., -1] if left.shape[0] > 0 else np.zeros(1)
    return bool(np.any(np.concatenate(
        [left, cur_act[np.newaxis, -1], future_acts[:right_n, -1]]
    ) > 0.5))


def tracking_error_low(meas_pos, meas_quat_xyzw, desired_pos, desired_rotvec,
                       pos_teb: float, ori_teb: float) -> bool:
    """Port of run_trained_agent_receding_horizon.py:check_if_tracking_error_low.

    Upstream reads `controller.ee_pos` / `.ee_ori_mat` off robosuite's OSC
    object; the measured pose stands in for both. The error definitions are
    upstream's: inf-norm on position, arccos((trace-1)/2) on orientation.
    """
    pos_err = float(np.linalg.norm(
        np.asarray(meas_pos, dtype=np.float64) - np.asarray(desired_pos, dtype=np.float64),
        ord=np.inf,
    ))
    real_R = Rotation.from_quat(np.asarray(meas_quat_xyzw, dtype=np.float64)).as_matrix()
    desired_R = Rotation.from_rotvec(np.asarray(desired_rotvec, dtype=np.float64)).as_matrix()
    trace = float(np.clip(np.trace(desired_R.T @ real_R), -1.0, 3.0))
    ori_err = float(np.arccos((trace - 1.0) / 2.0))
    return pos_err < pos_teb and ori_err < ori_teb


def rot6d_to_quat_xyzw(rot6d) -> np.ndarray:
    """Port of policy_local_utils.py's rotation_6d_to_matrix + rot6d_to_quat_xyzw.

    Gram-Schmidt on the two 3-vectors, then a sign convention on w so successive
    samples of a spline do not flip hemisphere between ticks.
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    b1 = r[:3] / max(float(np.linalg.norm(r[:3])), 1e-12)
    b2 = r[3:] - float(np.dot(b1, r[3:])) * b1
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-12)
    mat = np.stack((b1, b2, np.cross(b1, b2)), axis=-2)
    quat = Rotation.from_matrix(mat).as_quat()
    if quat[3] < 0.0:
        np.negative(quat, out=quat)
    return quat


def propagate_pose(pos, quat_xyzw, deltas) -> tuple[np.ndarray, np.ndarray]:
    """Compose (dpos, drotvec) rows onto a pose, in EE_DELTA's own convention.

    `goal = measured + dpos` and `goal_rot = drot * measured_rot`, matching
    OSCGoalBuilder.from_delta -- so propagating an anchor through the deltas that
    were actually commanded gives the pose the arm WOULD be at under perfect
    tracking. That is what makes a cumulative tracking error well defined on the
    delta path, where each individual delta is relative to its own step's
    measured pose and so carries no tracking information at all.
    """
    p = np.asarray(pos, dtype=np.float64).copy()
    r = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
    for row in np.asarray(deltas, dtype=np.float64):
        p = p + row[:3]
        r = Rotation.from_rotvec(row[3:6]) * r
    return p, r.as_quat()


def lead_goal(pos, quat_xyzw, next_pos, next_quat_xyzw, dt: float, lead) -> tuple[np.ndarray, np.ndarray]:
    """The goal moved `lead` seconds (per axis, (6,)) further along its own velocity.

    The velocity is the step from this goal to the one `dt` seconds later.
    """
    lead = np.asarray(lead, dtype=np.float64)
    r = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
    vel = (np.asarray(next_pos, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) / dt
    ang_vel = (Rotation.from_quat(np.asarray(next_quat_xyzw, dtype=np.float64)) * r.inv()).as_rotvec() / dt
    return np.asarray(pos, dtype=np.float64) + lead[:3] * vel, (Rotation.from_rotvec(lead[3:] * ang_vel) * r).as_quat()
