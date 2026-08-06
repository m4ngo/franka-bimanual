"""Extension to env_wrapper.py for OpenPI (pi05_libero) integration.

pi05_libero is trained in JOINT space: the checkpoint used here returns a
chunk of shape (action_horizon, 7) = [dq1..dq7], where dq_i are per-step
joint-angle deltas (LIBERO's raw actions are already deltas -- pi05_libero's
train config uses extra_delta_transform=False, so no additional delta
conversion is applied). Some older checkpoints may also append a gripper
channel; if present, it is treated as a normalized command in [-1, 1]
(LIBERO convention: -1 = open, 1 = closed; see LiberoOutputs in openpi).

Your existing residual pipeline (process_chunk / build_action / _ee_delta)
is EE-pose-delta based: it expects (T, 10) = [dx, dy, dz, dqx, dqy, dqz, dqw,
gripper, kp, kd] per step, i.e. a *Cartesian* EE pose delta encoded as a
translation (metres) and a delta quaternion (xyzw), matching franka_fk's
output convention.

joint_deltas_to_ee_chunk() bridges the two: it forward-integrates the
predicted joint-delta trajectory from the currently observed joint config,
runs franka_fk at each intermediate joint config, and takes consecutive-step
EE pose deltas. This produces the same (T, 10) EE-delta format your
BasePolicy.infer() already returns, so process_chunk/build_action are
unchanged downstream.

Import alongside env_wrapper; nothing here shadows existing names.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka.franka_fk import franka_fk

# LIBERO gripper convention (per openpi LiberoOutputs / make_libero_example):
# actions in [-1, 1], -1 = open, +1 = closed. Your _EE_ACTION_KEYS gripper
# convention (see build_action / _ee_delta callers) is normalised open<->closed
# on the same [-1, 1] scale already used elsewhere in this file (see
# process_chunk's `gripper = (step[7] - 0.5) * 2.0`, which maps a [0, 1] input
# to [-1, 1]). LIBERO's action is already [-1, 1], so no rescale is needed --
# only a sign check if your gripper's positive direction differs from LIBERO's.
# Flip here if your real-robot gripper action polarity is inverted vs LIBERO's.
_FLIP_GRIPPER_SIGN = False


def joint_deltas_to_ee_chunk(
    openpi_chunk: np.ndarray,
    current_q: np.ndarray,
    current_gripper: float,
    kp: float,
    kd: float,
) -> np.ndarray:
    """Convert an OpenPI joint-delta action chunk to the (T, 10) EE-delta
    chunk format process_chunk/build_action expect.

    Args:
        openpi_chunk: (T, 7) or (T, 8) array from the OpenPI websocket
            client. The Franka real checkpoint used here emits 7 arm-action
            columns (joint-angle deltas in rad); older checkpoints may also
            include an explicit gripper channel in the last column.
        current_q: (7,) current right-arm joint angles (rad), i.e. the seed
            configuration the chunk's deltas are integrated from. Pass the
            joint angles read at the same control step the chunk was
            inferred from (obs["r_joint_1..7"] at inference time), not a
            stale value -- forward integration drifts if the seed is stale.
        current_gripper: normalized gripper value in [0, 1] to carry through
            when the policy does not emit an explicit gripper channel.
        kp, kd: gains to fill into the trailing two columns, forwarded
            unchanged (mirrors build_action's kp/kd override behaviour;
            OpenPI has no notion of gains, so these come from the caller,
            e.g. a fixed default or the residual's previous output).

    Returns:
        (T, 10) array -- [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, kp, kd],
        directly consumable by process_chunk() and build_action().
    """
    T = openpi_chunk.shape[0]
    q = np.asarray(current_q, dtype=np.float64).copy()
    gripper_state = float(np.clip(current_gripper, 0.0, 1.0))

    pos_prev, quat_prev = franka_fk(q)
    rot_prev = Rotation.from_quat(quat_prev)

    out = np.zeros((T, 10), dtype=np.float32)
    for i in range(T):
        dq = openpi_chunk[i, :7].astype(np.float64)
        q = q + dq
        pos_next, quat_next = franka_fk(q)
        rot_next = Rotation.from_quat(quat_next)

        d_pos = (pos_next - pos_prev).astype(np.float32)
        # Delta quaternion s.t. rot_next = d_rot * rot_prev (world-frame delta,
        # matching the convention process_chunk / _propagate_pose_traj use
        # elsewhere in this pipeline for composing EE deltas).
        d_rot = (rot_next * rot_prev.inv()).as_quat().astype(np.float32)

        if openpi_chunk.shape[1] >= 8:
            gripper = float(openpi_chunk[i, 7])
            if _FLIP_GRIPPER_SIGN:
                gripper = -gripper
            gripper_state = float(np.clip(gripper, 0.0, 1.0))

        out[i, :3] = d_pos
        out[i, 3:7] = d_rot
        out[i, 7] = gripper_state
        out[i, 8] = kp
        out[i, 9] = kd

        pos_prev, rot_prev = pos_next, rot_next

    return out
