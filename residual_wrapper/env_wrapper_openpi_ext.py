"""Extension to env_wrapper.py for OpenPI (pi05_libero) integration.

pi05_libero outputs actions directly in Cartesian EE-delta space: a chunk of
shape (action_horizon, 6) or (action_horizon, 7) = [dx, dy, dz, ax, ay, az,
(gripper)], where:
  - dx, dy, dz     : per-step EE translation delta (metres)
  - ax, ay, az     : per-step EE rotation delta as an axis-angle (rotation)
                     vector, i.e. direction = rotation axis, magnitude =
                     rotation angle in radians (scipy Rotation.from_rotvec
                     convention)
  - gripper (opt.) : if present, a normalized command in [-1, 1] (LIBERO
                     convention: -1 = open, 1 = closed; see LiberoOutputs
                     in openpi)

There is no joint-space integration and no forward kinematics involved --
pi05_libero is not being run here as a joint-space policy, so franka_fk is
not needed for this conversion.

Existing residual pipeline expects (T, 10) = [dx, dy, dz, dqx, dqy, dqz, dqw,
gripper, kp, kd] per step, i.e. the rotation delta encoded as a quaternion
(xyzw) rather than axis-angle.

ee_deltas_to_ee_chunk() bridges the two: it converts each step's axis-angle
rotation delta to a quaternion and repacks columns into the (T, 10) format
process_chunk/build_action already consume. Translation passes through
unchanged.

Frame convention: resolved, WORLD. robosuite's set_goal_orientation composes
goal = R_delta @ R_current, and _osc_goal_delta does the same left-multiply, so
the delta is applied in the frame it is expressed in. LIBERO's world frame and
config/world.yaml's are the same construction (base at x = -0.66, z = 0.912,
table top 0.90 vs 0.905), so only this arm's base yaw separates them -- hence
the world -> base rotation below.

Import alongside env_wrapper; nothing here shadows existing names.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import franka_config as fc


# LIBERO gripper convention (per openpi LiberoOutputs / make_libero_example):
# actions in [-1, 1], -1 = open, +1 = closed. Your _EE_ACTION_KEYS gripper
# convention (see build_action / _ee_delta callers) is normalised open<->closed
# on the same [-1, 1] scale already used elsewhere in this file (see
# process_chunk's `gripper = (step[7] - 0.5) * 2.0`, which maps a [0, 1] input
# to [-1, 1]). LIBERO's action is already [-1, 1], so no rescale is needed --
# only a sign check if your gripper's positive direction differs from LIBERO's.
# Flip here if your real-robot gripper action polarity is inverted vs LIBERO's.
_FLIP_GRIPPER_SIGN = False
# osc_pose.json output_max, i.e. metres/radians per normalised unit. Same numbers
# the residual pipeline already reads; never a second copy.
_LIBERO_POS_SCALE = fc.policy("residual.pos_scale_m")
_LIBERO_ROT_SCALE = fc.policy("residual.rot_scale_rad")


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite transform_utils.quat2axisangle, verbatim.

    Not interchangeable with scipy's Rotation.as_rotvec(): scipy canonicalises to
    |theta| <= pi, robosuite does not. LIBERO's proprio sits at theta ~ pi (gripper
    down, w ~ 0), so the two disagree by a sign flip on exactly the modal state.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(w)) / den


def _world_to_base_rotation(arm_name: str) -> Rotation:
    """Rotation that maps a WORLD-frame vector into `arm_name`'s base frame.

    franka_config.robot_base_in_world(arm_name) returns a Pose whose
    orientation is base-in-world (i.e. rotates base-frame vectors into
    world). We need the inverse to go world -> base.
    """
    pose = fc.robot_base_in_world(arm_name)
    w, x, y, z = pose.quat_wxyz
    R_base_in_world = Rotation.from_quat([x, y, z, w])  # scipy wants xyzw
    return R_base_in_world.inv()


def ee_deltas_to_ee_chunk(
    openpi_chunk: np.ndarray,
    current_gripper: float,
    kp: float,
    kd: float,
    arm_name: str,
) -> np.ndarray:
    """Convert an OpenPI Cartesian-delta action chunk to the (T, 10) EE-delta
    chunk format process_chunk/build_action expect.

    Args:
        openpi_chunk: (T, 6) or (T, 7) array from the OpenPI websocket
            client. Columns are [dx, dy, dz, ax, ay, az, (gripper)] -- the
            first 3 are a translation delta in metres, the next 3 are a
            rotation delta as an axis-angle (rotvec) in radians. If a 7th
            column is present it's an explicit gripper command; otherwise
            current_gripper is carried through unchanged for every step.
        current_gripper: normalized gripper value in [0, 1] to carry through
            when the policy does not emit an explicit gripper channel.
        kp, kd: gains to fill into the trailing two columns, forwarded
            unchanged (mirrors build_action's kp/kd override behaviour;
            OpenPI has no notion of gains, so these come from the caller,
            e.g. a fixed default or the residual's previous output).

    Returns:
        (T, 10) array -- [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, kp, kd],
        directly consumable by process_chunk() and build_action().
    Args:
        arm_name: physical arm identifier ("left"/"right") as returned by
            BimanualFrankaConfig.arm_name(arm_key) / accepted by
            franka_config.robot_base_in_world(). Required because pi05's
            world-frame EE deltas must be rotated into this arm's base
            frame before EE_DELTA control consumes them -- world and base
            are only coincident when robot_base_in_world's orientation is
            identity (true for left in the current calibration, NOT
            guaranteed and NOT true for right).
    """
    T = openpi_chunk.shape[0]
    has_gripper_channel = openpi_chunk.shape[1] >= 7
    gripper_state = float(np.clip(current_gripper, 0.0, 1.0))

    d_pos_world = openpi_chunk[:, :3].astype(np.float64) * _LIBERO_POS_SCALE
    rotvec_world = openpi_chunk[:, 3:6].astype(np.float64) * _LIBERO_ROT_SCALE

    R_w2b = _world_to_base_rotation(arm_name)
    d_pos_base = R_w2b.apply(d_pos_world).astype(np.float32)
    rotvec_base = R_w2b.apply(rotvec_world)  # angle is frame-invariant; axis rotates
    d_quat_base = Rotation.from_rotvec(rotvec_base).as_quat().astype(np.float32)

    out = np.zeros((T, 10), dtype=np.float32)
    out[:, :3] = d_pos_base
    out[:, 3:7] = d_quat_base

    if has_gripper_channel:
        gripper = openpi_chunk[:, 6].astype(np.float64)
        if _FLIP_GRIPPER_SIGN:
            gripper = -gripper
        out[:, 7] = np.clip(gripper, -1.0, 1.0).astype(np.float32)
    else:
        out[:, 7] = gripper_state

    out[:, 8] = kp
    out[:, 9] = kd

    return out