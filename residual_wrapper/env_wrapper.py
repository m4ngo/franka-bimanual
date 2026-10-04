"""Environment-level observation and action utilities.

Constants and helpers that sit at the boundary between raw robot observations
and the policy / recording layer.  No policy or dataset imports here.
"""

import franka_config as fc
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka import (
    SingleArmFranka, SingleArmFrankaConfig, SingleArmRight, SingleArmRightConfig,
)
from lerobot_robot_bimanual_franka.franka_fk import franka_fk
from lerobot_robot_bimanual_franka.franka_jacobian import zero_jacobian

_PROFILE = "single_arm_franka"
_ARM_KEY = fc.profile(_PROFILE).depth_center_arm

# Rig profile -> robot class. Which physical arm each drives is in config/rig.yaml.
_RIGS = {
    "single_arm_franka": (SingleArmFranka, SingleArmFrankaConfig),
    "single_arm_right": (SingleArmRight, SingleArmRightConfig),
}

_RES_POS_GAIN = fc.policy("residual.res_pos_gain")
_RES_ROT_GAIN = fc.policy("residual.res_rot_gain")
_POS_SCALE = fc.policy("residual.pos_scale_m")     # metres per normalised unit
_ROT_SCALE = fc.policy("residual.rot_scale_rad")   # radians per normalised unit
_CHUNK_EXEC = fc.policy("residual.chunk_exec")     # steps to execute per inference call
_RESIDUAL_HORIZON = fc.policy("residual.horizon")  # base-chunk steps forwarded to the residual policy
_GAINS_MAG = fc.policy("residual.gains_mag")       # gains magnitude for clipping
_RESIDUAL_MAG = fc.policy("residual.residual_mag")  # residual magnitude for clipping
_RESIDUAL_TRANS_MAG = fc.policy("residual.residual_trans_mag")
_RESIDUAL_ROT_MAG = fc.policy("residual.residual_rot_mag")

_NUM_JOINTS = fc.num_joints()
_EE_ACTION_KEYS = tuple(
    f"{_ARM_KEY}_{ax}" for ax in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
)
_ACTION_KEYS = (*_EE_ACTION_KEYS, "kp", "kd")

# Scalar obs keys that make up observation.state, in dataset recording order.
_STATE_OBS_KEYS = (
    *(f"{_ARM_KEY}_joint_{i}" for i in range(1, _NUM_JOINTS + 1)),
    f"{_ARM_KEY}_gripper",
)

_DEPTH_POINT_COUNT = fc.control("observation.depth_point_count")
_DEPTH_FLAT_SIZE = _DEPTH_POINT_COUNT * 3


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

# --- Sim-convention correction (sim-trained policies) -----------------------
# franka_fk returns the Franka TCP position but the FLANGE orientation: its DH
# tail carries the hand's 0.1034 m translation but not the hand's 45° mounting
# rotation. Sim-trained students expect robosuite's obs convention instead:
# grip-SITE position + hand-BODY orientation. Constants measured at matched
# joint configs across postures — see config/world.yaml sim_alignment.
_sim_rotvec, _SIM_CONV_POS_TOOL = fc.sim_ee_convention()
_SIM_CONV_ROT = Rotation.from_rotvec(_sim_rotvec)  # fk(flange) -> hand-body

# --- Sim-WORLD alignment (sim-trained policies) ------------------------------
# Maps real-world quantities into sim's world convention. The real world frame
# (config/world.yaml) is already floor-origin with the base at z = 0.912, i.e.
# the sim convention, so this is identity unless world.yaml says otherwise.
# Applied to proprio pose, twist, and cloud together so the modalities stay
# mutually consistent.
_SIM_WORLD_POSE = fc.sim_world_alignment()
_SIM_WORLD_ROT = Rotation.from_matrix(_SIM_WORLD_POSE.rotation)
_SIM_WORLD_T = _SIM_WORLD_POSE.translation


def to_sim_world_pose(ee_pose_world: np.ndarray) -> np.ndarray:
    """Map [x,y,z,qx,qy,qz,qw,...] from the real world frame to sim's world
    convention (constants above). Trailing entries pass through."""
    out = ee_pose_world.copy()
    out[:3] = (_SIM_WORLD_ROT.apply(ee_pose_world[:3].astype(np.float64))
               + _SIM_WORLD_T).astype(np.float32)
    q = _SIM_WORLD_ROT * Rotation.from_quat(ee_pose_world[3:7].astype(np.float64))
    out[3:7] = q.as_quat().astype(np.float32)
    return out


def to_sim_world_points(points: np.ndarray) -> np.ndarray:
    """Map (N, 3) real-world points into sim's world convention."""
    return (_SIM_WORLD_ROT.apply(points.astype(np.float64))
            + _SIM_WORLD_T).astype(np.float32)


def to_sim_world_twist(twist: np.ndarray) -> np.ndarray:
    """Rotate a [lin(3), ang(3)] twist into sim's world convention
    (velocities rotate with the frame; the translation doesn't apply)."""
    t = np.asarray(twist, dtype=np.float64)
    return np.concatenate([
        _SIM_WORLD_ROT.apply(t[:3]), _SIM_WORLD_ROT.apply(t[3:])
    ]).astype(np.float32)


def current_ee_pose(obs: dict, sim_convention: bool = True) -> np.ndarray:
    """Return [x, y, z, qx, qy, qz, qw, gripper] for the active arm via FK.

    sim_convention (default True): express the pose in the sim-training obs
    convention (grip-site position, hand-body orientation) so sim-trained
    policies see in-distribution proprio. False returns the raw franka_fk
    convention (TCP position, flange orientation) for legacy comparison runs.
    """
    q = np.array([obs[f"{_ARM_KEY}_joint_{i}"] for i in range(1, _NUM_JOINTS + 1)])
    pos, quat_xyzw = franka_fk(q)
    if sim_convention:
        r_fk = Rotation.from_quat(quat_xyzw)
        pos = pos + r_fk.apply(_SIM_CONV_POS_TOOL)
        quat_xyzw = (r_fk * _SIM_CONV_ROT).as_quat()
    return np.concatenate([pos, quat_xyzw, [obs[f"{_ARM_KEY}_gripper"]]]).astype(np.float32)


def ee_pose_to_world(
    ee_pose: np.ndarray,
    r_robot_in_world: np.ndarray,
    t_robot_in_world: np.ndarray,
) -> np.ndarray:
    """Map [x, y, z, qx, qy, qz, qw, gripper] from robot base frame to world frame.

    The depth-camera point cloud is produced in world frame, but franka_fk
    returns the EE pose in the robot base frame; use this before any
    subtraction/comparison between the two (e.g. center_on_eef proprio).
    """
    out = ee_pose.copy()
    out[:3] = (r_robot_in_world @ ee_pose[:3].astype(np.float64) + t_robot_in_world).astype(np.float32)
    q_world = Rotation.from_matrix(r_robot_in_world) * Rotation.from_quat(ee_pose[3:7])
    out[3:7] = q_world.as_quat().astype(np.float32)
    return out


# Panda finger-joint range (m); robosuite gripper_qpos = [width/2, -width/2].
_PANDA_FINGER_MAX_M = fc.policy("gripper.panda_finger_max_m")


def default_home_q(name: str | None = None) -> np.ndarray:
    """Home configuration (rad) for the active arm.

    home_poses/*.json is the only source of home configurations; `name`
    defaults to arms.home_poses.default in config/arms.yaml.
    """
    return fc.home_q(name, key=_ARM_KEY)


def measured_ee_twist_world(snap, r_robot_in_world: np.ndarray) -> np.ndarray:
    """Measured EE twist [lin(3), ang(3)] = J(q) @ dq, rotated base -> world.

    The firmware's EE-velocity fields are commanded (O_dP_EE_d) or broken
    (measured reads returned zeros on this build), so compute the twist from
    measured joint velocities; J is recomputed analytically, never trusted
    from the snapshot.
    """
    q, dq, _J, ee_pos, _quat, _twist = snap
    J = zero_jacobian(np.asarray(q, dtype=np.float64),
                      ee_pos_base=np.asarray(ee_pos, dtype=np.float64))
    tw = J @ np.asarray(dq, dtype=np.float64)
    R = np.asarray(r_robot_in_world, dtype=np.float64)
    return np.concatenate([R @ tw[:3], R @ tw[3:]]).astype(np.float32)


def split_gripper(obs: np.ndarray) -> np.ndarray:
    """Replace normalized gripper obs[7] with sim-convention finger qpos (g, -g) in meters."""
    g = obs[7] * _PANDA_FINGER_MAX_M
    out = obs.astype(np.float32).copy()
    out[7] = g
    return np.concatenate([out, np.array([-g], dtype=np.float32)])


def extract_point_cloud(obs: dict) -> np.ndarray:
    """Reconstruct (2048, 3) point cloud from flat depth_* scalars in obs.

    Legacy layout only. get_observation no longer emits depth_* scalars; live
    callers read the array off SingleArmFranka.last_full_point_cloud.
    """
    flat = np.array([obs[f"depth_{i}"] for i in range(_DEPTH_FLAT_SIZE)], dtype=np.float32)
    return flat.reshape(_DEPTH_POINT_COUNT, 3)


def strip_depth(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("depth_")}


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

def process_chunk(chunk: np.ndarray) -> np.ndarray:
    """Convert the first _RESIDUAL_HORIZON steps of a base-policy chunk for the residual model.

    The base policy outputs per-step EE deltas directly, so no consecutive-pose
    differencing is needed.  Each step's position delta is divided by _POS_SCALE and
    each rotation delta quaternion (xyzw) is converted to a rotvec and divided by
    _ROT_SCALE to produce the normalised representation the residual policy expects.

    Args:
        chunk: (T, 10) array — [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, kp, kd].
               T must be >= _RESIDUAL_HORIZON.  Position deltas in metres; rotation
               delta encoded as a unit quaternion (xyzw).

    Returns:
        (_RESIDUAL_HORIZON, 9) — [dx, dy, dz, rx, ry, rz, gripper, kp, kd] normalised.
    """
    if len(chunk) < _RESIDUAL_HORIZON:
        raise ValueError(
            f"base chunk has {len(chunk)} steps, residual context needs {_RESIDUAL_HORIZON}; "
            f"raise the base policy's n_action_steps"
        )
    result = np.zeros((_RESIDUAL_HORIZON, 9), dtype=np.float32)
    for i in range(_RESIDUAL_HORIZON):
        step = chunk[i]
        delta_pos = step[:3] / _POS_SCALE
        delta_rot = Rotation.from_quat(step[3:7]).as_rotvec() / _ROT_SCALE
        gripper = (step[7] - 0.5) * 2.0
        result[i] = np.array([*delta_pos, *delta_rot, gripper, step[8], step[9]], dtype=np.float32)
    return result


def build_action(chunk_step: np.ndarray, kp: float, kd: float) -> dict:
    """Build a RobotAction dict from a base-policy chunk row, overriding gains.

    BasePolicy.infer() applies the lerobot postprocessor, which denormalises
    position deltas back to metres (the units stored in the training dataset).
    We forward them as-is; _ee_delta expects metres directly.  Rotation and
    gripper are passed through unchanged.
    """
    action = {k: float(v) for k, v in zip(_EE_ACTION_KEYS, chunk_step[:8])}
    action["kp"] = kp
    action["kd"] = kd
    return action


# ---------------------------------------------------------------------------
# Chunk-start-relative targets: the reach executor's stages, for an EE_POS base
# ---------------------------------------------------------------------------
# A LeRobot base trained on an EE_POS recording emits ABSOLUTE base-frame poses.
# run_residual.py runs them the way reach_residual.py runs the analytic reach
# base (multi-fast's ActionChunkWrapper in goal_mode="target"): the chunk is
# expressed relative to the pose the base planned from, the residual is summed
# in that normalised space, the sum is turned back into absolute targets, and
# each target is executed as the one-step delta from the pose measured at that
# step. Position and rotation are both target-typed here (a pose recording
# carries both), unlike reach's position-only curve.

def chunk_to_relative(chunk: np.ndarray, anchor_pos: np.ndarray,
                      anchor_quat_xyzw: np.ndarray) -> np.ndarray:
    """(T, 10) absolute targets [x,y,z,qx,qy,qz,qw,grip,kp,kd] -> (T, 9)
    chunk-start-relative normalised [pos/_POS_SCALE, rotvec/_ROT_SCALE, grip, kp, kd].

    The residual's action_chunk, and the space base and residual are summed in;
    reach_residual.chunk_poses inverted. `anchor` is the measured pose at the
    observation the base inferred on.
    """
    anchor_rot = Rotation.from_quat(np.asarray(anchor_quat_xyzw, dtype=np.float64))
    chunk = np.asarray(chunk, dtype=np.float64)
    out = np.empty((len(chunk), 9), dtype=np.float32)
    out[:, 0:3] = (chunk[:, 0:3] - np.asarray(anchor_pos, dtype=np.float64)) / _POS_SCALE
    out[:, 3:6] = (Rotation.from_quat(chunk[:, 3:7]) * anchor_rot.inv()).as_rotvec() / _ROT_SCALE
    out[:, 6:9] = chunk[:, 7:10]
    return out


def relative_to_poses(rel: np.ndarray, anchor_pos: np.ndarray,
                      anchor_quat_xyzw: np.ndarray) -> np.ndarray:
    """(T, 9) chunk-start-relative normalised -> (T+1, 7) [xyz, xyzw] absolute
    base-frame poses, the anchor first (reach_residual.chunk_poses)."""
    anchor_pos = np.asarray(anchor_pos, dtype=np.float64)
    anchor_rot = Rotation.from_quat(np.asarray(anchor_quat_xyzw, dtype=np.float64))
    rel = np.asarray(rel, dtype=np.float64)
    pos = anchor_pos + rel[:, 0:3] * _POS_SCALE
    rot = Rotation.from_rotvec(rel[:, 3:6] * _ROT_SCALE) * anchor_rot
    return np.vstack([np.concatenate([anchor_pos, anchor_rot.as_quat()]),
                      np.hstack([pos, rot.as_quat()])])


def gripper_to_sim(g):
    """Real gripper [0, 1] (1 = open) -> robosuite's [-1, 1] (-1 = open)."""
    return 1.0 - 2.0 * np.asarray(g)


def gripper_to_real(g):
    """robosuite's [-1, 1] (-1 = open) -> real gripper [0, 1] (1 = open)."""
    return (1.0 - np.asarray(g)) / 2.0


def residual_input(base_rel: np.ndarray) -> np.ndarray:
    """The base chunk as the student saw it in sim: gripper column in robosuite units."""
    out = np.array(base_rel, dtype=np.float32, copy=True)
    out[:, 6] = gripper_to_sim(out[:, 6])
    return out


def compose_chunk(base_rel: np.ndarray, res_chunk: np.ndarray, bound: float) -> np.ndarray:
    """multi-fast's StudentPredictor composition: final = clip(base + residual,
    -bound, bound) per normalised channel, with the base contributing zero gains
    (_augment_base). `bound` is the teacher's composed_action_bound.

    `res_chunk` rows are the residual's [damping, stiffness, dpos(3), drot(3),
    dgrip] (multi-fast's gains-first layout) and cover the first len(res_chunk)
    steps; the rest of the chunk is the base alone. res_pos_gain / res_rot_gain
    scale the correction before the sum and are no-ops at 1.0. The gripper is
    summed in robosuite units and returned in the real [0, 1].
    """
    final = np.array(base_rel, dtype=np.float32, copy=True)
    k = min(len(res_chunk), len(base_rel))
    if k == 0:
        return final
    res = np.asarray(res_chunk[:k], dtype=np.float32)
    final[:k, 0:3] = np.clip(base_rel[:k, 0:3] + res[:, 2:5] * _RES_POS_GAIN, -bound, bound)
    final[:k, 3:6] = np.clip(base_rel[:k, 3:6] + res[:, 5:8] * _RES_ROT_GAIN, -bound, bound)
    final[:k, 6] = gripper_to_real(np.clip(gripper_to_sim(base_rel[:k, 6]) + res[:, 8], -bound, bound))
    final[:k, 7] = np.clip(res[:, 1], -bound, bound)   # kp
    final[:k, 8] = np.clip(res[:, 0], -bound, bound)   # kd
    return final


def target_to_delta(target_pose: np.ndarray, ee_pos: np.ndarray,
                    ee_quat_xyzw: np.ndarray) -> np.ndarray:
    """An absolute target pose -> the normalised [dpos/_POS_SCALE, drotvec/_ROT_SCALE]
    that lands on it from the pose measured now, clipped to one step
    (reach_residual.target_to_delta, ActionChunkWrapper's second stage). The
    clip is the delta envelope's own saturation, not a new limit layer: a target
    further than one step away is approached at full step."""
    goal_rot = Rotation.from_quat(np.asarray(target_pose[3:7], dtype=np.float64))
    ee_rot = Rotation.from_quat(np.asarray(ee_quat_xyzw, dtype=np.float64))
    return np.concatenate([
        np.clip((np.asarray(target_pose[:3], dtype=np.float64) - ee_pos) / _POS_SCALE, -1.0, 1.0),
        np.clip((goal_rot * ee_rot.inv()).as_rotvec() / _ROT_SCALE, -1.0, 1.0),
    ])


def delta_action(delta_norm: np.ndarray, gripper: float, kp: float, kd: float) -> dict:
    """Normalised [dpos(3), drot(3)] -> the EE_DELTA RobotAction send_action
    takes: metres and a delta quaternion (RealReach._action_to_delta)."""
    dpos = np.asarray(delta_norm[:3], dtype=np.float64) * _POS_SCALE
    dquat = Rotation.from_rotvec(np.asarray(delta_norm[3:6], dtype=np.float64) * _ROT_SCALE).as_quat()
    action = {k: float(v) for k, v in zip(_EE_ACTION_KEYS, (*dpos, *dquat, gripper))}
    action["kp"] = float(kp)
    action["kd"] = float(kd)
    return action


# ---------------------------------------------------------------------------
# Robot connection
# ---------------------------------------------------------------------------

def start_controller(with_cameras: bool = True, rig: str = _PROFILE) -> SingleArmFranka:
    """with_cameras=False skips the camera rig entirely (no GigE connects, no
    per-tick reads) for kinematics-only consumers like sysid collection.

    All hardware addressing comes from the `rig` profile in config/rig.yaml.
    """
    if fc.profile(rig).depth_center_arm != _ARM_KEY:
        raise ValueError(f"rig {rig!r} does not expose the {_ARM_KEY!r} key prefix")
    robot_cls, config_cls = _RIGS[rig]
    config = config_cls(
        **({} if with_cameras else {"cameras": {}, "depth_cam": {}, "depth": False}),
    )
    robot = robot_cls(config)
    robot.connect()
    return robot
