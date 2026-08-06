"""Environment-level observation and action utilities.

Constants and helpers that sit at the boundary between raw robot observations
and the policy / recording layer.  No policy or dataset imports here.
"""

import franka_config as fc
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka import SingleArmFranka, SingleArmFrankaConfig
from lerobot_robot_bimanual_franka.franka_fk import franka_fk
from lerobot_robot_bimanual_franka.franka_jacobian import zero_jacobian

_PROFILE = "single_arm_franka"
_ARM_KEY = fc.profile(_PROFILE).depth_center_arm

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
    """Reconstruct (2048, 3) point cloud from flat depth_* scalars in obs."""
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
# Robot connection
# ---------------------------------------------------------------------------

def start_controller(with_cameras: bool = True) -> SingleArmFranka:
    """with_cameras=False skips the camera rig entirely (no GigE connects, no
    per-tick reads) for kinematics-only consumers like sysid collection.

    All hardware addressing comes from the `single_arm_franka` rig profile.
    """
    config = SingleArmFrankaConfig(
        **({} if with_cameras else {"cameras": {}, "depth_cam": {}, "depth": False}),
    )
    robot = SingleArmFranka(config)
    robot.connect()
    return robot
