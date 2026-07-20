"""Environment-level observation and action utilities.

Constants and helpers that sit at the boundary between raw robot observations
and the policy / recording layer.  No policy or dataset imports here.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka import SingleArmFranka, SingleArmFrankaConfig
from lerobot_robot_bimanual_franka.franka_fk import franka_fk

_POS_SCALE = 0.05       # metres per normalised unit
_ROT_SCALE = 0.5        # radians per normalised unit
_CHUNK_EXEC = 5         # steps to execute per inference call (both base and residual)
_RESIDUAL_HORIZON = 10  # base-chunk steps forwarded to the residual policy as context
_GAINS_MAG = 0.5        # gains magnitude for clipping
_RESIDUAL_MAG = 1.0     # residual magnitude for clipping
_RESIDUAL_TRANS_MAG = 0.2
_RESIDUAL_ROT_MAG = 0.2

_EE_ACTION_KEYS = ("r_x", "r_y", "r_z", "r_qx", "r_qy", "r_qz", "r_qw", "r_gripper")
_ACTION_KEYS = (*_EE_ACTION_KEYS, "kp", "kd")

# Scalar obs keys that make up observation.state, in dataset recording order.
_STATE_OBS_KEYS = (
    "r_joint_1", "r_joint_2", "r_joint_3", "r_joint_4",
    "r_joint_5", "r_joint_6", "r_joint_7", "r_gripper",
)

_DEPTH_POINT_COUNT = 2048
_DEPTH_FLAT_SIZE = _DEPTH_POINT_COUNT * 3


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

# --- Sim-convention correction (sim-trained policies) -----------------------
# franka_fk returns the Franka TCP position but the FLANGE orientation: its DH
# tail carries the hand's 0.1034 m translation but not the hand's 45° mounting
# rotation. Sim-trained students expect robosuite's obs convention instead:
# grip-SITE position + hand-BODY orientation. Without the correction the
# student sees quaternions rotated 45.03° from its training distribution and
# positions offset 6.9 mm from the (world-calibrated) point cloud. Constants
# measured at matched joint configs across postures, std 0.0 — see
# SYSID_UPDATE.md in multi-fast (2026-07-18); pinned by multi-fast
# scripts/sysid/test_controller_parity.py.
_SIM_CONV_ROT = Rotation.from_rotvec([0.0, 0.0, -0.785891])  # fk(flange) -> hand-body
_SIM_CONV_POS_TOOL = np.array([0.0, 0.0, -0.0069])           # fk(TCP) -> grip site, tool frame (m)


def current_ee_pose(obs: dict, sim_convention: bool = True) -> np.ndarray:
    """Return [x, y, z, qx, qy, qz, qw, gripper] for the right arm via FK.

    sim_convention (default True): express the pose in the sim-training obs
    convention (grip-site position, hand-body orientation) so sim-trained
    policies see in-distribution proprio. False returns the raw franka_fk
    convention (TCP position, flange orientation) for legacy comparison runs.
    """
    q = np.array([obs[f"r_joint_{i}"] for i in range(1, 8)])
    pos, quat_xyzw = franka_fk(q)
    if sim_convention:
        r_fk = Rotation.from_quat(quat_xyzw)
        pos = pos + r_fk.apply(_SIM_CONV_POS_TOOL)
        quat_xyzw = (r_fk * _SIM_CONV_ROT).as_quat()
    return np.concatenate([pos, quat_xyzw, [obs["r_gripper"]]]).astype(np.float32)


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
_PANDA_FINGER_MAX_M = 0.04


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
    per-tick reads) for kinematics-only consumers like sysid collection."""
    config = SingleArmFrankaConfig(
        r_server_ip="192.168.3.10",
        r_robot_ip="192.168.201.10",
        r_gripper_ip="192.168.201.10",
        r_port=18812,
        control_mode="EE_DELTA",
        **({} if with_cameras else {"cameras": {}}),
    )
    robot = SingleArmFranka(config)
    robot.connect()
    return robot
