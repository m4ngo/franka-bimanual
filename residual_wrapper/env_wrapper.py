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

def current_ee_pose(obs: dict) -> np.ndarray:
    """Return [x, y, z, qx, qy, qz, qw, gripper] for the right arm via FK."""
    q = np.array([obs[f"r_joint_{i}"] for i in range(1, 8)])
    pos, quat_xyzw = franka_fk(q)
    return np.concatenate([pos, quat_xyzw, [obs["r_gripper"]]]).astype(np.float32)


def split_gripper(obs: np.ndarray) -> np.ndarray:
    grip: float = obs[7]
    return np.concatenate([obs, np.array([-grip], dtype=np.float32)])


def extract_point_cloud(obs: dict) -> np.ndarray:
    """Reconstruct (2048, 3) point cloud from flat depth_* scalars in obs."""
    flat = np.array([obs[f"depth_{i}"] for i in range(_DEPTH_FLAT_SIZE)], dtype=np.float32)
    return flat.reshape(_DEPTH_POINT_COUNT, 3)


def strip_depth(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("depth_")}


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

def process_chunk(chunk: np.ndarray, current_ee: np.ndarray) -> np.ndarray:
    """Convert the first _RESIDUAL_HORIZON steps of a base-policy chunk to normalised deltas.

    Args:
        chunk: (T, 10) array — [r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw, r_gripper, kp, kd].
              T must be >= _RESIDUAL_HORIZON.
        current_ee: (8,) — [x, y, z, qx, qy, qz, qw, gripper] current robot state.

    Returns:
        (_RESIDUAL_HORIZON, 9) — [dx, dy, dz, rx, ry, rz, gripper, kp, kd] normalised.
    """
    result = np.zeros((_RESIDUAL_HORIZON, 9), dtype=np.float32)
    prev_pos = current_ee[:3]
    prev_quat = current_ee[3:7]

    for i in range(_RESIDUAL_HORIZON):
        step = chunk[i]
        next_pos = step[:3]
        next_quat = step[3:7]
        gripper = (step[7] - 0.5) * 2.0

        delta_pos = (next_pos - prev_pos) / _POS_SCALE
        delta_rot = (
            Rotation.from_quat(next_quat) * Rotation.from_quat(prev_quat).inv()
        ).as_rotvec() / _ROT_SCALE

        result[i] = np.array([*delta_pos, *delta_rot, gripper, step[8], step[9]], dtype=np.float32)

        prev_pos = next_pos
        prev_quat = next_quat

    return result


def build_action(chunk_step: np.ndarray, kp: float, kd: float) -> dict:
    """Build a RobotAction dict from a base-policy chunk row, overriding gains."""
    action = {k: float(v) for k, v in zip(_EE_ACTION_KEYS, chunk_step[:8])}
    action["kp"] = kp
    action["kd"] = kd
    return action


# ---------------------------------------------------------------------------
# Robot connection
# ---------------------------------------------------------------------------

def start_controller() -> SingleArmFranka:
    config = SingleArmFrankaConfig(
        r_server_ip="192.168.3.10",
        r_robot_ip="192.168.201.10",
        r_gripper_ip="192.168.201.10",
        r_port=18812,
        use_ee_pos=True,
    )
    robot = SingleArmFranka(config)
    robot.connect()
    return robot
