"""
Control loop for a base policy + residual policy on a single-arm Franka.

Action spaces
-------------
Base policy output  : [r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw, r_gripper, kp, kd]
                      Absolute EE pose (m, xyzw quat) in robot frame, plus gains.

Residual input chunk: (5, 9) per-step deltas derived from the base chunk.
                      [dx, dy, dz, rx, ry, rz, gripper, kp, kd]
                      Position deltas normalised to [-1, 1] where ±1 = ±0.05 m.
                      Rotation deltas (axis-angle) normalised where ±1 = ±0.5 rad.
                      First step delta is relative to the current EE pose; subsequent
                      steps are relative to the previous chunk action.

Residual output     : [dx, dy, dz, rx, ry, rz, kp, kd] (normalised, same scales).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import env_wrapper
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot_robot_bimanual_franka.franka_fk import franka_fk

logger = logging.getLogger(__name__)

_POSES_DIR = Path(__file__).resolve().parent.parent / "home_poses"
_DEFAULT_HOME_Q = [
    -0.28223089288736675, -0.5594522989991991, -0.4191884798561259,
    -1.82212661700904, 0.06416041394704838, 1.5246974433097138, -0.7569427650529224,
]

_POS_SCALE = 0.05       # metres per normalised unit
_ROT_SCALE = 0.5        # radians per normalised unit
_CHUNK_EXEC = 5         # base-policy actions to execute per inference call
_EE_ACTION_KEYS = ("r_x", "r_y", "r_z", "r_qx", "r_qy", "r_qz", "r_qw", "r_gripper")

# Scalar obs keys that make up observation.state, in dataset recording order.
_STATE_OBS_KEYS = (
    "r_joint_1", "r_joint_2", "r_joint_3", "r_joint_4",
    "r_joint_5", "r_joint_6", "r_joint_7", "r_gripper",
)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _current_ee_pose(obs: dict) -> np.ndarray:
    """Return [x, y, z, qx, qy, qz, qw, gripper] for the right arm via FK."""
    q = np.array([obs[f"r_joint_{i}"] for i in range(1, 8)])
    pos, quat_xyzw = franka_fk(q)
    return np.concatenate([pos, quat_xyzw, [obs["r_gripper"]]]).astype(np.float32)


def _format_obs_for_policy(obs: dict) -> dict:
    """Reformat raw robot obs into the observation.* keyed format the lerobot preprocessor expects.

    batch_to_transition (the preprocessor's to_transition function) only picks up keys that
    start with "observation.". The standard lerobot record loop adds this prefix via
    build_dataset_frame; we replicate that mapping here for SingleArmFranka.
    """
    formatted: dict = {
        "observation.state": np.array([obs[k] for k in _STATE_OBS_KEYS], dtype=np.float32),
    }
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:  # HWC camera image
            formatted[f"observation.images.{k}"] = v
    return formatted


_DEPTH_POINT_COUNT = 2048
_DEPTH_FLAT_SIZE = _DEPTH_POINT_COUNT * 3


def _extract_point_cloud(obs: dict) -> np.ndarray:
    """Reconstruct (2048, 3) point cloud from flat depth_* scalars in obs."""
    flat = np.array([obs[f"depth_{i}"] for i in range(_DEPTH_FLAT_SIZE)], dtype=np.float32)
    return flat.reshape(_DEPTH_POINT_COUNT, 3)


def _strip_depth(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("depth_")}


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

def process_chunk(chunk: np.ndarray, current_ee_pose: np.ndarray) -> np.ndarray:
    """Convert the first _CHUNK_EXEC steps of a base-policy chunk to normalised deltas.

    Args:
        chunk: (T, 10) array — [r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw, r_gripper, kp, kd].
        current_ee_pose: (8,) — [x, y, z, qx, qy, qz, qw, gripper] current robot state.

    Returns:
        (5, 9) — [dx, dy, dz, rx, ry, rz, gripper, kp, kd] normalised.
    """
    result = np.zeros((_CHUNK_EXEC, 9), dtype=np.float32)
    prev_pos = current_ee_pose[:3]
    prev_quat = current_ee_pose[3:7]

    for i in range(_CHUNK_EXEC):
        step = chunk[i]
        next_pos = step[:3]
        next_quat = step[3:7]

        delta_pos = (next_pos - prev_pos) / _POS_SCALE
        delta_rot = (
            Rotation.from_quat(next_quat) * Rotation.from_quat(prev_quat).inv()
        ).as_rotvec() / _ROT_SCALE

        result[i] = np.array([*delta_pos, *delta_rot, step[7], step[8], step[9]], dtype=np.float32)

        prev_pos = next_pos
        prev_quat = next_quat

    return result


def _build_action(chunk_step: np.ndarray, kp: float, kd: float) -> dict:
    """Build a RobotAction dict from a base-policy chunk row, overriding gains."""
    action = {k: float(v) for k, v in zip(_EE_ACTION_KEYS, chunk_step[:8])}
    action["kp"] = kp
    action["kd"] = kd
    return action


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class BasePolicy:
    """Thin wrapper around a pretrained lerobot ACT / diffusion policy."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(path, config=cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, pretrained_path=path)

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, obs: dict) -> np.ndarray:
        """Run one inference pass.

        Returns:
            (T, 10) numpy array in robot action space (unnormalised).
        """
        obs_t = prepare_observation_for_inference(_format_obs_for_policy(obs), self.device)
        obs_t = self.preprocessor(obs_t)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(obs_t)  # (1, T, action_dim)
        chunk = chunk.squeeze(0)  # (T, action_dim)

        # Unnormalise each step via the postprocessor.
        # For stateful postprocessors (e.g. relative actions), revisit this loop.
        steps = []
        for i in range(chunk.shape[0]):
            step = self.postprocessor(chunk[i : i + 1]).squeeze(0).cpu().numpy()
            steps.append(step)
        return np.stack(steps)  # (T, action_dim)


class ResidualPolicy:
    """Skeleton residual policy — returns zero corrections until a model is wired in."""

    def __init__(self) -> None:
        pass

    def infer(self, obs: dict) -> np.ndarray:
        """Args:
            obs: dict with keys:
                "action_chunk" (5, 9)  — normalised delta chunk from base policy
                "proprio"      (9,)    — current EE pose + left_gripper (-1.0, 0.0) + right_gripper (0.0, 1.0)
                "point_cloud"  (2048, 3)
                "gains"        (2,)    — [prev_kp, prev_kd]

        Returns:
            (8,) — [dx, dy, dz, rx, ry, rz, kp, kd], all normalised.
        """
        res = np.zeros(8, dtype=np.float32)
        res[2] = -1.0
        res[6] = 0.5
        res[7] = 0.0
        return res


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", required=True, help="Path to base policy checkpoint")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda/cpu)")
    parser.add_argument(
        "--home-pose-name",
        default=None,
        help=f"Name of a saved pose JSON in {_POSES_DIR} (overrides --home-q)",
    )
    parser.add_argument(
        "--home-q", nargs=7, type=float, default=_DEFAULT_HOME_Q,
        help="7 joint angles (rad) for the right arm home pose",
    )
    parser.add_argument("--home-gripper", type=float, default=1.0)
    parser.add_argument("--home-max-time-s", type=float, default=3.0)
    parser.add_argument("--home-tol-rad", type=float, default=0.05)
    parser.add_argument("--home-tol-m", type=float, default=0.025)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.home_pose_name:
        pose = json.loads((_POSES_DIR / f"{args.home_pose_name}.json").read_text())
        home_q = np.asarray(pose["r_q"], dtype=np.float64)
        home_gripper = float(pose.get("gripper", args.home_gripper))
    else:
        home_q = np.asarray(args.home_q, dtype=np.float64)
        home_gripper = args.home_gripper

    print("attempting connection to robot...")
    controller = env_wrapper.start_controller()
    print("robot initialized!")

    print("homing...")
    ok = controller.home(
        home_q_left=None,
        home_q_right=home_q,
        gripper_norm=home_gripper,
        max_time_s=args.home_max_time_s,
        tol_rad=args.home_tol_rad,
        tol_pos_m=args.home_tol_m,
    )
    if not ok:
        logger.warning("homing did not converge; proceeding anyway")

    print(f"attempting to start base policy: {args.base_policy}")
    base_policy = BasePolicy(args.base_policy, device=args.device)
    print("base policy started!")
    print("attempting to start residual policy")
    residual = ResidualPolicy()
    print("residual policy started")

    chunk: np.ndarray = np.empty((0, 10))
    chunk_used = _CHUNK_EXEC   # triggers immediate inference on first step
    prev_kp = 0.0
    prev_kd = 0.0

    try:
        while True:
            obs = controller.get_observation()
            ee_pose = _current_ee_pose(obs)
            point_cloud = _extract_point_cloud(obs)
            obs_no_depth = _strip_depth(obs)

            if chunk_used >= _CHUNK_EXEC:
                chunk = base_policy.infer(obs_no_depth)
                chunk_used = 0

            res_chunk = process_chunk(chunk, ee_pose)

            residual_obs = {
                "action_chunk": res_chunk,                                      # (5, 9)
                "proprio": ee_pose,                                             # (8,)
                "point_cloud": point_cloud,                                     # (2048, 3)
                "gains": np.array([prev_kp, prev_kd], dtype=np.float32),       # (2,)
            }
            res = residual.infer(residual_obs)  # (8,) [dx, dy, dz, rx, ry, rz, kp, kd]

            dpos = res[:3] * _POS_SCALE
            drot = res[3:6] * _ROT_SCALE
            controller.cache_delta(dpos, drot)

            action = _build_action(chunk[chunk_used], kp=float(res[6]), kd=float(res[7]))
            controller.send_action(action)

            prev_kp = float(res[6])
            prev_kd = float(res[7])
            chunk_used += 1

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
