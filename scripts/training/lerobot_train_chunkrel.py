#!/usr/bin/env python
"""lerobot-train on an EE_POS recording, with every training window's actions made chunk-relative.

Each absolute goal pose in a window is re-expressed relative to the measured pose
(O_T_EE, by FK of the joints) at the window's latest observation: the anchor
run_residual.py takes a base chunk relative to (env_wrapper.chunk_to_relative).
A frame's label depends on which window holds it, so this cannot be written into
the dataset per frame; ChunkRelativeDataset applies it per sample, and the action
stats the normaliser is built from are recomputed over the windows the sampler draws.

The policy then emits (T, 9) [dx, dy, dz, rx, ry, rz, gripper, kp, kd]: position
target - anchor (m, base frame), rotation the rotvec of R_target R_anchor^-1 (rad),
gripper/kp/kd unchanged. That is chunk_to_relative before its /pos_scale_m and
/rot_scale_rad.

  python scripts/training/lerobot_train_chunkrel.py <lerobot-train flags>
  python scripts/training/lerobot_train_chunkrel.py --describe <repo_id>
"""

import copy
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.datasets.factory import make_dataset
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos
from lerobot_robot_bimanual_franka.lerobot_source import (
    GRIPPER, POS, QUAT, arm_prefix, check_action_space, load_frames, resolve_root,
)

_NUM_JOINTS = 7
_QUANTILES = (1, 10, 50, 90, 99)


def to_chunk_relative(actions: np.ndarray, anchor_pos: np.ndarray, anchor_quat: np.ndarray) -> np.ndarray:
    """(T, 10) absolute [x,y,z,qx,qy,qz,qw,grip,kp,kd] -> (T, 9) relative to one anchor pose."""
    actions = np.asarray(actions, dtype=np.float64)
    out = np.empty((len(actions), 9), dtype=np.float32)
    out[:, 0:3] = actions[:, POS] - anchor_pos
    out[:, 3:6] = (Rotation.from_quat(actions[:, QUAT]) * Rotation.from_quat(anchor_quat).inv()).as_rotvec()
    out[:, 6:9] = actions[:, GRIPPER:]
    return out


def action_names(prefix: str) -> list[str]:
    return [f"{prefix}_{k}" for k in ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")] + ["kp", "kd"]


def window_rows(root, action_deltas: list[int], drop_n_last: int, episodes=None) -> np.ndarray:
    """Every chunk-relative action row the sampler can draw, padding included, as (N, 9)."""
    df, _ = load_frames(root)
    df = df.sort_values("index")
    index = df["index"].to_numpy()
    if not np.array_equal(index, np.arange(len(index))):
        raise ValueError(f"{root}: frame index is not contiguous from 0")
    actions = np.stack(df[ACTION].to_numpy()).astype(np.float64)
    check_action_space(actions)
    anchor_pos, anchor_quat = eef_poses_from_qpos(np.stack(df[OBS_STATE].to_numpy())[:, :_NUM_JOINTS])
    deltas = np.asarray(action_deltas)
    rows = []
    for ep, frames in df.groupby("episode_index")["index"]:
        if episodes is not None and ep not in episodes:
            continue
        start, end = int(frames.min()), int(frames.max()) + 1
        for t in range(start, end - drop_n_last):
            # LeRobot pads past an episode edge by repeating the edge frame.
            window = np.clip(t + deltas, start, end - 1)
            rows.append(to_chunk_relative(actions[window], anchor_pos[t], anchor_quat[t]))
    return np.concatenate(rows)


def stats_of(rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = rows.astype(np.float64)
    stats = {"min": rows.min(0), "max": rows.max(0), "mean": rows.mean(0), "std": rows.std(0),
             "count": np.array([len(rows)], dtype=np.int64)}
    stats.update({f"q{q:02d}": np.percentile(rows, q, axis=0) for q in _QUANTILES})
    return stats


class ChunkRelativeDataset(torch.utils.data.Dataset):
    """A LeRobotDataset whose action window is relative to the pose at its latest observation."""

    def __init__(self, dataset, policy_cfg) -> None:
        self.dataset = dataset
        self._now = list(policy_cfg.observation_delta_indices).index(0)
        prefix = arm_prefix(dataset.meta.info)
        rows = window_rows(dataset.root, policy_cfg.action_delta_indices,
                           getattr(policy_cfg, "drop_n_last_frames", 0), dataset.episodes)
        # The reader keeps the original meta: it still decodes the stored 10-wide action.
        self.meta = copy.copy(dataset.meta)
        self.meta.info = copy.deepcopy(dataset.meta.info)
        self.meta.info["features"][ACTION] = {"dtype": "float32", "shape": [9], "names": action_names(prefix)}
        self.meta.stats = {**dataset.meta.stats, ACTION: stats_of(rows)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        item = self.dataset[idx]
        q = item[OBS_STATE][self._now, :_NUM_JOINTS].double().numpy()
        pos, quat = eef_poses_from_qpos(q[None])
        item[ACTION] = torch.from_numpy(to_chunk_relative(item[ACTION].numpy(), pos[0], quat[0]))
        return item

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def make_chunkrel_dataset(cfg):
    if cfg.dataset.streaming:
        raise SystemExit("chunk-relative training needs a local dataset; drop --dataset.streaming")
    return ChunkRelativeDataset(make_dataset(cfg), cfg.policy)


def describe(repo_id: str) -> None:
    """The relative action stats over diffusion's default windows."""
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    cfg = DiffusionConfig()
    root = resolve_root(repo_id)
    _, info = load_frames(root)
    rows = window_rows(root, cfg.action_delta_indices, cfg.drop_n_last_frames)
    stats = stats_of(rows)
    print(f"{root}: {len(rows)} action rows")
    for i, name in enumerate(action_names(arm_prefix(info))):
        print(f"  {name:>10}  min {stats['min'][i]:+.4f}  max {stats['max'][i]:+.4f}  "
              f"mean {stats['mean'][i]:+.4f}  std {stats['std'][i]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--describe":
        describe(sys.argv[2])
        sys.exit(0)
    lerobot_train.make_dataset = make_chunkrel_dataset
    lerobot_train.main()
