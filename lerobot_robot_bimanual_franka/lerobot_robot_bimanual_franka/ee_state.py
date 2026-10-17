"""observation.state as the baselines read it: end-effector pose and gripper.

A recording stores [joint_1 .. joint_7, gripper]. SAIL and B-Spline are trained
on the O_T_EE pose those joints reach (baselines/common.py, eef_poses_from_qpos),
so our diffusion policy reads the same 8 numbers: [x, y, z, qx, qy, qz, qw,
gripper], position in metres in the robot base frame, quaternion xyzw.

JointStateToEEPose is a preprocessor step. A pipeline saves it by class path in
policy_preprocessor.json and imports it on load, so whatever loads a checkpoint
converts the robot's joint state exactly as training did, and a checkpoint
without the step keeps reading joints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.processor import ObservationProcessorStep
from lerobot.utils.constants import OBS_STATE

import franka_config as fc

from .ee_kinematics import eef_poses_from_qpos

NUM_JOINTS = fc.num_joints()
_QUANTILES = (1, 10, 50, 90, 99)


def ee_state_from_joint_state(state: np.ndarray) -> np.ndarray:
    """(..., num_joints + 1) [joints, gripper] -> (..., 8) [pos, quat_xyzw, gripper]."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] != NUM_JOINTS + 1:
        raise ValueError(f"expected [{NUM_JOINTS} joints, gripper] states, got width {state.shape[-1]}")
    flat = state.reshape(-1, state.shape[-1])
    pos, quat = eef_poses_from_qpos(flat[:, :NUM_JOINTS])
    return np.concatenate([pos, quat, flat[:, NUM_JOINTS:]], axis=1).reshape(*state.shape[:-1], 8)


def ee_state_stats(joint_states: np.ndarray) -> dict[str, np.ndarray]:
    """The normaliser stats LeRobot keeps per feature, over the converted states."""
    rows = ee_state_from_joint_state(joint_states)
    stats = {"min": rows.min(0), "max": rows.max(0), "mean": rows.mean(0), "std": rows.std(0),
             "count": np.array([len(rows)], dtype=np.int64)}
    stats.update({f"q{q:02d}": np.percentile(rows, q, axis=0) for q in _QUANTILES})
    return stats


def uses_ee_state(pretrained_dir: str | Path) -> bool:
    """Whether a saved checkpoint's preprocessor converts joints to the EE pose."""
    config = Path(pretrained_dir) / "policy_preprocessor.json"
    if not config.is_file():
        return False
    steps = json.loads(config.read_text()).get("steps", [])
    return any(s.get("class", "").endswith(f".{JointStateToEEPose.__name__}") for s in steps)


@dataclass
class JointStateToEEPose(ObservationProcessorStep):
    """Replace a [joints, gripper] observation.state with [pos, quat_xyzw, gripper]."""

    def observation(self, observation: dict) -> dict:
        state = observation.get(OBS_STATE)
        if state is None:
            return observation
        if torch.is_tensor(state):
            ee = ee_state_from_joint_state(state.detach().cpu().double().numpy())
            observation[OBS_STATE] = torch.as_tensor(ee, dtype=state.dtype, device=state.device)
        else:
            observation[OBS_STATE] = ee_state_from_joint_state(state).astype(np.asarray(state).dtype)
        return observation

    def transform_features(self, features):
        return features  # 8 numbers in, 8 out
