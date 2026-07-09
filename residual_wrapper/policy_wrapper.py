"""
Policy wrappers for the base (LeRobot ACT/diffusion) and residual policies.

Action spaces
-------------
Base policy output  : [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, kp, kd]
                      Per-step EE delta in robot frame, in physical units as returned
                      by the lerobot postprocessor.  Position columns (0–2) are in
                      metres (the units stored in the training dataset).  Rotation
                      columns (3–6) encode the delta as a unit quaternion (xyzw).
                      These are forwarded directly to send_action() in EE_DELTA mode.

Residual input chunk: (_RESIDUAL_HORIZON, 9) normalised per-step deltas.
                      [dx, dy, dz, rx, ry, rz, gripper, kp, kd]
                      Position deltas normalised to [-1, 1] where ±1 = ±0.05 m.
                      Rotation deltas (axis-angle rotvec) normalised where ±1 = ±0.5 rad.
                      Derived by converting each base-chunk step's delta quat to a rotvec
                      and dividing by the respective scales.

Residual output     : (_CHUNK_EXEC, 9) chunk — [kp, kd, dx, dy, dz, rx, ry, rz, grip_delta]
                      per step (normalised, same scales as input).  Gains are at
                      indices 0–1, positional/rotational deltas at 2–7, gripper at 8.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from env_wrapper import _STATE_OBS_KEYS, _CHUNK_EXEC, _GAINS_MAG, _RESIDUAL_MAG
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.datasets import LeRobotDataset 

logger = logging.getLogger(__name__)

_MULTI_FAST_PATH = Path(__file__).resolve().parent.parent / "multi-fast"


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
            (T, 10) numpy array in physical units (postprocessor applied):
            positions in metres, rotation as unit quaternion (xyzw), gripper in [0,1].
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
    
class Trajectory(BasePolicy):
    def __init__(self, path: str, device: str = "cuda") -> None:
        self.reset()
        actions = LeRobotDataset(path, episodes=[0]).select_columns("action")
        self.trajectory = actions_array = np.array(actions["action"]) 
    
    def reset(self) -> None:
        self.cur_step = 0

    def infer(self, obs: dict) -> np.ndarray:
        """Run one inference pass.

        Returns:
            (T, 10) numpy array in physical units (postprocessor applied):
            positions in metres, rotation as unit quaternion (xyzw), gripper in [0,1].
        """
        res = self.trajectory[self.cur_step : self.cur_step + 10]
        self.cur_step += 5
        return res


class ResidualPolicy:
    """CrossAttentionPolicy residual loaded from a best.pt checkpoint.

    Mirrors the dispatch in multi-fast/eval_distill.py (lines 629-644): reads
    encoder_type from ckpt["model_init_kwargs"] and routes to CrossAttentionPolicy
    or MultiTaskPointCloudPolicy accordingly.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        import sys
        if str(_MULTI_FAST_PATH) not in sys.path:
            sys.path.insert(0, str(_MULTI_FAST_PATH))
        from utils.distill.policy import CrossAttentionPolicy, MultiTaskPointCloudPolicy

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_init_kwargs" not in ckpt:
            raise KeyError("Checkpoint missing 'model_init_kwargs'.")

        encoder_type = ckpt["model_init_kwargs"].get("encoder_type", "pointnet_lite")
        policy_cls = (
            CrossAttentionPolicy
            if encoder_type in ("pointnet_xa", "pointnet_xa2")
            else MultiTaskPointCloudPolicy
        )
        self.model = policy_cls(**ckpt["model_init_kwargs"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        data_kwargs = ckpt.get("data_kwargs", {})
        self.center_on_eef: bool = bool(data_kwargs.get("center_on_eef", False))
        logger.info(
            "ResidualPolicy loaded: cls=%s encoder=%s center_on_eef=%s",
            policy_cls.__name__, encoder_type, self.center_on_eef,
        )

    def infer(self, obs: dict) -> np.ndarray:
        """Run one residual inference pass.

        Args:
            obs: dict with keys:
                "action_chunk" (10, 9) — normalised delta chunk from base policy
                                         columns 0:7 = [dx, dy, dz, rx, ry, rz, grip]
                                         columns 7:9 = [kp, kd] (dropped before model)
                "proprio"      (9,)    — [x, y, z, qx, qy, qz, qw, grip_r, -grip_r,
                                          kp, kd, vx, vy, vz, wx, wy, wz]
                "point_cloud"  (2048, 3) — xyz in robot/world frame

        Returns:
            (_CHUNK_EXEC, 9) — per step: [kp, kd, dx, dy, dz, rx, ry, rz, grip_delta]
                               all normalised (gains-first layout from variable-impedance
                               sim training).
        """
        action_chunk: np.ndarray = obs["action_chunk"]   # (10, 9)
        proprio: np.ndarray = obs["proprio"]              # (9,)
        point_cloud: np.ndarray = obs["point_cloud"]     # (2048, 3)

        # Feed only the 7 per-step channels the model was trained on; drop kp/kd.
        base_action = action_chunk[:, :7].flatten().astype(np.float32)  # (70,)

        pcd = point_cloud.astype(np.float32)
        if self.center_on_eef:
            pcd = pcd.copy()
            pcd[:, :3] -= proprio[:3]  # subtract EEF xyz

        pcd_t = torch.as_tensor(pcd, dtype=torch.float32, device=self.device).unsqueeze(0)
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=self.device).unsqueeze(0)
        base_action_t = torch.as_tensor(base_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            out = self.model(pcd_t, proprio_t, base_action_t)  # (1, 45)

        result = out.squeeze(0).cpu().numpy().reshape(_CHUNK_EXEC, 9)  # (5, 9)
        result[..., :2] = np.clip(result[..., :2], -_GAINS_MAG, _GAINS_MAG)
        result[..., 2:] = np.clip(result[..., 2:], -_RESIDUAL_MAG, _RESIDUAL_MAG)
        return result
