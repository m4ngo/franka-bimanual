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

Residual output     : (_CHUNK_EXEC, 9) chunk — [damping, stiffness, dx, dy, dz, rx, ry, rz, grip_delta]
                      (gains first, DAMPING before stiffness — multi-fast convention)
                      per step (normalised, same scales as input).  Gains are at
                      indices 0–1, positional/rotational deltas at 2–7, gripper at 8.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from env_wrapper import _STATE_OBS_KEYS, _CHUNK_EXEC, _GAINS_MAG, _RESIDUAL_MAG, _RESIDUAL_TRANS_MAG, _RESIDUAL_ROT_MAG
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference, populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES
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
        obs_t = prepare_observation_for_inference(_format_obs_for_policy(obs), self.device)
        obs_t = self.preprocessor(obs_t)
        obs_only = {k: v for k, v in obs_t.items() if k.startswith("observation.")}

        with torch.inference_mode():
            if hasattr(self.policy, "_queues") and self.policy._queues is not None:
                # Mirror what select_action does before calling predict_action_chunk (lerobot 0.5.1
                # has no offline mode for predict_action_chunk — queues must be pre-populated).
                batch_for_queues = dict(obs_only)
                if self.policy.config.image_features:
                    batch_for_queues[OBS_IMAGES] = torch.stack(
                        [batch_for_queues[key] for key in self.policy.config.image_features], dim=-4
                    )
                self.policy._queues = populate_queues(self.policy._queues, batch_for_queues)
                chunk = self.policy.predict_action_chunk(batch_for_queues)  # (1, T, action_dim)
            else:
                chunk = self.policy.predict_action_chunk(obs_only)
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
        # print(ckpt["model_init_kwargs"])
        # ckpt["model_init_kwargs"]["disable_pcd"] = True
        if "model_init_kwargs" not in ckpt:
            raise KeyError("Checkpoint missing 'model_init_kwargs'.")

        encoder_type = ckpt["model_init_kwargs"].get("encoder_type", "pointnet_lite")
        policy_cls = (
            CrossAttentionPolicy
            # if encoder_type in ("pointnet_xa", "pointnet_xa2")
            # else MultiTaskPointCloudPolicy
        )
        self.model = policy_cls(**ckpt["model_init_kwargs"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        # Preprocessing comes from the checkpoint's data_kwargs (the training
        # contract), never hardcoded.
        data_kwargs = ckpt.get("data_kwargs", {})
        self.center_on_eef: bool = bool(data_kwargs.get("center_on_eef", False))
        self.num_points: int = int(data_kwargs.get("num_points", 2048))
        self.use_rgb: bool = bool(data_kwargs.get("use_rgb", False))
        crop = data_kwargs.get("crop_half_extent", None)
        self.crop_half_extent: float | None = None if crop is None else float(crop)
        if self.use_rgb:
            raise ValueError(
                "Checkpoint trained with use_rgb=True; the real cloud is xyz-only. "
                "Retrain without RGB or use a different checkpoint."
            )
        # Exact cloud fed to the network on the most recent infer() call
        # (post crop/downsample/re-centering); for diagnostics.
        self.last_network_pcd: np.ndarray | None = None
        logger.info(
            "ResidualPolicy loaded: cls=%s encoder=%s center_on_eef=%s num_points=%d "
            "crop_half_extent=%s (frame=%s proprio_keys=%s)",
            policy_cls.__name__, encoder_type, self.center_on_eef, self.num_points,
            self.crop_half_extent, data_kwargs.get("frame"), data_kwargs.get("proprio_keys"),
        )

    def _prepare_pcd(self, pcd: np.ndarray, eef_pos: np.ndarray) -> np.ndarray:
        """Apply the checkpoint's preprocessing: crop -> resample -> center
        (same order as the sim dataset path)."""
        pcd = pcd[:, :3]
        if self.crop_half_extent is not None:
            mask = np.all(np.abs(pcd - eef_pos) <= self.crop_half_extent, axis=1)
            if mask.any():
                pcd = pcd[mask]
        if len(pcd) != self.num_points:
            idx = np.random.choice(len(pcd), self.num_points, replace=len(pcd) < self.num_points)
            pcd = pcd[idx]
        pcd = pcd.copy()
        if self.center_on_eef:
            pcd -= eef_pos
        return pcd

    def infer(self, obs: dict) -> np.ndarray:
        """Run one residual inference pass.

        Args:
            obs: dict with keys:
                "action_chunk" (10, 9) — normalised delta chunk from base policy
                                         columns 0:7 = [dx, dy, dz, rx, ry, rz, grip]
                                         columns 7:9 = [kp, kd] (dropped before model)
                "proprio"      (17,)    — [x, y, z, qx, qy, qz, qw, finger_qpos_m, -finger_qpos_m,
                                          damping_norm, kp_norm, vx, vy, vz, wx, wy, wz]
                "point_cloud"  (2048, 3) — xyz in robot/world frame

        Returns:
            (_CHUNK_EXEC, 9) — per step: [damping, stiffness, dx, dy, dz, rx, ry, rz, grip_delta]
                               all normalised (gains-first layout from variable-impedance
                               sim training).
        """
        action_chunk: np.ndarray = obs["action_chunk"]   # (10, 9)
        proprio: np.ndarray = obs["proprio"]              # (9,)
        point_cloud: np.ndarray = obs["point_cloud"]     # (2048, 3)

        # Feed only the 7 per-step channels the model was trained on; drop kp/kd.
        base_action = action_chunk[:, :7].flatten().astype(np.float32)  # (70,)

        pcd = self._prepare_pcd(point_cloud.astype(np.float32), proprio[:3])
        self.last_network_pcd = pcd

        pcd_t = torch.as_tensor(pcd, dtype=torch.float32, device=self.device).unsqueeze(0)
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=self.device).unsqueeze(0)
        base_action_t = torch.as_tensor(base_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            out = self.model(pcd_t, proprio_t, base_action_t)  # (1, 45)

        result = out.squeeze(0).cpu().numpy().reshape(_CHUNK_EXEC, 9)  # (5, 9)
        result[..., :2] = np.clip(result[..., :2], -_GAINS_MAG, _GAINS_MAG)
        result[..., 2:5] = np.clip(result[..., 2:5], -_RESIDUAL_TRANS_MAG, _RESIDUAL_TRANS_MAG)
        result[..., 5:8] = np.clip(result[..., 5:8], -_RESIDUAL_ROT_MAG, _RESIDUAL_ROT_MAG)
        result[..., 8] = np.clip(result[..., 8], -_RESIDUAL_MAG, _RESIDUAL_MAG)
        return result
