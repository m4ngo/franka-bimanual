"""Policy wrappers for the base (LeRobot ACT/diffusion) and residual policies.

Action spaces
-------------
Base policy output  : (T, 10) [x, y, z, qx, qy, qz, qw, gripper, kp, kd]
                      ABSOLUTE EE pose in the robot base frame (EE_POS), in the
                      physical units the lerobot postprocessor returns: metres and a
                      unit quaternion (xyzw). run_residual.py refuses a checkpoint
                      trained on per-step deltas (BasePolicy.action_space).

Residual input chunk: (_RESIDUAL_HORIZON, 9) [dx, dy, dz, rx, ry, rz, gripper, kp, kd]
                      Each base pose relative to the pose the base planned from:
                      position / 0.05 m, rotvec / 0.5 rad (env_wrapper.chunk_to_relative),
                      gripper in robosuite's [-1, 1] (env_wrapper.residual_input).

Residual output     : (_CHUNK_EXEC, 9) [damping, stiffness, dx, dy, dz, rx, ry, rz, grip_delta]
                      per step, normalised as the input. Gains first, damping before
                      stiffness (multi-fast convention).
"""

import contextlib
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from env_wrapper import _CHUNK_EXEC, _GAINS_MAG, _RESIDUAL_MAG, _STATE_OBS_KEYS
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
from lerobot.utils.constants import OBS_IMAGES
from lerobot_robot_bimanual_franka.lerobot_source import action_space as _action_space

logger = logging.getLogger(__name__)

_MULTI_FAST_PATH = Path(__file__).resolve().parent.parent / "multi-fast"
_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}


def _format_obs_for_policy(obs: dict) -> dict:
    """Raw robot obs -> the observation.* keys the lerobot preprocessor picks up,
    as the record loop's build_dataset_frame names them."""
    formatted = {"observation.state": np.array([obs[k] for k in _STATE_OBS_KEYS], dtype=np.float32)}
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:  # HWC camera image
            formatted[f"observation.images.{k}"] = v
    return formatted


class ObservationHistory:
    """A base policy's view of the episode: one observation per control step.

    Kept the way LeRobot's select_action keeps its queue -- the first
    observation repeated until n_obs_steps are held -- so a chunk inferred at a
    boundary plans from consecutive steps, not from the previous boundary.
    """

    n_obs_steps = 1
    _frames: "deque | tuple" = ()

    def reset(self) -> None:
        self._frames = ()

    def observe(self, obs: dict) -> None:
        if self._frames:
            self._frames.append(obs)
        else:
            self._frames = deque([obs] * self.n_obs_steps, maxlen=self.n_obs_steps)

    def window(self) -> list[dict]:
        """The last n_obs_steps observations, oldest first; a copy a worker thread can keep."""
        return list(self._frames)


class BasePolicy(ObservationHistory):
    """A pretrained lerobot ACT / diffusion policy.

    It sets the control loop's step budget, and its cost is GPU compute -- the
    diffusion UNet's sequential denoising calls -- so `amp` and `compile_mode`
    are the levers: fp16 plus a compiled UNet runs it in 21.8 ms against 33.0 ms.
    """

    def __init__(self, path: str, device: str = "cuda",
                 amp: str | None = None, compile_mode: str | None = None) -> None:
        amp_key = str(amp or "none").lower()
        if amp_key not in _AMP_DTYPES:
            raise ValueError(f"amp must be one of {sorted(_AMP_DTYPES)}, got {amp!r}")
        self._amp_dtype = _AMP_DTYPES[amp_key]
        self.device = torch.device(device)
        self.path = Path(path)

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, pretrained_path=path)
        self.n_obs_steps = int(getattr(cfg, "n_obs_steps", 1) or 1)
        self.policy.reset()
        self._queued = getattr(self.policy, "_queues", None) is not None  # diffusion's obs queue
        if compile_mode and compile_mode.lower() != "none":
            self._compile_unet(compile_mode)

    def reset(self) -> None:
        super().reset()
        self.policy.reset()

    def action_space(self) -> str:
        """`EE_POS` or `EE_DELTA`, read off the checkpoint's own unnormaliser stats:
        lerobot_source.action_space's rule, applied to the extremes of the action range."""
        from safetensors.numpy import load_file

        files = sorted(self.path.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
        if not files:
            raise FileNotFoundError(f"no unnormaliser stats under {self.path}; cannot tell whether "
                                    "this checkpoint emits poses or deltas")
        stats = load_file(str(files[0]))
        if "action.min" in stats and "action.max" in stats:
            corners = np.stack([stats["action.min"][:3], stats["action.max"][:3]])
        elif "action.mean" in stats and "action.std" in stats:
            mean, std = stats["action.mean"][:3], stats["action.std"][:3]
            corners = np.stack([mean - 3 * std, mean + 3 * std])
        else:
            raise KeyError(f"{files[0]} holds no action min/max or mean/std")
        rows = np.zeros((2, 7), dtype=np.float64)
        rows[:, :3] = corners
        return _action_space(rows)

    def infer(self, window: list[dict]) -> np.ndarray:
        """(T, 10) planned from `window` (ObservationHistory.window, oldest first).

        A queued policy (diffusion) has the window replayed into a fresh queue, so
        predict_action_chunk stacks what select_action would have queued step by step.
        """
        frames = [self._preprocess(obs) for obs in window]
        with torch.inference_mode(), self._autocast():
            if self._queued:
                self._replay(frames)
            chunk = self.policy.predict_action_chunk(frames[-1])  # (1, T, action_dim)
        # fp32 before the postprocessor: its unnormalisation stats are fp32.
        chunk = chunk.float().squeeze(0)
        return np.stack([self.postprocessor(chunk[i:i + 1]).squeeze(0).cpu().numpy()
                         for i in range(chunk.shape[0])])

    def _preprocess(self, obs: dict) -> dict:
        batch = self.preprocessor(prepare_observation_for_inference(_format_obs_for_policy(obs), self.device))
        return {k: v for k, v in batch.items() if k.startswith("observation.")}

    def _replay(self, frames: list[dict]) -> None:
        """Refill the policy's observation queue from frames, as select_action fills it."""
        image_keys = self.policy.config.image_features
        self.policy.reset()
        for f in frames:
            if image_keys:
                f[OBS_IMAGES] = torch.stack([f[k] for k in image_keys], dim=-4)
            self.policy._queues = populate_queues(self.policy._queues, f)

    def _autocast(self):
        if self._amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self._amp_dtype)

    def _compile_unet(self, mode: str) -> None:
        """Compile the denoising UNet, the only module worth compiling; eager on any failure."""
        unet = getattr(getattr(self.policy, "diffusion", None), "unet", None)
        if unet is None:
            logger.warning("compile_mode=%s ignored: %s has no diffusion.unet",
                           mode, type(self.policy).__name__)
            return
        try:
            self.policy.diffusion.unet = torch.compile(unet, mode=mode)
            logger.info("compiled diffusion UNet (mode=%s); first inference pays the "
                        "compile cost, so warm up before the episode", mode)
        except Exception:
            logger.exception("torch.compile failed; running eager")


class Trajectory(ObservationHistory):
    """Episode 0 of a recording replayed as base chunks: 10 rows per call, advancing one chunk."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        self.trajectory = np.array(LeRobotDataset(path, episodes=[0]).select_columns("action")["action"])
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.cur_step = 0

    def action_space(self) -> str:
        return _action_space(np.asarray(self.trajectory, dtype=np.float64))

    def infer(self, window: list[dict]) -> np.ndarray:
        chunk = self.trajectory[self.cur_step:self.cur_step + 10]
        self.cur_step += _CHUNK_EXEC
        return chunk


class ResidualPolicy:
    """The CrossAttentionPolicy student from a multi-fast best.pt.

    Preprocessing, output clips and the composition bound come from the
    checkpoint -- its data_kwargs and teacher stamp -- as eval_distill's
    StudentPredictor reads them; policy.yaml fills in only for an unstamped one.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        if str(_MULTI_FAST_PATH) not in sys.path:
            sys.path.insert(0, str(_MULTI_FAST_PATH))
        from utils.distill.policy import CrossAttentionPolicy

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_init_kwargs" not in ckpt:
            raise KeyError("Checkpoint missing 'model_init_kwargs'.")
        model_kwargs = ckpt["model_init_kwargs"]
        self.model = CrossAttentionPolicy(**model_kwargs)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        data_kwargs = ckpt.get("data_kwargs", {})
        if data_kwargs.get("use_rgb", False):
            raise ValueError("Checkpoint trained with use_rgb=True; the real cloud is xyz-only. "
                             "Retrain without RGB or use a different checkpoint.")
        self.center_on_eef = bool(data_kwargs.get("center_on_eef", False))
        self.num_points = int(data_kwargs.get("num_points", 2048))
        crop = data_kwargs.get("crop_half_extent")
        self.crop_half_extent = None if crop is None else float(crop)
        self.teacher: dict | None = data_kwargs.get("teacher")
        self._read_bounds(self.teacher or {}, int(model_kwargs.get("output_chunk_size", _CHUNK_EXEC)))

        self.zero_residual = False  # ablation: keep the gains, drop position/rotation/gripper
        self.zero_gains = False     # ablation: keep position/rotation/gripper, stock gains
        self.last_network_pcd: np.ndarray | None = None  # the cloud the network saw last
        logger.info(
            "ResidualPolicy loaded: encoder=%s center_on_eef=%s num_points=%d crop_half_extent=%s "
            "(frame=%s proprio_keys=%s) teacher=%s -> clip gains %.2f residual %.2f, "
            "composed bound %.1f",
            model_kwargs.get("encoder_type", "pointnet_lite"), self.center_on_eef, self.num_points,
            self.crop_half_extent, data_kwargs.get("frame"), data_kwargs.get("proprio_keys"),
            self.teacher, self.gains_mag, self.residual_bound, self.composed_action_bound,
        )

    def _read_bounds(self, teacher: dict, chunk_size: int) -> None:
        if not teacher:
            logger.warning("checkpoint carries no teacher stamp; clipping with policy.yaml's "
                           "residual magnitudes, which may not be what it trained on")
        self.gains_mag = float(teacher.get("gains_mag") or _GAINS_MAG)
        residual_mag = float(teacher.get("residual_mag") or _RESIDUAL_MAG)
        # A cumulative teacher's labels are a cumsum over the chunk.
        self.residual_bound = residual_mag * (chunk_size if teacher.get("cumulative_residual") else 1)
        self.composed_action_bound = float(teacher.get("composed_action_bound") or 1.0)

    def infer(self, obs: dict) -> np.ndarray:
        """(_CHUNK_EXEC, 9) [damping, stiffness, dpos(3), drot(3), dgrip], normalised and clipped.

        obs: "action_chunk" (horizon, 9) from env_wrapper.residual_input, whose kp/kd
        columns the model does not take; "proprio" (17,) [pose(7), finger qpos(2),
        damping, kp, twist(6)]; "point_cloud" (N, 3) in the proprio pose's frame.
        """
        proprio = obs["proprio"]
        pcd = self._prepare_pcd(obs["point_cloud"].astype(np.float32), proprio[:3])
        self.last_network_pcd = pcd
        base_action = obs["action_chunk"][:, :7].flatten().astype(np.float32)
        with torch.inference_mode():
            out = self.model(self._tensor(pcd), self._tensor(proprio), self._tensor(base_action))

        result = out.squeeze(0).cpu().numpy().reshape(_CHUNK_EXEC, 9)
        result[:, :2] = np.clip(result[:, :2], -self.gains_mag, self.gains_mag)
        result[:, 2:] = np.clip(result[:, 2:], -self.residual_bound, self.residual_bound)
        if self.zero_residual:
            result[:, 2:] = 0.0
        if self.zero_gains:
            result[:, :2] = 0.0
        return result

    def _prepare_pcd(self, pcd: np.ndarray, eef_pos: np.ndarray) -> np.ndarray:
        """Crop, resample, center: the sim dataset path's order."""
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

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
