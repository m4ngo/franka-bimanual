"""A pretrained LeRobot policy planning one chunk from a window of observations.

The inference half of residual_wrapper/policy_wrapper.BasePolicy, with nothing
robot-side: `infer_frames` takes observations already keyed the way the
policy's preprocessor reads them (`observation.state`, `observation.images.*`).
BasePolicy formats the arm's observation and calls this; the sim's policy
server (baselines/dp_bridge/policy_server.py) formats LIBERO's and calls the
same, so a base policy plans identically on the arm and in sim.

Imports lerobot and torch only, so it loads in an image with neither the robot
driver nor franka_config.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
from lerobot.utils.constants import OBS_IMAGES

logger = logging.getLogger(__name__)

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}


class ChunkPolicy:
    """The checkpoint, its saved pre/post-processors, and one chunk per call.

    `amp` and `compile_mode` are the levers on its cost, which is the diffusion
    UNet's sequential denoising calls: fp16 plus a compiled UNet runs it in
    21.8 ms against 33.0 ms.
    """

    def __init__(self, path: str, device: str = "cuda",
                 amp: str | None = None, compile_mode: str | None = None) -> None:
        amp_key = str(amp or "none").lower()
        if amp_key not in AMP_DTYPES:
            raise ValueError(f"amp must be one of {sorted(AMP_DTYPES)}, got {amp!r}")
        self._amp_dtype = AMP_DTYPES[amp_key]
        self.device = torch.device(device)
        self.path = Path(path)

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        self.config = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(cfg, pretrained_path=path)
        self.n_obs_steps = int(getattr(cfg, "n_obs_steps", 1) or 1)
        self.policy.reset()
        self._queued = getattr(self.policy, "_queues", None) is not None  # diffusion's obs queue
        if compile_mode and compile_mode.lower() != "none":
            self._compile_unet(compile_mode)

    def reset(self) -> None:
        self.policy.reset()

    def infer_frames(self, frames: list[dict]) -> np.ndarray:
        """(T, action_dim) planned from `frames`, oldest first, each one unbatched
        observation keyed as the dataset's features are.

        A queued policy (diffusion) has the window replayed into a fresh queue, so
        predict_action_chunk stacks what select_action would have queued step by step.
        """
        frames = [self._preprocess(obs) for obs in frames]
        with torch.inference_mode(), self._autocast():
            if self._queued:
                self._replay(frames)
            chunk = self.policy.predict_action_chunk(frames[-1])  # (1, T, action_dim)
        # fp32 before the postprocessor: its unnormalisation stats are fp32.
        chunk = chunk.float().squeeze(0)
        return np.stack([self.postprocessor(chunk[i:i + 1]).squeeze(0).cpu().numpy()
                         for i in range(chunk.shape[0])])

    def _preprocess(self, obs: dict) -> dict:
        batch = self.preprocessor(prepare_observation_for_inference(obs, self.device))
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
