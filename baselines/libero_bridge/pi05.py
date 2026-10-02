#!/usr/bin/env python3
"""multi-fast's base policy (pi0.5) as a third backend for the LIBERO rollout.

SAIL and B-Spline are complete methods. multi-fast is a base policy plus a
learned residual, so its comparable row today is the base half: the
pi0.5-LIBERO checkpoint the residual is trained on top of. That is what makes
the table three-way now. When a FAST residual exists for these tasks it goes on
top of exactly this policy and files itself as `multifast`; this files as
`pi05`, because reporting a base as the whole method would overstate it.

Nothing here reimplements the policy. `load` builds the config block
multi-fast's own `load_base_policy` expects and hands back its
`Pi05BaseWrapper`, so the observation translation -- the 8-D state vector, the
224x224 resize-with-pad, the action unnormalisation -- is the same code
multi-fast trains and evaluates through. The only thing this module owns is
turning one LIBERO observation into the batch-of-one that wrapper reads.

Unlike the two baselines, pi0.5 emits LIBERO's OWN action space: seven
normalised OSC deltas. There is no absolute pose to invert, so the rollout
steps the env with what comes back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from baselines.libero_bridge.sim_env import policy_frame

_MULTI_FAST = Path(__file__).resolve().parent.parent.parent / "multi-fast"

# Every value below is cfg/libero/fast_libero_90.yaml's, so this backend runs
# the base policy multi-fast itself runs. openpi caches the checkpoint under
# ~/.cache/openpi, so the gs:// path costs nothing after the first load.
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
CONFIG_NAME = "pi05_libero"
CHUNK_SIZE = 5            # pi0.5 replan frequency
PREDICTION_HORIZON = 10   # full chunk length
ACTION_DIM = 7

LOW_DIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def render_resolution() -> int:
    """What the env must render at for this policy, read from multi-fast.

    Not the 256 the converter used for the baselines: multi-fast evaluates
    pi0.5 on its own ENV_RESOLUTION, and the checkpoint is sensitive to it --
    at 256 the stove task scores 0/20 against 9/10 here, because pi0.5's
    resize-with-pad to 224 then downsamples a sharper image than any of its
    recorded success rates were measured on.
    """
    if str(_MULTI_FAST) not in sys.path:
        sys.path.insert(0, str(_MULTI_FAST))
    from utils.envs.libero import ENV_RESOLUTION
    return int(ENV_RESOLUTION)


def load(suite: str, task_index: int, *, chunk_size: int = CHUNK_SIZE,
         checkpoint: str = CHECKPOINT, config_name: str = CONFIG_NAME,
         prediction_horizon: int = PREDICTION_HORIZON):
    """multi-fast's own `Pi05BaseWrapper`, loaded from its own loader.

    The task instruction is left null so the loader resolves it from the LIBERO
    benchmark API, which is how multi-fast resolves it -- retyping the sentence
    here is how a rollout ends up prompting for a different task than it runs.
    """
    if str(_MULTI_FAST) not in sys.path:
        sys.path.insert(0, str(_MULTI_FAST))
    # Before the first JAX import, or it has no effect. JAX otherwise grabs 90%
    # of the card on initialisation, which would kill whatever else is on it --
    # and a rollout is exactly the thing that runs while the next task trains.
    # openpi's own data loader sets the same pair. Override either in the
    # environment; multi-fast's cluster scripts cap by MEM_FRACTION instead.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    from omegaconf import OmegaConf

    from utils.base_policy_utils import load_base_policy

    cfg = OmegaConf.create({
        "base_policy": {
            "type": "pi05",
            "checkpoint": checkpoint,
            "config_name": config_name,
            "chunk_size": int(chunk_size),
            "prediction_horizon": int(prediction_horizon),
            "task_instruction": None,
            "observation_meta": {
                "low_dim_keys": list(LOW_DIM_KEYS),
                "image_keys": list(IMAGE_KEYS),
            },
        },
        "libero": {"suite": suite, "task_id": int(task_index)},
    })
    return load_base_policy(cfg)


def observation(raw: dict, flip: bool = True) -> dict:
    """One LIBERO observation -> the batch-of-one `Pi05BaseWrapper` reads.

    Images are CHW float in [0, 1] and already rotated 180 degrees, because
    multi-fast's LIBEROObservationWrapper does both before the wrapper sees
    them and its `_translate_obs` assumes it has been done. `policy_frame` with
    no size does the rotation and nothing else; pi0.5's own resize-with-pad to
    224 happens inside the wrapper.
    """
    obs = {k: np.asarray(raw[k], dtype=np.float32)[None] for k in LOW_DIM_KEYS}
    for key in IMAGE_KEYS:
        frame = policy_frame(raw[key], None, flip)
        obs[key] = (np.transpose(frame, (2, 0, 1))[None].astype(np.float32) / 255.0)
    return obs


def meta(policy, chunk_size: int, train_dataset: str | None = None) -> dict:
    """What goes in the run manifest, in the shape the ZMQ backends' meta has.

    `train_dataset` is the `<suite>/task_<i>` the baselines' converter stamps on
    its HDF5s, so a pi0.5 run lands in the same directory as the two policies it
    is being compared against. pi0.5 was pretrained across LIBERO rather than on
    one task's file, so nothing else here would supply it.
    """
    return {
        "backend": "pi05",
        "train_dataset": train_dataset,
        "act_dim": ACTION_DIM,
        "action_keys": ["actions"],
        "action_horizon": int(chunk_size),
        "prediction_horizon": int(getattr(policy, "prediction_horizon", PREDICTION_HORIZON)),
        "checkpoint": CHECKPOINT,
        "config_name": CONFIG_NAME,
        "task_instruction": getattr(policy, "task_instruction", None),
        "obs_keys": list(LOW_DIM_KEYS) + list(IMAGE_KEYS),
        "in_process": True,
    }
