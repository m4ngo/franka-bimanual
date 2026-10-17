#!/usr/bin/env python
"""lerobot-train for `--policy.type=diffusion`, on the recipe SAIL and B-Spline share.

    python scripts/training/lerobot_train_recipe.py --dataset.repo_id=<id> --policy.type=diffusion ...

baselines/diffusion_recipe.py holds the recipe. This adds four things:

  * its flags, ahead of the caller's: the UNet widths, one encoder per camera,
    10 DDIM inference steps, AdamW with no gradient clipping, the cosine
    schedule, batch 64, seed 42. A caller's flag still wins, with a warning. A
    `--resume=true` run gets none; its saved config already carries them.
  * the crop, `train_common.crop_size` of the dataset's image size (203 of 224),
    set on the policy config before the policy is built. Random in training,
    center at inference.
  * EMA, B-Spline's own EMAModel loaded from the submodule, stepped after every
    optimizer step. pretrained_model/ holds the EMA weights, which is what
    inference loads in both baselines too. training_state/ also holds the live
    weights and the EMA's step, so a resume continues both. The Hub push sends
    the EMA weights.
  * the robot state SAIL and B-Spline read: observation.state becomes the
    end-effector pose and gripper [x, y, z, qx, qy, qz, qw, gripper] instead of
    joint angles (lerobot_robot_bimanual_franka.ee_state). The conversion is a
    preprocessor step saved with the checkpoint, so rollouts apply it too, and
    the state's normaliser stats are recomputed over the converted states. A
    resumed checkpoint keeps whichever state it was trained on.

lerobot_train_chunkrel.py applies the same through install() and with_recipe_args().
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import lerobot.scripts.lerobot_train as lerobot_train  # noqa: E402
from lerobot.datasets.feature_utils import dataset_to_policy_features  # noqa: E402
from lerobot.processor import NormalizerProcessorStep  # noqa: E402
from lerobot.utils.constants import OBS_STATE, TRAINING_STATE_DIR  # noqa: E402
from lerobot_robot_bimanual_franka.ee_state import (  # noqa: E402
    JointStateToEEPose, ee_state_stats, uses_ee_state,
)
from lerobot_robot_bimanual_franka.lerobot_source import load_frames  # noqa: E402
from safetensors.torch import load_model, save_model  # noqa: E402

from baselines import diffusion_recipe as recipe  # noqa: E402
from baselines.train_common import crop_size  # noqa: E402

EMA_MODEL_FILE = (_REPO_ROOT / "baselines" / "bspline_policy" / "diffusion_policy" / "diffusion_policy"
                  / "model" / "diffusion" / "ema_model.py")
LIVE_WEIGHTS = "live_model.safetensors"
EMA_STATE = "ema_state.json"

_STATE: dict = {"policy": None, "ema": None, "ds_meta": None}


def _ema_class():
    # Only torch is imported there; the decay rule is then B-Spline's own code.
    spec = importlib.util.spec_from_file_location("bspline_ema_model", EMA_MODEL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EMAModel


def _flag_key(arg: str) -> str | None:
    return arg[2:].split("=", 1)[0] if arg.startswith("--") and "=" in arg else None


def _is_resume(argv: list[str]) -> bool:
    for i, arg in enumerate(argv):
        if arg.startswith("--resume="):
            return arg.split("=", 1)[1].lower() in ("true", "1", "yes")
        if arg == "--resume" and i + 1 < len(argv):
            return argv[i + 1].lower() in ("true", "1", "yes")
    return False


def with_recipe_args(argv: list[str]) -> list[str]:
    """argv with the recipe's flags inserted before the caller's."""
    args = argv[1:]
    if _is_resume(args):
        return list(argv)
    types = [a.split("=", 1)[1] for a in args if a.startswith("--policy.type=")]
    if types != ["diffusion"]:
        raise SystemExit(f"{Path(argv[0]).name} trains --policy.type=diffusion only; got {types or 'none'}")
    ours = recipe.lerobot_args()
    recipe_values = {_flag_key(a): a.split("=", 1)[1] for a in ours}
    for arg in args:
        key = _flag_key(arg)
        if key in recipe_values and arg.split("=", 1)[1] != recipe_values[key]:
            logging.warning("--%s=%s overrides the shared diffusion recipe's %s",
                            key, arg.split("=", 1)[1], recipe_values[key])
    return [argv[0], *ours, *args]


def install() -> None:
    make_policy = lerobot_train.make_policy
    update_policy = lerobot_train.update_policy
    save_checkpoint = lerobot_train.save_checkpoint
    load_training_state = lerobot_train.load_training_state
    make_pre_post_processors = lerobot_train.make_pre_post_processors
    ema_cls = _ema_class()

    def make_policy_with_ema(cfg, ds_meta=None, **kwargs):
        if cfg.pretrained_path is None and cfg.crop_shape is None and ds_meta is not None:
            visual = [ft.shape for ft in dataset_to_policy_features(ds_meta.features).values()
                      if ft.type.name == "VISUAL"]
            if visual:
                cfg.crop_shape = crop_size(*(cfg.resize_shape or visual[0][1:]))
                logging.info("crop %s of %s images", cfg.crop_shape, cfg.resize_shape or visual[0][1:])
        _STATE["ds_meta"] = ds_meta
        policy = make_policy(cfg, ds_meta=ds_meta, **kwargs)
        ema = ema_cls(copy.deepcopy(policy), update_after_step=0, inv_gamma=recipe.EMA_INV_GAMMA,
                      power=recipe.EMA_POWER, min_value=recipe.EMA_MIN_DECAY,
                      max_value=recipe.EMA_MAX_DECAY)
        policy.push_model_to_hub = ema.averaged_model.push_model_to_hub
        _STATE.update(policy=policy, ema=ema)
        return policy

    def update_policy_with_ema(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
                               *args, **kwargs):
        out = update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
                            *args, **kwargs)
        _STATE["ema"].step(accelerator.unwrap_model(policy))
        return out

    def save_ema_checkpoint(checkpoint_dir, step, cfg, policy, *args, **kwargs):
        ema = _STATE["ema"]
        save_checkpoint(checkpoint_dir, step, cfg, ema.averaged_model, *args, **kwargs)
        state_dir = Path(checkpoint_dir) / TRAINING_STATE_DIR
        save_model(policy, str(state_dir / LIVE_WEIGHTS))
        (state_dir / EMA_STATE).write_text(json.dumps({"optimization_step": ema.optimization_step,
                                                       "decay": ema.decay}))

    def load_live_training_state(checkpoint_dir, optimizer, scheduler):
        step, optimizer, scheduler = load_training_state(checkpoint_dir, optimizer, scheduler)
        state_dir = Path(checkpoint_dir) / TRAINING_STATE_DIR
        if not (state_dir / LIVE_WEIGHTS).is_file():
            raise SystemExit(f"{checkpoint_dir} was not written by {Path(__file__).name}: it has no "
                             f"{TRAINING_STATE_DIR}/{LIVE_WEIGHTS}, so its EMA cannot be continued")
        # The policy was built from pretrained_model/, i.e. the EMA weights, which
        # the EMA's copy already holds.
        load_model(_STATE["policy"], str(state_dir / LIVE_WEIGHTS))
        _STATE["ema"].optimization_step = int(json.loads((state_dir / EMA_STATE).read_text())["optimization_step"])
        logging.info("resumed live weights and EMA at step %d", _STATE["ema"].optimization_step)
        return step, optimizer, scheduler

    def make_processors_with_ee_state(policy_cfg, pretrained_path=None, **kwargs):
        # A fresh run converts; a resumed one keeps what its checkpoint was trained on.
        ee = pretrained_path is None or uses_ee_state(pretrained_path)
        ds_meta = _STATE["ds_meta"]
        if ee:
            df, _ = load_frames(Path(ds_meta.root))
            # In place: lerobot-train hands this same dict to the normaliser.
            ds_meta.stats[OBS_STATE] = ee_state_stats(df[OBS_STATE].to_list())
            logging.info("observation.state: end-effector pose + gripper, stats over %d frames", len(df))
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg, pretrained_path=pretrained_path, **kwargs)
        if ee and pretrained_path is None:
            at = next(i for i, step in enumerate(preprocessor.steps)
                      if isinstance(step, NormalizerProcessorStep))
            preprocessor.steps.insert(at, JointStateToEEPose())
        return preprocessor, postprocessor

    lerobot_train.make_policy = make_policy_with_ema
    lerobot_train.make_pre_post_processors = make_processors_with_ee_state
    lerobot_train.update_policy = update_policy_with_ema
    lerobot_train.save_checkpoint = save_ema_checkpoint
    lerobot_train.load_training_state = load_live_training_state


if __name__ == "__main__":
    sys.argv = with_recipe_args(sys.argv)
    install()
    lerobot_train.main()
