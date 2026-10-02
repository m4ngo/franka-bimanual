#!/usr/bin/env python3
"""robomimic's train.py, with the observation window cut to what the policy reads.

    .venv-sail/bin/python baselines/sail_bridge/robomimic_train.py --config <json>
    .venv-sail/bin/python baselines/sail_bridge/robomimic_train.py --config <json> --check

`SequenceDataset.get_item` fetches `train.seq_length` frames (32, the prediction
horizon) of EVERY observation key per sample, and
`DiffusionPolicyUNet.process_batch_for_training` then keeps the first
`observation_horizon` (2) of them. With three 224x224 cameras that is 16x the
image data the model reads, read out of the HDF5 for every sample: measured at
5 s per gradient step on this workstation, with the GPU idle.

This launcher patches the obs fetch to `observation_horizon` frames and hands
off to the stock training entry point. The batch the model receives is
identical -- --check proves it by comparing samples from a patched and an
unpatched dataset -- so a checkpoint trained this way is exactly what upstream
would have produced. The submodule itself is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import robomimic.utils.dataset as dataset_module
from robomimic.config import config_factory

_ORIGINAL = dataset_module.SequenceDataset.get_obs_sequence_from_demo


def _obs_window(observation_horizon: int, frame_stack: int) -> int:
    """`seq_length` such that (frame_stack - 1) + seq_length == observation_horizon."""
    return max(1, int(observation_horizon) - (int(frame_stack) - 1))


def install(obs_seq_length: int) -> None:
    def patched(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        return _ORIGINAL(self, demo_id, index_in_demo, keys,
                         num_frames_to_stack=num_frames_to_stack,
                         seq_length=min(int(seq_length), obs_seq_length), prefix=prefix)
    dataset_module.SequenceDataset.get_obs_sequence_from_demo = patched


def uninstall() -> None:
    dataset_module.SequenceDataset.get_obs_sequence_from_demo = _ORIGINAL


def configure_wandb(config) -> None:
    """robomimic asserts a wandb entity and reads it from a gitignored
    macros_private.py inside the submodule; take it from $WANDB_ENTITY or the
    login's default instead, and let $WANDB_NAME name the run (robomimic passes
    experiment.name, which is also the output subdirectory)."""
    if not config.experiment.logging.log_wandb:
        return
    import robomimic.macros as macros
    import wandb

    entity = os.environ.get("WANDB_ENTITY") or macros.WANDB_ENTITY or wandb.Api().default_entity
    if not entity:
        raise SystemExit("wandb is on but no entity is known: set $WANDB_ENTITY or `wandb login`")
    macros.WANDB_ENTITY = entity
    name = os.environ.get("WANDB_NAME")
    if name:
        original = wandb.init

        def init(*args, **kwargs):
            kwargs["name"] = name
            return original(*args, **kwargs)

        wandb.init = init
    print(f"wandb: entity {entity}, project {config.experiment.logging.wandb_proj_name}, "
          f"run {name or config.experiment.name}")


def load_config(path: str):
    ext_cfg = json.load(open(path))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    return config


def check(config, n: int = 8) -> int:
    """Patched vs unpatched samples must agree on the frames the policy keeps."""
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.train_utils as TrainUtils

    ObsUtils.initialize_obs_utils_with_config(config)
    to = int(config.algo.horizon.observation_horizon)
    window = _obs_window(to, config.train.frame_stack)
    data_path = config.train.data if isinstance(config.train.data, str) else config.train.data[0]["path"]
    shape_meta = TrainUtils.FileUtils.get_shape_metadata_from_dataset(
        dataset_path=data_path, action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys, ds_format=config.train.data_format, verbose=False)
    # Neither dataset may cache getitem results, or the comparison is of caches.
    with config.values_unlocked():
        config.train.hdf5_cache_mode = "low_dim"

    def build():
        return TrainUtils.dataset_factory(config, obs_keys=shape_meta["all_obs_keys"])

    uninstall()
    full = build()
    install(window)
    cut = build()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(full), size=min(n, len(full)))
    bad = 0
    for i in idx:
        a, b = full.get_item(int(i)), cut.get_item(int(i))
        for k in a["obs"]:
            if not np.array_equal(np.asarray(a["obs"][k])[:to], np.asarray(b["obs"][k])):
                print(f"MISMATCH sample {i} key {k}: {np.asarray(a['obs'][k]).shape} vs {np.asarray(b['obs'][k]).shape}")
                bad += 1
        for k in ("actions",):
            if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
                print(f"MISMATCH sample {i} {k}")
                bad += 1
    print(f"obs window {window} (+{config.train.frame_stack - 1} stacked) instead of "
          f"{config.train.seq_length}; {len(idx)} samples compared, {bad} mismatches")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--check", action="store_true",
                   help="compare patched and unpatched samples instead of training")
    p.add_argument("--no-obs-window", action="store_true",
                   help="run stock robomimic, fetching the full sequence per sample")
    args, passthrough = p.parse_known_args()

    config = load_config(args.config)
    if args.check:
        return check(config)
    configure_wandb(config)
    if not args.no_obs_window:
        window = _obs_window(config.algo.horizon.observation_horizon, config.train.frame_stack)
        install(window)
        print(f"obs window: {window} frame(s) + {config.train.frame_stack - 1} stacked per sample "
              f"(stock robomimic would fetch {config.train.seq_length})")

    from robomimic.scripts import train as robomimic_train
    sys.argv = [sys.argv[0], "--config", args.config, *passthrough]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--auto-remove-exp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    robomimic_train.main(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
