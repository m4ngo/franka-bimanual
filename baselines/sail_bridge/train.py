#!/usr/bin/env python3
"""Train SAIL on a converted recording: precision labels, config, robomimic.

    python -m baselines.sail_bridge.train ~/franka_data/baseline_prep/<dataset>/sail.hdf5

Upstream's pipeline (baselines/SAIL.md) is five passes; the converter replaced
pass 1, and this runs passes 2-4 in the SAIL venv:

    2. AWE waypoints            save_awe_waypoint_concurrent.py   -> waypoints_dp
    3. precision labels         label_awe_trajectory_precision.py -> <key>_with_precision
    4. robomimic/scripts/train.py with a config generated from
       exps/templates/diffusion_policy_SAIL.json, through robomimic_train.py
       (which stops the dataset fetching 16x the images the model reads)

Passes 2 and 3 edit the HDF5 in place and are skipped when their keys already
exist (--relabel forces them). The generated config differs from the template
only where our data does: camera keys, the crop size derived from the file's
image size, the dataset path, the output directory, and the epoch budget.

Checkpoints land in <output-dir>/<name>/<timestamp>/models/model_epoch_N.pth
(the timestamp is robomimic's own), which is what sail_rollout.sh --ckpt takes.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import h5py  # noqa: E402

from baselines.interpreters import python_for  # noqa: E402
from baselines.train_common import crop_size, policies_dir, stream  # noqa: E402

logger = logging.getLogger("baselines.sail.train")

SAIL_ROOT = _REPO_ROOT / "baselines" / "sail"
TEMPLATE = SAIL_ROOT / "robomimic" / "exps" / "templates" / "diffusion_policy_SAIL.json"
AWE_SCRIPT = SAIL_ROOT / "robomimic" / "SAIL" / "precision_processing" / "save_awe_waypoint_concurrent.py"
LABEL_SCRIPT = SAIL_ROOT / "robomimic" / "SAIL" / "precision_processing" / "label_awe_trajectory_precision.py"
# Not upstream's robomimic/scripts/train.py directly: the launcher cuts the
# per-sample observation window to what the policy reads, then calls it.
TRAIN_SCRIPT = Path(__file__).resolve().parent / "robomimic_train.py"

DEFAULT_ACTION_KEY = "absolute_actions_with_precision"
# The pose layout every converter writes, plus the label column.
POSE_DIM = 7


@dataclass
class Info:
    path: Path
    source_repo_id: str | None
    demos: list[str]
    image_keys: list[str]
    image_hw: tuple[int, int] | None
    action_widths: dict[str, int]
    has_waypoints: bool
    precision_keys: list[str] = field(default_factory=list)


def inspect_hdf5(path: Path) -> Info:
    with h5py.File(path, "r") as f:
        repo_id = f.attrs.get("source_repo_id")
        if isinstance(repo_id, bytes):
            repo_id = repo_id.decode()
        demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
        if not demos:
            raise SystemExit(f"{path} holds no demos")
        first = f["data"][demos[0]]
        image_keys = sorted(k for k in first["obs"].keys() if k.endswith("_image"))
        image_hw = None
        if image_keys:
            shape = first["obs"][image_keys[0]].shape
            image_hw = (int(shape[1]), int(shape[2]))
        widths = {k: int(first[k].shape[1]) for k in first.keys()
                  if k != "obs" and first[k].ndim == 2}
        has_waypoints = all("waypoints_dp" in f["data"][d] for d in demos)
        precision_keys = sorted(k for k in widths if k.endswith("_with_precision"))
    return Info(path, str(repo_id) if repo_id else None, demos, image_keys, image_hw,
                widths, has_waypoints, precision_keys)


def precision_summary(path: Path) -> list[tuple[str, float, int]]:
    """(demo, fraction of steps labelled precise, steps) per demo."""
    out = []
    with h5py.File(path, "r") as f:
        for demo in sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1])):
            p = f["data"][demo].get("precisions")
            if p is None:
                continue
            arr = p[()]
            out.append((demo, float((arr > 0.5).mean()) if len(arr) else 0.0, len(arr)))
    return out


def build_config(info: Info, args, output_dir: Path) -> dict:
    cfg = json.loads(TEMPLATE.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["name"] = args.name
    cfg["experiment"]["logging"]["log_wandb"] = bool(args.wandb)
    if args.wandb_project:
        cfg["experiment"]["logging"]["wandb_proj_name"] = args.wandb_project
    cfg["experiment"]["logging"]["log_tb"] = not args.no_tb
    cfg["experiment"]["epoch_every_n_steps"] = int(args.epoch_every_n_steps)
    cfg["experiment"]["save"]["every_n_epochs"] = int(args.save_every)
    tr = cfg["train"]
    tr["data"] = str(info.path.resolve())
    tr["output_dir"] = str(output_dir)
    tr["dataset_keys"] = [args.action_key]
    tr["action_keys"] = [args.action_key]
    tr["num_epochs"] = int(args.epochs)
    # robomimic saves only every every_n_epochs or at listed epochs; a budget
    # off that grid would otherwise end on an unsaved epoch.
    save = cfg["experiment"]["save"]
    save["epochs"] = sorted(set(save["epochs"]) | {int(args.epochs)})
    if args.resume is not None:
        # robomimic rebuilds the model and EMA from the checkpoint and carries the
        # epoch count on; the optimizer starts fresh (the checkpoint holds none)
        # at the same constant learning rate.
        tr["load_ckpt"] = str(args.resume.resolve())
        tr["epoch_start"] = checkpoint_epoch(args.resume) + 1
    tr["batch_size"] = int(args.batch_size)
    tr["seed"] = int(args.seed)
    tr["num_data_workers"] = int(args.data_workers)
    if args.action_key not in tr["action_config"]:
        tr["action_config"][args.action_key] = {"normalization": "min_max"}
    obs = cfg["observation"]["modalities"]["obs"]
    obs["rgb"] = list(info.image_keys)
    rgb = cfg["observation"]["encoder"]["rgb"]["obs_randomizer_kwargs"]
    if info.image_hw is not None:
        rgb["crop_height"], rgb["crop_width"] = crop_size(*info.image_hw)
    else:
        cfg["observation"]["encoder"]["rgb"]["obs_randomizer_class"] = None
    return cfg


_CKPT_EPOCH = re.compile(r"model_epoch_(\d+)\.pth$")


def checkpoint_epoch(path: Path) -> int:
    m = _CKPT_EPOCH.search(path.name)
    if m is None:
        raise SystemExit(f"{path} is not a robomimic model_epoch_N.pth checkpoint")
    return int(m.group(1))


def newest_checkpoint(output_dir: Path, name: str) -> Path | None:
    ckpts = glob.glob(str(output_dir / name / "*" / "models" / "model_epoch_*.pth"))
    if not ckpts:
        return None
    return Path(max(ckpts, key=os.path.getmtime))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hdf5", type=Path, help="sail.hdf5 from scripts/prepare_baseline_datasets.py")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default ~/franka_data/policies/<train-dataset>")
    p.add_argument("--name", default="sail", help="experiment name; a subdirectory of --output-dir")
    p.add_argument("--resume", type=Path, default=None, metavar="MODEL_EPOCH_N.PTH",
                   help="continue from this checkpoint: epochs N+1 .. the epoch count, into a new "
                        "<output-dir>/<name>/<timestamp>/ beside it")
    p.add_argument("--action-key", default=DEFAULT_ACTION_KEY,
                   help="what the policy predicts; upstream trains on the reached pose "
                        "plus its precision label")
    p.add_argument("--epochs", type=int, default=None, help="default: the template's")
    p.add_argument("--steps", type=int, default=None,
                   help="gradient-step budget in place of --epochs: epochs = "
                        "ceil(steps / epoch_every_n_steps), the same count lerobot-train's "
                        "--steps means, so the three policies can be trained equally long")
    p.add_argument("--epoch-every-n-steps", type=int, default=None,
                   help="gradient steps per epoch; default: the template's")
    p.add_argument("--save-every", "--checkpoint-every", dest="save_every", type=int, default=None,
                   help="checkpoint every N epochs; default: the template's. The last epoch is "
                        "always saved. --checkpoint-every is B-Spline's name for it, so one "
                        "--extra can carry it to both")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-workers", type=int, default=None)
    p.add_argument("--no-tb", action="store_true", help="no tensorboard logging")
    p.add_argument("--wandb", action="store_true", help="log online to wandb; default off")
    p.add_argument("--wandb-project", default=None, help="default: the template's wandb_proj_name")
    p.add_argument("--wandb-entity", default=None,
                   help="robomimic insists on one; default $WANDB_ENTITY, else the login's default")
    p.add_argument("--wandb-name", default=None, help="run name on wandb; default: --name")
    p.add_argument("--err-threshold", type=float, default=0.005,
                   help="AWE reconstruction error (m) that picks the waypoints")
    p.add_argument("--num-workers", type=int, default=6, help="AWE worker processes")
    p.add_argument("--relabel", action="store_true",
                   help="rerun the waypoint and precision passes even if their keys exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands and the generated config, run nothing")
    p.add_argument("--python", default=None, help="interpreter of the SAIL env; default: resolved")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.hdf5.is_file():
        p.error(f"{args.hdf5} does not exist")
    for path in (TEMPLATE, AWE_SCRIPT, LABEL_SCRIPT, TRAIN_SCRIPT):
        if not path.is_file():
            p.error(f"{path} is missing; git submodule update --init baselines/sail")

    if args.steps is not None and args.epochs is not None:
        p.error("--steps and --epochs are two ways to say the same thing; pass one")
    template = json.loads(TEMPLATE.read_text())
    for attr, key in (("epochs", ("train", "num_epochs")),
                      ("epoch_every_n_steps", ("experiment", "epoch_every_n_steps")),
                      ("save_every", ("experiment", "save", "every_n_epochs")),
                      ("batch_size", ("train", "batch_size")),
                      ("seed", ("train", "seed")),
                      ("data_workers", ("train", "num_data_workers"))):
        if getattr(args, attr) is None:
            node = template
            for k in key:
                node = node[k]
            setattr(args, attr, node)

    if args.steps is not None:
        # robomimic counts epochs; an epoch is epoch_every_n_steps gradient steps.
        args.epochs = -(-int(args.steps) // int(args.epoch_every_n_steps))
        logger.info("--steps %d -> %d epochs of %d steps (%d steps)", args.steps, args.epochs,
                    args.epoch_every_n_steps, args.epochs * int(args.epoch_every_n_steps))

    info = inspect_hdf5(args.hdf5)
    output_dir = policies_dir(info.source_repo_id, args.output_dir)
    python = [args.python] if args.python else python_for("sail")

    logger.info("dataset %s: %d demos, cameras %s at %s, trained on %r",
                args.hdf5, len(info.demos), info.image_keys, info.image_hw, info.source_repo_id)
    if "absolute_actions" not in info.action_widths:
        p.error("no absolute_actions in the file; this is not a SAIL conversion")

    base_key = args.action_key.removesuffix("_with_precision")
    wants_label = args.action_key.endswith("_with_precision")
    if base_key not in info.action_widths:
        p.error(f"action key {base_key!r} is not in the file; it has {sorted(info.action_widths)}")

    steps: list[tuple[str, list[str]]] = []
    if wants_label and (args.relabel or not info.has_waypoints):
        steps.append(("AWE waypoints", [*python, str(AWE_SCRIPT), "--dataset", str(args.hdf5.resolve()),
                                        "--err_threshold", str(args.err_threshold),
                                        "--num_workers", str(args.num_workers)]))
    if wants_label and (args.relabel or args.action_key not in info.precision_keys):
        steps.append(("precision labels", [*python, str(LABEL_SCRIPT), "--dataset",
                                           str(args.hdf5.resolve()), "--action_key", base_key]))
    for label, cmd in steps:
        if args.dry_run:
            logger.info("[dry-run] %s: %s", label, " ".join(cmd))
            continue
        logger.info("=== %s ===", label)
        rc, _ = stream(cmd, SAIL_ROOT)
        if rc != 0:
            logger.error("%s failed (exit %d)", label, rc)
            return rc

    if wants_label and not args.dry_run:
        info = inspect_hdf5(args.hdf5)
        if args.action_key not in info.precision_keys:
            logger.error("%s still missing after labelling", args.action_key)
            return 1
        for demo, frac, n in precision_summary(args.hdf5):
            logger.info("  %s: %5.1f%% of %d steps labelled precise", demo, 100 * frac, n)
        width = info.action_widths[args.action_key]
        if width != POSE_DIM + 1:
            p.error(f"{args.action_key} is {width} wide, expected {POSE_DIM + 1}; the rollout "
                    "strips the last column as the label and would amputate the gripper")

    cfg = build_config(info, args, output_dir)
    config_path = output_dir / f"{args.name}-config.json"
    train_cmd = [*python, str(TRAIN_SCRIPT), "--config", str(config_path)]
    if args.dry_run:
        logger.info("[dry-run] config -> %s\n%s", config_path, json.dumps(cfg, indent=2))
        logger.info("[dry-run] train: %s", " ".join(train_cmd))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=4))
    logger.info("config written to %s", config_path)
    if args.resume is not None:
        if checkpoint_epoch(args.resume) >= int(args.epochs):
            logger.error("%s already holds epoch %d of %d; nothing to resume",
                         args.resume, checkpoint_epoch(args.resume), args.epochs)
            return 1
        logger.info("resuming at epoch %d of %d from %s", checkpoint_epoch(args.resume) + 1, args.epochs, args.resume)
    logger.info("=== training (%d epochs x %d steps, batch %d) ===",
                args.epochs, args.epoch_every_n_steps, args.batch_size)
    # robomimic_train.py reads these instead of robomimic's gitignored macros_private.py.
    env = dict(os.environ)
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_name:
        env["WANDB_NAME"] = args.wandb_name
    t0 = time.time()
    # train.py catches every exception, prints it, and exits 0. When
    # <output-dir>/<name> already exists it asks whether to DELETE it -- every
    # earlier run of this task -- and "n" is what makes it add a new timestamped
    # subdirectory instead, which is the layout wanted here.
    rc, failed = stream(train_cmd, SAIL_ROOT, env=env, watch="run failed with error", stdin_text="n\n")
    ckpt = newest_checkpoint(output_dir, args.name)
    if rc != 0 or failed or ckpt is None or ckpt.stat().st_mtime < t0:
        logger.error("training did not produce a checkpoint (exit %d, failed=%s)", rc, failed)
        return 1
    logger.info("trained in %.0fs; newest checkpoint:\n  %s", time.time() - t0, ckpt)
    logger.info("roll it out with:\n  ./scripts/sail_rollout.sh --start-server --ckpt %s "
                "--guide-config baselines/sail/robomimic/SAIL/guide_template/base_cfg_weight_1.json "
                "--rig=single_arm_right", ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
