#!/usr/bin/env python3
"""SAIL precision labels from gripper events, in place of its AWE + DBSCAN pass.

    python -m baselines.sail_bridge.gripper_labels ~/franka_data/baseline_prep/<dataset>/sail.hdf5

The paper gives two ways to find the steps to execute slowly (Sec. 4.3): DBSCAN
over AWE waypoints, which is what the repo's label_awe_trajectory_precision.py
does, and gripper events -- the gripper command opening or closing. On a
pick-and-place recording the DBSCAN labels can cover nearly everything (92% of
filter-10-6-bowl-plate at the 2.5 cm default), which leaves `--speed` nothing to
speed up. This labels a step precise when it lies within `--before` steps
before a gripper command change or `--after` steps after one, or while the
measured gripper is still moving. A change is the repo's own
dev_utils.GRIPPER_COMMAND_CHANGE_THRESHOLD.

Writes what label_awe_trajectory_precision.py writes -- `precisions` and
`<action-key>_with_precision` per demo -- so sail_bridge/train.py finds the
labels present and runs neither AWE pass; `data.attrs["precision_source"]`
records which pass made them. `train.py --relabel` puts the AWE labels back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import franka_config as fc  # noqa: E402

# dev_utils.GRIPPER_COMMAND_CHANGE_THRESHOLD: a command change bigger than this is an event.
_CHANGE = 0.2


def gripper_precision(command: np.ndarray, measured: np.ndarray, fps: float, before: int,
                      after: int, moving_per_s: float) -> np.ndarray:
    """1.0 on steps near a gripper command change or while the gripper moves, else 0.0.

    `command` and `measured` are normalised gripper positions, (T,).
    """
    n = len(command)
    label = np.zeros(n, dtype=bool)
    for c in np.flatnonzero(np.abs(np.diff(command)) > _CHANGE) + 1:   # first step of the new command
        label[max(0, c - before):min(n, c + after)] = True
    label[1:] |= np.abs(np.diff(measured)) * fps > moving_per_s
    return label.astype(np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hdf5", type=Path, help="sail.hdf5 from scripts/prepare_baseline_datasets.py")
    p.add_argument("--action-key", default="absolute_actions",
                   help="the action the label column is appended to")
    p.add_argument("--before", type=int, default=8,
                   help="steps before a gripper command change that are precise: the approach")
    p.add_argument("--after", type=int, default=4,
                   help="steps from a gripper command change on that are precise, at least")
    p.add_argument("--moving-per-s", type=float, default=0.05,
                   help="measured gripper speed, in normalised range per second, above which a "
                        "step is precise however long the fingers take")
    p.add_argument("--dry-run", action="store_true", help="report the fractions, write nothing")
    args = p.parse_args()

    fps = float(fc.control_fps())
    fractions = []
    with h5py.File(args.hdf5, "r" if args.dry_run else "r+") as f:
        demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
        for name in demos:
            d = f["data"][name]
            actions = d[args.action_key][()]
            label = gripper_precision(actions[:, 6], d["obs/robot0_gripper_qpos"][:, 0], fps,
                                      args.before, args.after, args.moving_per_s)
            fractions.append(label.mean())
            print(f"  {name}: {100 * label.mean():5.1f}% of {len(label)} steps labelled precise")
            if args.dry_run:
                continue
            for key in ("precisions", f"{args.action_key}_with_precision"):
                if key in d:
                    del d[key]
            d.create_dataset("precisions", data=label)
            d.create_dataset(f"{args.action_key}_with_precision",
                             data=np.hstack([actions, label[:, None]]).astype(actions.dtype))
        if not args.dry_run:
            f["data"].attrs["precision_source"] = "gripper"
            f["data"].attrs["precision_params"] = json.dumps(
                {"before": args.before, "after": args.after, "moving_per_s": args.moving_per_s,
                 "change": _CHANGE, "fps": fps})
    print(f"{len(demos)} demos, {100 * np.mean(fractions):.0f}% of steps labelled precise"
          + (" (dry run, nothing written)" if args.dry_run else f"; written to {args.hdf5}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
