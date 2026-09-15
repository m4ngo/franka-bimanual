#!/usr/bin/env python3
"""Convert a LeRobot EE_POS recording into B-Spline Policy's robomimic HDF5 format.

    python -m baselines.bspline_bridge.dataset ~/franka_data/my-recording --out my.hdf5

Field list and why the fields are what they are: baselines/README.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from baselines.common import (  # noqa: E402
    NUM_JOINTS,
    parse_image_size,
    pos_rotvec_gripper,
    reached_and_commanded_poses,
    run_conversion,
)
from lerobot_robot_bimanual_franka.lerobot_source import GRIPPER  # noqa: E402


def convert_episode(actions, states, arm, screen):
    qpos = states[:, :NUM_JOINTS]
    gripper_obs = states[:, NUM_JOINTS].astype("float32")
    gripper_act = actions[:, GRIPPER]

    reached_pos, reached_quat, commanded_pos, commanded_quat = reached_and_commanded_poses(
        qpos, actions, arm, screen
    )

    return {
        # obs is what the robot itself reports (reached); actions is the goal
        # it was sent (commanded) -- using the same pose for both would make
        # the policy's target identical to its own input.
        "obs/arm_pos": reached_pos.astype("float32"),
        "obs/arm_quat": reached_quat.astype("float32"),
        "obs/gripper_pos": gripper_obs[:, None],
        "actions": pos_rotvec_gripper(commanded_pos, commanded_quat, gripper_act),
    }


def convert(source: str, out: Path, **kwargs) -> int:
    return run_conversion(source, out, convert_episode, "baselines/bspline_bridge/dataset.py", **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset root, or a repo id resolved under ~/franka_data")
    p.add_argument("--out", required=True, type=Path, help="output HDF5 path")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices")
    p.add_argument("--trim-start", default="auto")
    p.add_argument("--max-trim", type=int, default=5)
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--no-images", action="store_true", help="skip camera frames (schema/shape tests)")
    p.add_argument("--source-repo-id", default=None,
                   help="dataset's HuggingFace id, stamped on the file so a rollout "
                        "can name its output directory after the task. Defaults to "
                        "the `dataset` argument, which is normally already that id")
    p.add_argument("--image-size", default=None,
                   help="WxH to resize camera frames to, e.g. 84x84. The file's image "
                        "size is what the policy trains and rolls out at")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    episodes = {int(x) for x in args.episodes.split(",")} if args.episodes else None
    return convert(
        args.dataset, args.out, episodes=episodes, trim_start=args.trim_start,
        max_trim=args.max_trim, min_steps=args.min_steps, include_images=not args.no_images,
        source_repo_id=args.source_repo_id,
        image_size=parse_image_size(args.image_size),
    )


if __name__ == "__main__":
    raise SystemExit(main())
