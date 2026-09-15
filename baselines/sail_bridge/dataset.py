#!/usr/bin/env python3
"""Convert a LeRobot EE_POS recording into SAIL's robomimic HDF5 format.

    python -m baselines.sail_bridge.dataset ~/franka_data/my-recording --out my.hdf5

Field list and why the fields are what they are: baselines/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from baselines.common import (  # noqa: E402
    NUM_JOINTS,
    parse_image_size,
    pos_rotvec_gripper,
    reached_and_commanded_poses,
    run_conversion,
)
from lerobot_robot_bimanual_franka.ee_goals import OSCGoalBuilder  # noqa: E402
from lerobot_robot_bimanual_franka.lerobot_source import GRIPPER  # noqa: E402


def convert_episode(actions, states, arm, screen):
    qpos = states[:, :NUM_JOINTS]
    gripper_obs = states[:, NUM_JOINTS].astype("float32")
    gripper_act = actions[:, GRIPPER]

    reached_pos, reached_quat, commanded_pos, commanded_quat = reached_and_commanded_poses(
        qpos, actions, arm, screen
    )
    # delta_from_absolute already returns (dpos, drotvec) -- pos_rotvec_gripper
    # expects a quaternion to convert, so it doesn't apply here; concatenate directly.
    dpos, drot = OSCGoalBuilder.delta_from_absolute(
        commanded_pos, commanded_quat, reached_pos, reached_quat
    )
    delta_actions = np.concatenate([dpos, drot, gripper_act[:, None]], axis=1).astype("float32")

    return {
        "obs/robot0_eef_pos": reached_pos.astype("float32"),
        "obs/robot0_eef_quat": reached_quat.astype("float32"),
        "obs/robot0_joint_pos": qpos.astype("float32"),
        "obs/robot0_gripper_qpos": gripper_obs[:, None],
        "actions": delta_actions,
        "absolute_actions": pos_rotvec_gripper(reached_pos, reached_quat, gripper_act),
        "commanded_absolute_actions": pos_rotvec_gripper(commanded_pos, commanded_quat, gripper_act),
    }


# robomimic's train.py loads env_args before it checks whether rollouts are on.
# type 6 is EnvType.REAL_TYPE, so is_robosuite_env() stays False.
_ENV_ARGS = json.dumps({"env_name": "bimanual_franka", "type": 6, "env_kwargs": {}})


def convert(source: str, out: Path, **kwargs) -> int:
    return run_conversion(
        source, out, convert_episode, "baselines/sail_bridge/dataset.py",
        root_attrs={"env_args": _ENV_ARGS}, **kwargs,
    )


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
