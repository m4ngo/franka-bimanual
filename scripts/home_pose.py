"""Save / apply a named Franka home pose.

Poses live in the directory named by `arms.home_poses.dir` in config/arms.yaml
(default `~/franka_ws/home_poses/<name>.json`). The format is trivial:

    Bimanual:
        {
            "l_q":     [q1, q2, q3, q4, q5, q6, q7],
            "r_q":     [q1, q2, q3, q4, q5, q6, q7],
            "gripper": 1.0
        }

    Single-arm:
        {
            "r_q":     [q1, q2, q3, q4, q5, q6, q7],
            "gripper": 1.0
        }

Subcommands:

    save NAME    Connect, read the current joint state, write to
                             home_poses/NAME.json. Use after guiding the arm(s) by hand in
                             Program mode (then switch back to Execution + re-enable FCI
                             before running, otherwise the connect call will hang).

    apply NAME   Load home_poses/NAME.json and drive the arm(s) there via
                             `BimanualFranka.home()` or `SingleArmFranka.home()`.
                             Useful for sanity-checking a saved pose before committing to
                             a record/rollout run with it.

  list         Print the saved pose names.

Usage:

Saving new home pose (i.e. after guiding arms in Program mode):
$ python scripts/home_pose.py save <home_pose_name>

Applying saved home pose (i.e. testing/resetting):
$ python scripts/home_pose.py apply <home_pose_name>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import franka_config as fc
import numpy as np

from lerobot.robots import make_robot_from_config

from lerobot_robot_bimanual_franka import (
    BimanualFranka,
    BimanualFrankaConfig,
    SingleArmFranka,
    SingleArmFrankaConfig,
)

POSES_DIR = fc.home_poses_dir()


def _make_robot() -> BimanualFranka:
    """Bimanual follower; addressing comes from the `bimanual_franka` profile."""
    return make_robot_from_config(BimanualFrankaConfig(control_mode="JOINT_POS"))


def _make_single_arm_robot() -> SingleArmFranka:
    """Single-arm follower; addressing comes from the `single_arm_franka` profile."""
    return make_robot_from_config(SingleArmFrankaConfig(control_mode="JOINT_POS"))


def _path_for(name: str) -> Path:
    return POSES_DIR / f"{name}.json"


def _is_single_arm(args: argparse.Namespace) -> bool:
    return bool(args.single_arm)


def cmd_save(args: argparse.Namespace) -> None:
    path = _path_for(args.name)
    robot = _make_single_arm_robot() if _is_single_arm(args) else _make_robot()
    robot.connect()
    try:
        kin = robot.robot_manager.current_kinematic_state_batch(list(robot.active_arms))
        pose = {"gripper": float(args.gripper)}
        if _is_single_arm(args):
            pose["r_q"] = [float(x) for x in kin["r"][0]]
        else:
            pose["l_q"] = [float(x) for x in kin["l"][0]]
            pose["r_q"] = [float(x) for x in kin["r"][0]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pose, indent=2) + "\n")
        print(json.dumps(pose, indent=2))
        print(f"\nSaved to {path}")
    finally:
        robot.disconnect()


def cmd_apply(args: argparse.Namespace) -> None:
    path = _path_for(args.name)
    pose = json.loads(path.read_text())
    single_arm = _is_single_arm(args) or ("l_q" not in pose and "r_q" in pose)
    robot = _make_single_arm_robot() if single_arm else _make_robot()
    robot.connect()
    try:
        if single_arm:
            ok = robot.home(
                home_q_left=None,
                home_q_right=np.asarray(pose["r_q"], dtype=np.float64),
                gripper_norm=float(pose.get("gripper", 1.0)),
                max_time_s=args.max_time_s,
                tol_rad=args.tol_rad,
            )
        else:
            ok = robot.home(
                home_q_left=np.asarray(pose["l_q"], dtype=np.float64),
                home_q_right=np.asarray(pose["r_q"], dtype=np.float64),
                gripper_norm=float(pose.get("gripper", 1.0)),
                max_time_s=args.max_time_s,
                tol_rad=args.tol_rad,
            )
        print("home(): converged" if ok else "home(): timed out before reaching tolerance")
    finally:
        robot.disconnect()


def cmd_list(_: argparse.Namespace) -> None:
    if not POSES_DIR.exists():
        print(f"(no poses saved yet; {POSES_DIR} doesn't exist)")
        return
    names = sorted(p.stem for p in POSES_DIR.glob("*.json"))
    if not names:
        print(f"(no poses in {POSES_DIR})")
        return
    for n in names:
        print(n)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_save = sub.add_parser("save", help="Read and save the current joint pose")
    sp_save.add_argument("name", help="Pose name (stored as home_poses/NAME.json)")
    sp_save.add_argument("--single-arm", action="store_true",
                         help="Save a right-arm-only pose format (r_q + gripper).")
    sp_save.add_argument("--gripper", type=float, default=fc.control("homing.gripper_norm"),
                         help="Normalized gripper target to record (0=closed, 1=open). Default 1.0.")
    sp_save.set_defaults(func=cmd_save)

    sp_apply = sub.add_parser("apply", help="Drive the arms to a saved pose")
    sp_apply.add_argument("name", help="Pose name")
    sp_apply.add_argument("--single-arm", action="store_true",
                         help="Apply the pose as a right-arm-only home pose.")
    sp_apply.add_argument("--max-time-s", type=float, default=fc.control("homing.max_time_s"))
    sp_apply.add_argument("--tol-rad", type=float, default=fc.control("homing.tol_rad"))
    sp_apply.set_defaults(func=cmd_apply)

    sp_list = sub.add_parser("list", help="List saved pose names")
    sp_list.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
