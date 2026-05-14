"""Save / apply a named bimanual-Franka home pose.

Poses live in `~/franka_ws/home_poses/<name>.json`. The format is trivial:

  {
    "l_q":     [q1, q2, q3, q4, q5, q6, q7],
    "r_q":     [q1, q2, q3, q4, q5, q6, q7],
    "gripper": 1.0
  }

Subcommands:

  save NAME    Connect, read the current joint state, write to
               home_poses/NAME.json. Use after guiding the arms by hand in
               Program mode (then switch back to Execution + re-enable FCI
               before running, otherwise the connect call will hang).

  apply NAME   Load home_poses/NAME.json and drive the arms there via
               `BimanualFranka.home()`. Useful for sanity-checking a saved
               pose before committing to a record/rollout run with it.

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

import numpy as np

from lerobot.robots import make_robot_from_config

from lerobot_robot_bimanual_franka import BimanualFranka, BimanualFrankaConfig

POSES_DIR = Path(__file__).resolve().parent.parent / "home_poses"

_RIG = dict(
    l_server_ip="192.168.3.11", l_robot_ip="192.168.200.2",
    l_gripper_ip="192.168.2.21", l_port=18813,
    r_server_ip="192.168.3.10", r_robot_ip="192.168.201.10",
    r_gripper_ip="192.168.2.20", r_port=18812,
)


def _make_robot() -> BimanualFranka:
    cfg = BimanualFrankaConfig(**_RIG, use_ee_pos=False)
    return make_robot_from_config(cfg)


def _path_for(name: str) -> Path:
    return POSES_DIR / f"{name}.json"


def cmd_save(args: argparse.Namespace) -> None:
    path = _path_for(args.name)
    robot = _make_robot()
    robot.connect()
    try:
        kin = robot.robot_manager.current_kinematic_state_batch(list(robot.active_arms))
        pose = {
            "l_q": [float(x) for x in kin["l"][0]],
            "r_q": [float(x) for x in kin["r"][0]],
            "gripper": float(args.gripper),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pose, indent=2) + "\n")
        print(json.dumps(pose, indent=2))
        print(f"\nSaved to {path}")
    finally:
        robot.disconnect()


def cmd_apply(args: argparse.Namespace) -> None:
    path = _path_for(args.name)
    pose = json.loads(path.read_text())
    robot = _make_robot()
    robot.connect()
    try:
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
    sp_save.add_argument("--gripper", type=float, default=1.0,
                         help="Normalized gripper target to record (0=closed, 1=open). Default 1.0.")
    sp_save.set_defaults(func=cmd_save)

    sp_apply = sub.add_parser("apply", help="Drive the arms to a saved pose")
    sp_apply.add_argument("name", help="Pose name")
    sp_apply.add_argument("--max-time-s", type=float, default=5.0)
    sp_apply.add_argument("--tol-rad", type=float, default=0.05)
    sp_apply.set_defaults(func=cmd_apply)

    sp_list = sub.add_parser("list", help="List saved pose names")
    sp_list.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
