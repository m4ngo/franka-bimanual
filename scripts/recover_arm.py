#!/usr/bin/env python3
"""Reconnect and call recover_from_errors on the right arm (stop teleop first)."""

from __future__ import annotations

import argparse
import sys

from lerobot_robot_bimanual_franka import ControlMode, SingleArmFranka, SingleArmFrankaConfig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-ip", default="192.168.3.10")
    p.add_argument("--robot-ip", default="192.168.201.10")
    p.add_argument("--port", type=int, default=18812)
    args = p.parse_args()

    cfg = SingleArmFrankaConfig(
        r_server_ip=args.server_ip,
        r_robot_ip=args.robot_ip,
        r_gripper_ip=args.robot_ip,
        r_port=args.port,
        control_mode=ControlMode.JOINT_POS,
    )
    robot = SingleArmFranka(cfg)
    try:
        robot.robot_manager.add_robot("r", args.server_ip, args.robot_ip, args.port, use_cartesian=False)
        robot.robot_manager.drivers["r"].robot.recover_from_errors()
        print("Arm recovered.")
        return 0
    except Exception as e:
        print(f"Recovery failed: {e}", file=sys.stderr)
        return 1
    finally:
        robot.robot_manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
