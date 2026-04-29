"""Print processed joint values from both GELLO leaders, side-by-side.

Use this to verify which serial port maps to which arm before running
``lerobot-teleoperate --teleop.type=bimanual_gello``. Position each GELLO at a
known pose (e.g. the calibration home pose) and check that the values look as
expected; if they don't, your USB port assignment likely swapped between runs.

Example:

    python scripts/diagnose_bimanual_gello.py \
        --portl /dev/ttyUSB0 --portr /dev/ttyUSB1 --id gello_teleop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

from lerobot_teleoperator_gello import (
    BimanualGelloConfig,
    GelloLeaderFields,
)
from lerobot_teleoperator_gello.lerobot_teleoperator_gello.bimanual_gello import BimanualGello


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump per-arm GELLO joint values")
    parser.add_argument("--portl", default="/dev/ttyUSB0", help="Left GELLO serial port")
    parser.add_argument("--portr", default="/dev/ttyUSB1", help="Right GELLO serial port")
    parser.add_argument(
        "--id",
        default="gello_teleop",
        help="Parent id (per-arm calibration ids will be {id}_left / {id}_right)",
    )
    parser.add_argument("--steps", type=int, default=20, help="How many samples to print")
    parser.add_argument("--hz", type=float, default=2.0, help="Sample rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_logging()
    register_third_party_plugins()

    cfg = BimanualGelloConfig(
        id=args.id,
        left_arm_config=GelloLeaderFields(port=args.portl),
        right_arm_config=GelloLeaderFields(port=args.portr),
    )
    teleop = BimanualGello(cfg)
    teleop.connect()
    print(f"Connected. left={cfg.left_arm_config.port} (calib id {teleop.left_arm.id}), "
          f"right={cfg.right_arm_config.port} (calib id {teleop.right_arm.id})")
    print("Place each GELLO at the calibration home pose; expect ~ "
          "[0, 0, 0, -1.57, 0, 1.57, 0] rad and gripper ~ 110 mm.\n")

    period = 1.0 / args.hz
    try:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            action = teleop.get_action()
            print("  | ".join(f"{k}={action[k]:+.3f}" for k in (
                "l_joint_1", "l_joint_4", "l_joint_6", "l_gripper",
                "r_joint_1", "r_joint_4", "r_joint_6", "r_gripper",
            )))
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
