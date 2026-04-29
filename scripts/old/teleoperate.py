"""Direct teleoperation script for bimanual Franka with two GELLO devices.

Runs one GELLO teleoperator per arm and fuses the two action streams into a
single bimanual Franka action every control step.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

from lerobot.processor import RobotObservation, make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lerobot_robot_bimanual_franka import BimanualFrankaConfig
from lerobot_teleoperator_gello import GelloConfig

# Number of joints per Franka arm.
NUM_JOINTS = 7
# Fixed control loop rate (Hz).
CONTROL_LOOP_HZ = 20


def _build_bimanual_action(
    left_action: dict[str, float],
    right_action: dict[str, float],
) -> dict[str, float]:
    """Fuse per-arm teleop commands into a single bimanual action dict."""
    action: dict[str, float] = {}
    for i in range(1, NUM_JOINTS + 1):
        action[f"l_joint_{i}"] = float(left_action[f"joint_{i}"])
        action[f"r_joint_{i}"] = float(right_action[f"joint_{i}"])
    action["l_gripper"] = float(left_action["gripper"])
    action["r_gripper"] = float(right_action["gripper"])
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-GELLO bimanual Franka teleoperation")
    parser.add_argument("--portl", required=True, help="Serial device path for left GELLO")
    parser.add_argument("--portr", required=True, help="Serial device path for right GELLO")
    return parser.parse_args()


def _make_robot_config() -> BimanualFrankaConfig:
    return BimanualFrankaConfig(
        l_server_ip="192.168.3.11",
        l_robot_ip="192.168.200.2",
        l_gripper_ip="192.168.2.21",
        l_port=18813,
        r_server_ip="192.168.3.10",
        r_robot_ip="192.168.201.10",
        r_gripper_ip="192.168.2.20",
        r_port=18812,
        use_ee_delta=False,
    )


def _make_teleop_configs(args: argparse.Namespace) -> tuple[GelloConfig, GelloConfig]:
    left = GelloConfig(
        port=args.portl,
        id="gello_teleop_left",
    )
    right = GelloConfig(
        port=args.portr,
        id="gello_teleop_right",
    )
    return left, right


def main() -> None:
    args = parse_args()
    init_logging()
    logging.info("Starting bimanual Franka <-> dual GELLO teleoperation")
    register_third_party_plugins()

    left_teleop_cfg, right_teleop_cfg = _make_teleop_configs(args)
    left_teleop = make_teleoperator_from_config(left_teleop_cfg)
    right_teleop = make_teleoperator_from_config(right_teleop_cfg)
    robot = make_robot_from_config(_make_robot_config())

    left_teleop_action_processor, _, _ = make_default_processors()
    right_teleop_action_processor, robot_action_processor, _ = make_default_processors()

    left_teleop.connect()
    right_teleop.connect()
    try:
        logging.info("Attempting robot connection")
        robot.connect()
        logging.info("Robot connection succeeded!")
    except Exception as exc:  # noqa: BLE001 - log and keep running open-loop
        logging.warning("Robot connection failed: %s", exc)

    loop_period = 1.0 / CONTROL_LOOP_HZ

    try:
        while True:
            loop_start = time.perf_counter()

            obs: RobotObservation = {}
            if robot.is_connected:
                try:
                    obs = robot.get_observation()
                except DeviceNotConnectedError:
                    logging.warning("Robot disconnected while reading observation")
                    obs = {}

            left_raw_action = left_teleop.get_action()
            right_raw_action = right_teleop.get_action()
            left_action = left_teleop_action_processor((left_raw_action, obs))
            right_action = right_teleop_action_processor((right_raw_action, obs))

            robot_action = _build_bimanual_action(left_action, right_action)
            robot_action = robot_action_processor((robot_action, obs))

            if robot.is_connected:
                robot.send_action(robot_action)

            elapsed = time.perf_counter() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    except KeyboardInterrupt:
        logging.info("Teleoperation interrupted by user")
    finally:
        logging.info("Starting graceful shutdown")
        # Ignore Ctrl+C during shutdown so hardware teardown completes.
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            left_teleop.disconnect()
            right_teleop.disconnect()
            if robot.is_connected:
                robot.disconnect()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
