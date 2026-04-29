"""Direct teleoperation script for bimanual Franka with two 3Dconnexion SpaceMice.

Runs one SpaceMouse teleoperator per arm in EE-delta mode and fuses the two
action streams into a single bimanual Franka twist+gripper action every
control step.

Each SpaceMouse provides 6-DoF Cartesian velocity input plus two buttons
that latch the gripper target to fully open / fully closed.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lerobot_robot_bimanual_franka import BimanualFrankaConfig
from lerobot_teleoperator_spacemouse import SpaceMouseConfig

# EE-delta axis names exposed by both the SpaceMouse teleop and the
# bimanual Franka in EE-delta mode.
EE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
# Fixed control loop rate (Hz). Matches the GELLO teleoperate script so
# the safety screen and PD controller see the same cadence.
CONTROL_LOOP_HZ = 20
# How many consecutive send_action failures we tolerate before treating the
# robot link as dead and exiting the loop. A single failure is logged as a
# warning so transient hiccups don't kill the session.
MAX_CONSECUTIVE_SEND_FAILURES = 3


def _build_bimanual_action(
    left_action: dict[str, float],
    right_action: dict[str, float],
) -> dict[str, float]:
    """Fuse per-arm SpaceMouse twists + gripper targets into one bimanual action."""
    action: dict[str, float] = {}
    for axis in EE_AXES:
        action[f"l_{axis}"] = float(left_action[axis])
        action[f"r_{axis}"] = float(right_action[axis])
    action["l_gripper"] = float(left_action["gripper"])
    action["r_gripper"] = float(right_action["gripper"])
    return action

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-SpaceMouse bimanual Franka EE-delta teleoperation"
    )
    parser.add_argument(
        "--portl",
        default=os.getenv("SPACEMOUSE_LEFT_HIDRAW", "/dev/hidraw4"),
        help="hidraw device path for the SpaceMouse driving the LEFT arm",
    )
    parser.add_argument(
        "--portr",
        default=os.getenv("SPACEMOUSE_RIGHT_HIDRAW", "/dev/hidraw5"),
        help="hidraw device path for the SpaceMouse driving the RIGHT arm",
    )
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
        use_ee_delta=True,
    )


def _make_teleop_configs(
    args: argparse.Namespace,
) -> tuple[SpaceMouseConfig, SpaceMouseConfig]:
    left = SpaceMouseConfig(
        hidraw_path=args.portl,
        id=os.getenv("SPACEMOUSE_LEFT_ID", "spacemouse_teleop_left")
    )
    right = SpaceMouseConfig(
        hidraw_path=args.portr,
        id=os.getenv("SPACEMOUSE_RIGHT_ID", "spacemouse_teleop_right")
    )
    return left, right


def main() -> None:
    args = parse_args()
    init_logging()
    logging.info("Starting bimanual Franka <-> dual SpaceMouse EE-delta teleoperation")
    register_third_party_plugins()

    left_teleop_cfg, right_teleop_cfg = _make_teleop_configs(args)
    left_teleop = make_teleoperator_from_config(left_teleop_cfg)
    right_teleop = make_teleoperator_from_config(right_teleop_cfg)
    robot = make_robot_from_config(_make_robot_config())

    left_teleop.connect()
    right_teleop.connect()
    try:
        logging.info("Attempting robot connection")
        robot.connect()
        logging.info("Robot connection succeeded!")
    except Exception as exc:  # noqa: BLE001 - log and keep running open-loop
        logging.warning("Robot connection failed: %s", exc)

    loop_period = 1.0 / CONTROL_LOOP_HZ
    consecutive_send_failures = 0

    try:
        while True:
            loop_start = time.perf_counter()

            if robot.is_connected:
                try:
                    robot.get_observation()
                except DeviceNotConnectedError:
                    logging.warning("Robot disconnected while reading observation")
                except Exception as exc:  # noqa: BLE001 - log + keep loop alive
                    logging.warning("Observation read failed: %s", exc)

            left_action = left_teleop.get_action()
            right_action = right_teleop.get_action()
            robot_action = _build_bimanual_action(left_action, right_action)

            if robot.is_connected:
                try:
                    robot.send_action(robot_action)
                    consecutive_send_failures = 0
                except Exception as exc:  # noqa: BLE001
                    consecutive_send_failures += 1
                    logging.warning(
                        "send_action failed (%d/%d): %s",
                        consecutive_send_failures,
                        MAX_CONSECUTIVE_SEND_FAILURES,
                        exc,
                    )
                    if consecutive_send_failures >= MAX_CONSECUTIVE_SEND_FAILURES:
                        logging.error(
                            "Robot link appears dead after %d consecutive failures; "
                            "exiting teleoperation loop.",
                            consecutive_send_failures,
                        )
                        break

            elapsed = time.perf_counter() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    except KeyboardInterrupt:
        logging.info("Teleoperation interrupted by user")
    finally:
        logging.info("Starting graceful shutdown")
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for name, device in (
                ("left SpaceMouse", left_teleop),
                ("right SpaceMouse", right_teleop),
                ("robot", robot if robot.is_connected else None),
            ):
                if device is None:
                    continue
                try:
                    device.disconnect()
                except Exception as exc:  # noqa: BLE001 - keep tearing the rest down
                    logging.warning("Failed to disconnect %s cleanly: %s", name, exc)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
