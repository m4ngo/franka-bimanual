"""Single-arm teleoperation script for bimanual Franka using one GELLO device.

Activates only one side of the bimanual Franka so that a single GELLO leader
can drive the selected arm while the other arm stays idle.
"""

from __future__ import annotations

import argparse
import logging
import os
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
from lerobot_teleoperator_gello import GelloConfig

# Number of joints per Franka arm.
NUM_JOINTS = 7
# How often to log a debug sample when --debug is set (one log per N steps).
_DEBUG_LOG_INTERVAL = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-arm Franka <-> one GELLO teleoperation")
    parser.add_argument(
        "--arm", choices=["left", "right"], default="left", help="Arm controlled by GELLO"
    )
    parser.add_argument(
        "--port",
        default=os.getenv("GELLO_PORT", "/dev/ttyUSB1"),
        help="GELLO serial port",
    )
    parser.add_argument(
        "--id",
        default=os.getenv("GELLO_ID", "gello_teleop_left"),
        help="GELLO calibration id",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="Control loop frequency")
    parser.add_argument("--debug", action="store_true", help="Log command and observation deltas")
    return parser.parse_args()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _make_robot_config(active_arm: str) -> BimanualFrankaConfig:
    return BimanualFrankaConfig(
        l_server_ip=os.getenv("L_SERVER_IP", "192.168.3.11"),
        l_robot_ip=os.getenv("L_ROBOT_IP", "192.168.200.2"),
        l_gripper_ip=os.getenv("L_GRIPPER_IP", "192.168.2.21"),
        l_port=_env_int("L_PORT", 18813),
        r_server_ip=os.getenv("R_SERVER_IP", "192.168.3.10"),
        r_robot_ip=os.getenv("R_ROBOT_IP", "192.168.201.10"),
        r_gripper_ip=os.getenv("R_GRIPPER_IP", "192.168.2.20"),
        r_port=_env_int("R_PORT", 18812),
        use_ee_delta=False,
        active_arms=(active_arm,),
    )


def _build_action(teleop_action: dict[str, float], controlled_prefix: str) -> dict[str, float]:
    """Build a single-arm robot action from a GELLO teleop sample."""
    action: dict[str, float] = {
        f"{controlled_prefix}_joint_{i}": float(teleop_action[f"joint_{i}"])
        for i in range(1, NUM_JOINTS + 1)
    }
    action[f"{controlled_prefix}_gripper"] = float(teleop_action["gripper"])
    return action


def main() -> None:
    args = parse_args()

    init_logging()
    register_third_party_plugins()

    controlled_prefix = "l" if args.arm == "left" else "r"

    robot = make_robot_from_config(_make_robot_config(controlled_prefix))
    teleop = make_teleoperator_from_config(GelloConfig(port=args.port, id=args.id))

    teleop.connect()
    robot.connect()

    loop_period = 1.0 / args.hz
    logging.info("Starting single-arm teleop on %s arm at %.2f Hz", args.arm, args.hz)

    try:
        step = 0
        while True:
            t0 = time.perf_counter()

            try:
                obs = robot.get_observation()
            except DeviceNotConnectedError:
                logging.warning("Robot disconnected while reading observation")
                break

            raw_action = teleop.get_action()
            robot_action = _build_action(raw_action, controlled_prefix)
            robot.send_action(robot_action)

            if args.debug and step % _DEBUG_LOG_INTERVAL == 0:
                c_obs = float(obs[f"{controlled_prefix}_joint_1"])
                c_cmd = float(robot_action[f"{controlled_prefix}_joint_1"])
                logging.info(
                    "cmd-vs-obs | %s j1: %.4f vs %.4f",
                    controlled_prefix,
                    c_cmd,
                    c_obs,
                )
            step += 1

            dt = time.perf_counter() - t0
            if dt < loop_period:
                time.sleep(loop_period - dt)

    except KeyboardInterrupt:
        logging.info("Teleoperation interrupted by user")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
