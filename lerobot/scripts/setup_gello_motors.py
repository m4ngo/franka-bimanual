"""GELLO motor ID setup utility.

Iterates over each Dynamixel motor on the GELLO bus and assigns IDs / baudrate
using the plugin's built-in setup routine.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lerobot_teleoperator_gello import Gello, GelloConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up GELLO motor IDs")
    parser.add_argument("--port", required=True, help="Serial device path for the Dynamixel bus")
    parser.add_argument("--id", default="gello_teleop", help="Identifier saved with the calibration")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Optional directory where calibration data should be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teleop = Gello(
        GelloConfig(
            port=args.port,
            id=args.id,
            calibration_dir=args.calibration_dir,
        )
    )
    teleop.setup_motors()


if __name__ == "__main__":
    main()
