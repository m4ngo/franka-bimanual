"""GELLO teleoperator interface using Dynamixel motors.

Implements the LeRobot Teleoperator interface for the GELLO FR3 leader device
(Philipp Wu et al., https://wuphilipp.github.io/gello_site/). Reads joint
positions from Dynamixel servos, applies a per-joint calibration, and exposes
normalized joint + gripper commands for the follower robot.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Mapping

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_gello import GelloConfig

logger = logging.getLogger(__name__)

# Dynamixel encoder resolution: counts per 2*pi rad.
_DYNAMIXEL_COUNTS = 4096
# Gripper command range exposed to the follower robot, in millimeters.
_GRIPPER_MAX_MM = 110.0
# Fallback sleep after a read error before retrying the bus.
_READ_LOOP_BACKOFF_S = 0.05
# Grace period for the read thread to exit on disconnect.
_READ_THREAD_JOIN_S = 2.0


@dataclass
class GelloCalibration:
    # Map from motor name to the encoder offset (in counts) at the home pose.
    joint_offsets: dict[str, int]
    # Motor counts at the open and fully-closed gripper positions.
    gripper_open_position: int
    gripper_closed_position: int


class Gello(Teleoperator):
    """GELLO leader device for the Franka FR3."""

    config_class = GelloConfig
    name = "gello"

    # Radians per encoder count.
    RAD_PER_COUNT = 2 * np.pi / (_DYNAMIXEL_COUNTS - 1)

    JOINT_NAMES = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "gripper",
    ]

    def __init__(self, config: GelloConfig):
        super().__init__(config)
        self.config = config

        expected = len(self.JOINT_NAMES)
        if (
            len(config.calibration_position) != expected
            or len(config.joint_signs) != expected
        ):
            raise ValueError(
                "GelloConfig joint calibration must define one value per joint for the FR3 leader."
            )

        self._calibration: GelloCalibration | None = None
        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors={
                name: Motor(idx, "xl330-m288", MotorNormMode.RANGE_M100_100)
                for idx, name in enumerate(self.JOINT_NAMES, start=1)
            },
        )

        # Async read loop state.
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.lock: Lock = Lock()
        self.latest_action: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return {motor: float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Handshake is done manually after baudrate is set so the initial
        # connect does not assume a specific baudrate.
        self.bus.connect(handshake=False)
        self.bus.set_baudrate(self.config.baudrate)
        self.bus._handshake()
        self.bus._assert_motors_exist()

        self._load_calibration()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        self.configure()

        if self.config.use_async:
            if self.is_calibrated:
                # Seed latest_action before starting the read thread.
                raw = self.bus.sync_read("Present_Position", normalize=False)
                self.latest_action = self._process_action(raw)
                self._start_read_thread()
            else:
                logger.info(
                    "%s connected without calibration; async action stream will start after calibration.",
                    self,
                )

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.use_async:
            self._stop_read_thread()

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

    def calibrate(self) -> None:
        # Stop the async read loop so the bus isn't in use during calibration.
        if self.config.use_async:
            self._stop_read_thread()

        self.bus.disable_torque()
        if self._calibration:
            user_input = input(
                "Press ENTER to use existing calibration, or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Using existing calibration")
                return

        logger.info(f"\nRunning calibration of {self}")
        input(f"Move {self} to the home position and press ENTER....")
        start_joints = self.bus.sync_read("Present_Position", normalize=False)

        gripper_open = int(start_joints["gripper"])
        calibration = GelloCalibration(
            joint_offsets={motor: int(start_joints[motor]) for motor in self.JOINT_NAMES},
            gripper_open_position=gripper_open,
            gripper_closed_position=gripper_open - self.config.gripper_travel_counts,
        )
        self._calibration = calibration

        with open(self.calibration_fpath, "w") as f:
            json.dump(calibration.__dict__, f)
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
                # Extended position mode allows multi-turn motion so small
                # assembly errors don't leave a joint jammed at 0 or 4095.
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # Current-based position control on the gripper lets the operator
        # press through the command to open it manually without overloading
        # the servo, then have it snap back when released.
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

    def setup_motors(self) -> None:
        """Iteratively assign IDs to each motor; called from the CLI utility."""
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.use_async:
            with self.lock:
                if self.latest_action is None:
                    # No async read yet; fall back to a synchronous one.
                    logger.warning(
                        f"{self} async read loop has not updated latest_action yet. "
                        "Performing synchronous read."
                    )
                    raw = self.bus.sync_read("Present_Position", normalize=False)
                    self.latest_action = self._process_action(raw)
                return self.latest_action.copy()

        start = time.perf_counter()
        raw = self.bus.sync_read("Present_Position", normalize=False)
        result = self._process_action(raw)
        print(f"bus.sync_read took {time.perf_counter() - start} seconds")
        return result

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_action(self, raw_action: Mapping[str, Any]) -> dict[str, float]:
        """Convert raw motor counts to robot-facing joint angles + gripper mm."""
        calibration = self._calibration
        if calibration is None:
            raise RuntimeError(f"{self} is not calibrated.")

        result: dict[str, float] = {}
        # The "gripper" joint is computed here for uniform error semantics
        # (missing offsets raise consistently), then overwritten below.
        for idx, motor in enumerate(self.JOINT_NAMES):
            offset = calibration.joint_offsets[motor]
            sign = self.config.joint_signs[idx]
            ref_rad = self.config.calibration_position[idx]
            result[motor] = (
                sign * (float(raw_action[motor]) - offset) * self.RAD_PER_COUNT + ref_rad
            )

        # Map gripper counts -> [0, _GRIPPER_MAX_MM] with open at _GRIPPER_MAX_MM.
        gripper_range = calibration.gripper_closed_position - calibration.gripper_open_position
        normalized = (float(raw_action["gripper"]) - calibration.gripper_open_position) / gripper_range
        result["gripper"] = (1.0 - normalized) * _GRIPPER_MAX_MM
        return result

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        alpha = self.config.smoothing
        while not self.stop_event.is_set():
            try:
                raw = self.bus.sync_read("Present_Position", normalize=False)
                new_action = self._process_action(raw)

                with self.lock:
                    if self.latest_action is None:
                        self.latest_action = new_action
                    else:
                        # Exponential moving average per joint.
                        for k, v in new_action.items():
                            self.latest_action[k] = alpha * v + (1 - alpha) * self.latest_action[k]

            except Exception:
                time.sleep(_READ_LOOP_BACKOFF_S)

    def _start_read_thread(self) -> None:
        # Ensure any previous thread is fully stopped before starting a new one.
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=_READ_THREAD_JOIN_S)

        self.thread = None
        self.stop_event = None

    def _load_calibration(self, fpath: Path | None = None) -> None:
        if fpath is None:
            fpath = self.calibration_fpath
        if fpath.is_file():
            with open(fpath, "r") as f:
                self._calibration = GelloCalibration(**json.load(f))
            logger.info(f"Calibration loaded from {fpath}")
        else:
            logger.info(f"No calibration file found at {fpath}")
            self._calibration = None
