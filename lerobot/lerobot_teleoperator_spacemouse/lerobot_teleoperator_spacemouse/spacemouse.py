"""3Dconnexion SpaceMouse teleoperator for EE-delta robot control.

Thin wrapper around `pyspacemouse` that exposes the device as a LeRobot
:class:`Teleoperator`. Each call to :pymeth:`get_action` returns:

- linear and angular twist components (m/s, rad/s) scaled from the device
  axes (already normalized to [-1, 1] by pyspacemouse);
- a ``gripper`` target in millimeters, latched from the two device buttons:
  left button = close (drive to ``gripper_min_mm``), right button = open
  (drive to ``gripper_max_mm``).
"""

import logging

import pyspacemouse

from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_spacemouse import SpaceMouseConfig

logger = logging.getLogger(__name__)

# Upper bound on how many times we drain the HID report queue per
# get_action() call. pyspacemouse.read() consumes at most one report per
# call; bounding the loop guarantees we never spin indefinitely if the
# device is somehow streaming faster than we can keep up.
_MAX_DRAIN_PER_TICK = 64


class SpaceMouse(Teleoperator):
    """3Dconnexion SpaceMouse leader producing EE-delta twists and a latched gripper target."""

    config_class = SpaceMouseConfig
    name = "spacemouse"

    # Order matches the EE-delta convention used by the bimanual Franka
    # (linear xyz, angular roll/pitch/yaw).
    AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(self, config: SpaceMouseConfig):
        super().__init__(config)
        self.config = config

        if config.gripper_min_mm > config.gripper_max_mm:
            raise ValueError(
                "SpaceMouseConfig requires gripper_min_mm <= gripper_max_mm "
                f"(got {config.gripper_min_mm} > {config.gripper_max_mm})."
            )

        self._device: pyspacemouse.SpaceMouseDevice | None = None
        self._gripper_target_mm: float = float(config.initial_gripper_mm)

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return {axis: float for axis in self.AXIS_NAMES} | {"gripper": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._device is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # open_by_path() configures the device with non-blocking reads, so
        # get_action() can poll the latest state without ever stalling the
        # control loop.
        self._device = pyspacemouse.open_by_path(self.config.hidraw_path)
        self._gripper_target_mm = float(self.config.initial_gripper_mm)
        logger.info("%s connected on %s", self, self.config.hidraw_path)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        device, self._device = self._device, None
        if device is not None:
            device.close()
        logger.info("%s disconnected.", self)

    def calibrate(self) -> None:
        # SpaceMouse axes self-zero in hardware; nothing to do here.
        pass

    def configure(self) -> None:
        # All configuration lives in SpaceMouseConfig; nothing to push.
        pass

    def get_action(self) -> dict[str, float]:
        if self._device is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Drain the HID report backlog so we always act on the most recent
        # device state. pyspacemouse processes one report per read() call
        # but the SpaceMouse emits separate reports per channel (linear,
        # angular, buttons) at ~100 Hz. Calling read() once per 20 Hz
        # control tick lets a multi-cycle queue build up in the kernel's
        # hidraw buffer, which manifests as laggy input AND as the robot
        # continuing to track the previous twist after the operator
        # releases the joystick (because the "release" reports are still
        # waiting in the queue). state.t is updated only when a new report
        # is processed, so we stop draining as soon as it stops advancing.
        state = self._device.read()
        last_t = state.t
        for _ in range(_MAX_DRAIN_PER_TICK):
            state = self._device.read()
            if state.t == last_t:
                break
            last_t = state.t

        # Buttons: index 0 = left (close), index 1 = right (open). If both
        # are pressed in the same sample we prefer "open" so an accidental
        # double-press doesn't crush the gripper.
        buttons = list(state.buttons)
        if len(buttons) >= 2 and buttons[1]:
            self._gripper_target_mm = float(self.config.gripper_max_mm)
        elif buttons and buttons[0]:
            self._gripper_target_mm = float(self.config.gripper_min_mm)

        t_scale = self.config.translation_scale
        r_scale = self.config.rotation_scale
        tx, ty, tz = self.config.translation_signs
        rx, ry, rz = self.config.rotation_signs
        return {
            "x": state.y * t_scale * tx,
            "y": state.x * t_scale * ty,
            "z": state.z * t_scale * tz,
            "roll": state.roll * r_scale * rx,
            "pitch": state.pitch * r_scale * ry,
            "yaw": state.yaw * r_scale * rz,
            "gripper": self._gripper_target_mm,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # The SpaceMouse Compact has no force-feedback channel.
        raise NotImplementedError
