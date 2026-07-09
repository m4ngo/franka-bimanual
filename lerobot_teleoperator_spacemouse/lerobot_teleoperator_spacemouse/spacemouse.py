"""3Dconnexion SpaceMouse teleoperator for absolute EE-pose robot control.

Thin wrapper around `pyspacemouse` that exposes the device as a LeRobot
:class:`Teleoperator`. Each call to :pymeth:`get_action` integrates the
device twist into a running absolute EE pose and returns:

- absolute Cartesian position (x, y, z) in metres, updated by the device's
  linear axes (already normalized to [-1, 1] by pyspacemouse);
- absolute orientation as a unit quaternion (qx, qy, qz, qw), updated by
  composing a small-angle rotation from the device's angular axes onto the
  current orientation;
- a ``gripper`` target normalised to ``[0, 1]`` (0 = fully closed,
  1 = ``gripper_norm_max_mm``), latched from the two device buttons:
  left button = close (drive to ``gripper_min_mm``), right button = open
  (drive to ``gripper_max_mm``).

Call :pymeth:`seed_state` before starting teleop to initialise the integrated
pose from the robot arm's actual EE state.
"""

import logging

import pyspacemouse
import numpy as np
from scipy.spatial.transform import Rotation

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
    """3Dconnexion SpaceMouse leader producing an absolute EE pose and a latched gripper target."""

    config_class = SpaceMouseConfig
    name = "spacemouse"

    AXIS_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")

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

        self.cur_pos: np.ndarray = np.asarray(config.initial_pos, dtype=np.float64)
        self.cur_rot: Rotation = Rotation.from_quat(config.initial_rot)  # stored as xyzw

        self._prefix = config.prefix
        self._use_delta = config.use_delta

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def seed_state(self, pos: np.ndarray, rot_xyzw: np.ndarray) -> None:
        """Initialise the integrated EE pose from the robot's live state.

        Call this once after connecting (and before the first :pymeth:`get_action`)
        so the spacemouse starts tracking from the arm's true EE position rather
        than ``config.initial_pos`` / ``config.initial_rot``.

        Args:
            pos: EE Cartesian position ``[x, y, z]`` in metres.
            rot_xyzw: EE orientation as a unit quaternion ``[qx, qy, qz, qw]``.
        """
        self.cur_pos = np.asarray(pos, dtype=np.float64).copy()
        self.cur_rot = Rotation.from_quat(rot_xyzw)

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return {(self._prefix + axis): float for axis in self.AXIS_NAMES} | {f"{self._prefix}gripper": float, "kp": float, "kd": float}

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

        delta_pos = np.array([
            state.y * t_scale * tx,
            state.x * t_scale * ty,
            state.z * t_scale * tz,
        ], dtype=np.float64)
        # Integrate orientation: compose a small-angle rotation onto cur_rot.
        # Rotation.from_euler interprets the angles as intrinsic xyz (roll/pitch/yaw).
        delta_rot = Rotation.from_euler("xyz", [
            state.roll  * r_scale * rx,
            state.pitch * r_scale * ry,
            state.yaw   * r_scale * rz,
        ])

        # Update integrated state using clean (non-noisy) deltas.
        # Spacemouse x/y are swapped relative to robot frame.
        self.cur_pos = self.cur_pos + delta_pos
        self.cur_rot = delta_rot * self.cur_rot

        # Select output pose: delta or absolute.
        out_pos: np.ndarray = delta_pos if self._use_delta else self.cur_pos
        out_rot: Rotation   = delta_rot  if self._use_delta else self.cur_rot

        # Apply noise at output only — never to the integrated state.
        # if self.config.use_noise:
        #     out_pos = out_pos + np.random.normal(0.0, self.config.noise_pos_scale, 3)
        #     out_rot = Rotation.from_euler("xyz", np.random.normal(0.0, self.config.noise_rot_scale, 3)) * out_rot

        x, y, z = out_pos
        qx, qy, qz, qw = out_rot.as_quat()

        return {
            f"{self._prefix}x":       float(x),
            f"{self._prefix}y":       float(y),
            f"{self._prefix}z":       float(z),
            f"{self._prefix}qx":      float(qx),
            f"{self._prefix}qy":      float(qy),
            f"{self._prefix}qz":      float(qz),
            f"{self._prefix}qw":      float(qw),
            f"{self._prefix}gripper": self._gripper_target_mm,
            "kp": 0.0,
            "kd": 0.0,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # The SpaceMouse Compact has no force-feedback channel.
        raise NotImplementedError
