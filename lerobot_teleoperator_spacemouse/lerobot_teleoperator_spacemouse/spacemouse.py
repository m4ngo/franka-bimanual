"""3Dconnexion SpaceMouse teleoperator for absolute EE-pose robot control.

Thin wrapper around `pyspacemouse` that exposes the device as a LeRobot
:class:`Teleoperator`. Each call to :pymeth:`get_action` integrates the
device twist into a running absolute EE pose and returns:

- EE position delta (x, y, z): device axes in normalized ``[-1, 1]`` when
  ``use_delta=True`` (physical scaling lives in the robot's
  ``osc_output_max_pos``);
- EE rotation delta (rx, ry, rz): normalized axis-angle components in
  ``[-1, 1]`` when ``use_delta=True`` (physical scaling lives in
  ``osc_output_max_rot``); absolute orientation as a unit quaternion when
  ``use_delta=False``;
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

    AXIS_NAMES_POS = ("x", "y", "z", "qx", "qy", "qz", "qw")
    AXIS_NAMES_DELTA = ("x", "y", "z", "rx", "ry", "rz")

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

    @property
    def axis_names(self) -> tuple[str, ...]:
        return self.AXIS_NAMES_DELTA if self._use_delta else self.AXIS_NAMES_POS

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
        return {(self._prefix + axis): float for axis in self.axis_names} | {f"{self._prefix}gripper": float, "kp": float, "kd": float}

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

        # Diagnostic: log the first non-zero axis reading. If you never see
        # this line while moving the spacemouse, the device is on the wrong
        # hidraw node (buttons are on a different interface than 6-DOF axes).
        _ax = (state.x, state.y, state.z, state.roll, state.pitch, state.yaw)
        if not getattr(self, "_axes_logged", False) and any(abs(v) > 0.01 for v in _ax):
            self._axes_logged = True
            logger.info(
                "%s first non-zero axis reading — x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f",
                self, state.x, state.y, state.z, state.roll, state.pitch, state.yaw,
            )
        logger.debug(
            "%s axes: x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f",
            self, state.x, state.y, state.z, state.roll, state.pitch, state.yaw,
        )

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

        dbt = self.config.deadband_translation
        dbr = self.config.deadband_rotation

        def _db(v: float, threshold: float) -> float:
            return 0.0 if abs(v) <= threshold else v

        sx = _db(state.x, dbt)
        sy = _db(state.y, dbt)
        sz = _db(state.z, dbt)
        sroll  = _db(state.roll,  dbr)
        spitch = _db(state.pitch, dbr)
        syaw   = _db(state.yaw,   dbr)

        delta_pos = np.array([
            sy * t_scale * tx,
            sx * t_scale * ty,
            sz * t_scale * tz,
        ], dtype=np.float64)
        delta_rot_aa = np.array([
            sroll  * r_scale * rx,
            spitch * r_scale * ry,
            syaw   * r_scale * rz,
        ], dtype=np.float64)
        delta_rot = Rotation.from_euler("xyz", delta_rot_aa)

        self.cur_pos = self.cur_pos + delta_pos
        self.cur_rot = delta_rot * self.cur_rot

        out_pos: np.ndarray = delta_pos if self._use_delta else self.cur_pos
        if self._use_delta:
            out_rot_aa = delta_rot_aa
        else:
            out_rot_aa = None
            out_rot_quat = self.cur_rot

        if self.config.use_noise:
            out_pos = out_pos + np.random.normal(0.0, self.config.noise_pos_scale, 3)
            if self._use_delta:
                out_rot_aa = out_rot_aa + np.random.normal(0.0, self.config.noise_rot_scale, 3)
            else:
                out_rot_quat = Rotation.from_euler("xyz", np.random.normal(0.0, self.config.noise_rot_scale, 3)) * out_rot_quat

        x, y, z = out_pos
        action = {
            f"{self._prefix}x": float(x),
            f"{self._prefix}y": float(y),
            f"{self._prefix}z": float(z),
            f"{self._prefix}gripper": self._gripper_target_mm,
            "kp": 0.0,
            "kd": 0.0,
        }
        if self._use_delta:
            rx_out, ry_out, rz_out = out_rot_aa
            action[f"{self._prefix}rx"] = float(rx_out)
            action[f"{self._prefix}ry"] = float(ry_out)
            action[f"{self._prefix}rz"] = float(rz_out)
        else:
            qx, qy, qz, qw = out_rot_quat.as_quat()
            action[f"{self._prefix}qx"] = float(qx)
            action[f"{self._prefix}qy"] = float(qy)
            action[f"{self._prefix}qz"] = float(qz)
            action[f"{self._prefix}qw"] = float(qw)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # The SpaceMouse Compact has no force-feedback channel.
        raise NotImplementedError
