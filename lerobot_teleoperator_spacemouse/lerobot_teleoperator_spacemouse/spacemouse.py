"""3Dconnexion SpaceMouse teleoperator, emitting the sim policies' action.

Thin wrapper around `pyspacemouse` exposing the device as a LeRobot
:class:`Teleoperator`.

Action semantics (``use_delta=True``, the EE_DELTA path)
-------------------------------------------------------
Deliberately identical to what a base policy emits in simulation, so teleop
demonstrations and policy rollouts drive the controller through the same units
and conventions:

- **Full stick deflection == a normalized +/-1 policy action.** pyspacemouse
  normalizes its axes to [-1, 1], and ``translation_scale`` /
  ``rotation_scale`` default to robosuite's ``osc_pose.json`` output_max
  (0.05 m, 0.5 rad). Deflection therefore maps onto exactly the envelope
  ``OperationalSpaceController.scale_action`` produces, which is also what
  ``osc_torque_controller.clip_delta`` enforces downstream.
- **The rotation delta is an axis-angle vector**, as in ``set_goal_orientation``.
  It goes on the wire as the equivalent quaternion only because
  ``EE_FEATURE_KEYS`` is quaternion-shaped; ``BimanualFranka._delta_rotvec``
  converts it straight back, exactly, for any angle below pi.
- **Both channels are in the robot base frame**, each through its own
  device->base rotation (``LINEAR_DEVICE_TO_BASE`` / ``ANGULAR_DEVICE_TO_BASE``;
  see the note there on why they differ).

With ``use_delta=False`` the same deltas are integrated into an absolute pose
for the EE_POS path instead. Call :pymeth:`seed_state` first so that pose starts
at the arm's real EE rather than ``config.initial_pos``.

The ``gripper`` value is latched from the two buttons: left = close
(``gripper_min_mm``), right = open (``gripper_max_mm``); open wins if both are
pressed, so a fumbled double-press cannot crush the gripper.
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

# Device -> robot base, per channel. These are NOT the same matrix, and that is
# an empirical fact about pyspacemouse rather than a geometry mistake: its
# (roll, pitch, yaw) triple is not reported on the same axes as its (x, y, z)
# triple on this device. Deriving the angular map from the linear one -- correct
# if both triples shared an axis convention -- put roll and pitch on swapped
# robot axes, confirmed on hardware.
#
# The hardware probe is the authority here, not a derivation:
#   python scripts/check_spacemouse.py --hidraw /dev/hidraw3
#   python scripts/check_osc_axes.py --yes
# Both must stay proper rotations; _validate_device_map() enforces that at import
# so a reflection can never sneak in and make one channel mirror-imaged.
LINEAR_DEVICE_TO_BASE = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

ANGULAR_DEVICE_TO_BASE = np.eye(3, dtype=np.float64)


def _validate_device_map() -> None:
    for name, m in (("LINEAR_DEVICE_TO_BASE", LINEAR_DEVICE_TO_BASE),
                    ("ANGULAR_DEVICE_TO_BASE", ANGULAR_DEVICE_TO_BASE)):
        det = float(np.linalg.det(m))
        orth = float(np.max(np.abs(m @ m.T - np.eye(3))))
        if abs(det - 1.0) > 1e-9 or orth > 1e-9:
            raise ValueError(
                f"{name} must be a proper rotation (det={det:.6f}, "
                f"orthogonality error={orth:.2e}); a reflection would mirror "
                "that channel."
            )


_validate_device_map()


def _apply_deadzone(v: np.ndarray, deadzone: float) -> np.ndarray:
    """Zero small deflections, then rescale so full deflection still reaches 1.

    Not cosmetic: the puck cross-talks badly. Twisting it to yaw measurably
    drives the x/y linear axes to ~0.23, which at full scale is ~1.2 cm of
    translation per tick that the operator never asked for. Rescaling the
    surviving range keeps the mapping linear and preserves the normalized
    +/-1 == policy-action correspondence.
    """
    if deadzone <= 0.0:
        return v
    mag = np.abs(v)
    scaled = (mag - deadzone) / (1.0 - deadzone)
    return np.sign(v) * np.clip(scaled, 0.0, 1.0)


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
        # self._gripper_target_mm: float = float(config.initial_gripper_mm)

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
        # self._gripper_target_mm = float(self.config.initial_gripper_mm)
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
        delta = 0.0
        if len(buttons) >= 2 and buttons[1]:
            # self._gripper_target_mm = float(self.config.gripper_max_mm)
            delta = float(self.config.gripper_max_mm)
        elif buttons and buttons[0]:
            # self._gripper_target_mm = float(self.config.gripper_min_mm)
            delta = float(self.config.gripper_min_mm)

        # Raw device axes, normalized to [-1, 1] by pyspacemouse. These ARE the
        # normalized policy action once the deadzone rescale is applied.
        lin_dev = _apply_deadzone(
            np.array([state.x, state.y, state.z], dtype=np.float64), self.config.deadzone
        )
        ang_dev = _apply_deadzone(
            np.array([state.roll, state.pitch, state.yaw], dtype=np.float64), self.config.deadzone
        )

        # Into the base frame (per-channel map, see above), then per-axis sign
        # trims, then the scale that makes full deflection == a +/-1 policy action.
        lin_norm = LINEAR_DEVICE_TO_BASE @ lin_dev * np.asarray(self.config.translation_signs, dtype=np.float64)
        ang_norm = ANGULAR_DEVICE_TO_BASE @ ang_dev * np.asarray(self.config.rotation_signs, dtype=np.float64)

        delta_pos = lin_norm * self.config.translation_scale
        # Axis-angle, matching osc.py's set_goal_orientation input. from_rotvec,
        # not from_euler: only the former agrees with that convention beyond
        # infinitesimal angles.
        delta_rotvec = ang_norm * self.config.rotation_scale
        delta_rot = Rotation.from_rotvec(delta_rotvec)

        # Integrated pose for the absolute (EE_POS) path. Position sums; the
        # rotation pre-multiplies, i.e. composes in the base frame, the same way
        # set_goal_orientation composes delta onto the current orientation.
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
            f"{self._prefix}gripper": delta,
            "kp": 0.0,
            "kd": 0.0,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # The SpaceMouse Compact has no force-feedback channel.
        raise NotImplementedError
