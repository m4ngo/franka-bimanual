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
for the EE_POS path instead. The integrator runs in BOTH modes, so this class
always holds a live target pose; ``use_delta`` only selects which of the two --
the step or the target -- goes on the wire.

**Seed the target before the first :pymeth:`get_action`.** In EE_POS mode the
emitted pose IS the OSC goal, so an unseeded integrator commands
``config.initial_pos`` on step one and the arm jumps there. ``lerobot-teleoperate``
has no hook for this, which is why ``scripts/teleop_single_arm.py`` drives the
loop instead and calls :pymeth:`seed_state` between connect and the first step.

The ``gripper`` value is a target position normalized to [0, 1] against the
gripper's full travel -- the follower's own action units -- LATCHED by the two
buttons: left = close (``gripper_closed_norm``), right = open
(``gripper_open_norm``), open wins if both are pressed so a fumbled double-press
cannot crush the gripper, and neither pressed holds the last target.
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

        if config.gripper_closed_norm > config.gripper_open_norm:
            raise ValueError(
                "SpaceMouseConfig requires gripper_closed_norm <= gripper_open_norm "
                f"(got {config.gripper_closed_norm} > {config.gripper_open_norm})."
            )

        self._device: pyspacemouse.SpaceMouseDevice | None = None
        self._gripper_target: float = float(config.initial_gripper_norm)

        self.cur_pos: np.ndarray = np.asarray(config.initial_pos, dtype=np.float64)
        self.cur_rot: Rotation = Rotation.from_quat(config.initial_rot)  # stored as xyzw

        self._prefix = config.prefix
        self._use_delta = config.use_delta
        # Set by bind_pose_source(); returns the arm's measured EE pose so the
        # integrator can be held back from running away from it.
        self._pose_source = None

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

    def bind_pose_source(self, source) -> None:
        """Supply a callable returning the arm's measured ``(pos, quat_xyzw)``.

        Anti-windup for the EE_POS integrator, and the reason EE_POS is safe to
        offer at all. ``cur_pos``/``cur_rot`` advance by up to one full delta per
        tick regardless of whether the arm follows, so with the arm slowed or
        stalled the target runs away at ``translation_scale * fps`` -- 1.0 m/s at
        the defaults -- and the OSC's ``kp * error`` grows with it until the torque
        clamp. Bounding the lead bounds that force.

        EE_DELTA needs none of this: ``BimanualFranka`` rebuilds the goal from the
        MEASURED pose every step, so the lead is structurally one clipped delta.

        ``source`` may return None when no state is available yet, in which case
        the target is left alone for that tick.
        """
        self._pose_source = source

    def _limit_lead(self) -> None:
        """Hold the integrated target within max_lead of the arm's real pose."""
        if self._pose_source is None:
            return
        measured = self._pose_source()
        if measured is None:
            return
        pos, quat_xyzw = measured

        lead = self.cur_pos - np.asarray(pos, dtype=np.float64)
        dist = float(np.linalg.norm(lead))
        if dist > self.config.max_lead_m:
            self.cur_pos = pos + lead * (self.config.max_lead_m / dist)

        measured_rot = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
        rotvec = (self.cur_rot * measured_rot.inv()).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > self.config.max_lead_rad:
            self.cur_rot = (
                Rotation.from_rotvec(rotvec * (self.config.max_lead_rad / angle)) * measured_rot
            )

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
        self._gripper_target = float(self.config.initial_gripper_norm)
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

        # Buttons: index 0 = left (close), index 1 = right (open). If both are
        # pressed in the same sample we prefer "open" so an accidental
        # double-press doesn't crush the gripper. The target LATCHES: neither
        # pressed holds the last one, which is what makes the emitted value an
        # absolute position rather than a per-tick nudge the follower has to
        # integrate. That integration used to live in BimanualFranka.send_action
        # and applied to every leader, including the ones already emitting an
        # absolute position.
        buttons = list(state.buttons)
        if len(buttons) >= 2 and buttons[1]:
            self._gripper_target = float(self.config.gripper_open_norm)
        elif buttons and buttons[0]:
            self._gripper_target = float(self.config.gripper_closed_norm)

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
        # Then held back to within max_lead of the arm, so a stick the arm cannot
        # follow stops winding the target up. No-op in EE_DELTA, where the emitted
        # value is this tick's step and the integrator is only carried for state.
        if not self._use_delta:
            self._limit_lead()

        # Select output pose: this tick's step, or the running target.
        out_pos: np.ndarray = delta_pos if self._use_delta else self.cur_pos
        out_rot: Rotation   = delta_rot  if self._use_delta else self.cur_rot

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
            f"{self._prefix}gripper": self._gripper_target,
            "kp": 0.0,
            "kd": 0.0,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # The SpaceMouse Compact has no force-feedback channel.
        raise NotImplementedError
