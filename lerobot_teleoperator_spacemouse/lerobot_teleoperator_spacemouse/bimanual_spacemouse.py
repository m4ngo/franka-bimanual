"""Bimanual SpaceMouse teleoperator: two single-device leaders fused into one.

Action keys are emitted with ``l_`` / ``r_`` prefixes to match the schema
expected by BimanualFranka in EE-position mode
(e.g. ``l_x``, ``l_qw``, ``l_gripper`` … ``r_x``, ``r_qw``, ``r_gripper``).

Before starting the control loop you can sync the integrated pose of each arm
to the robot's actual EE state via the public ``left_arm`` / ``right_arm``
attributes::

    teleop.left_arm.seed_state(left_pos, left_rot_xyzw)
    teleop.right_arm.seed_state(right_pos, right_rot_xyzw)
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_bimanual_spacemouse import BimanualSpaceMouseConfig
from .config_spacemouse import SpaceMouseConfig
from .spacemouse import SpaceMouse

logger = logging.getLogger(__name__)


def _make_spacemouse(parent: BimanualSpaceMouseConfig, side: str, fields) -> SpaceMouse:
    cfg = SpaceMouseConfig(
        id=f"{parent.id}_{side}" if parent.id else None,
        **asdict(fields),
    )
    return SpaceMouse(cfg)


class BimanualSpaceMouse(Teleoperator):
    """Pair of SpaceMouse leaders presented as a single bimanual teleoperator."""

    config_class = BimanualSpaceMouseConfig
    name = "bimanual_spacemouse"

    def __init__(self, config: BimanualSpaceMouseConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = _make_spacemouse(config, "left", config.left_arm_config)
        self.right_arm = _make_spacemouse(config, "right", config.right_arm_config)

    # Action channels that are NOT per-arm. BimanualFranka.send_action reads these
    # unprefixed (they select one OSC gain schedule for the whole controller), so
    # prefixing them the way the pose keys are prefixed emits l_kp/r_kp and leaves
    # action["kp"] missing -- a KeyError on the first step of every bimanual run.
    _SHARED_KEYS = ("kp", "kd")

    @property
    def action_features(self) -> dict[str, type]:
        left, right = self.left_arm.action_features, self.right_arm.action_features
        return {
            **{f"l_{k}": v for k, v in left.items() if k not in self._SHARED_KEYS},
            **{f"r_{k}": v for k, v in right.items() if k not in self._SHARED_KEYS},
            **{k: left[k] for k in self._SHARED_KEYS if k in left},
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        self.left_arm.connect(calibrate=calibrate)
        try:
            self.right_arm.connect(calibrate=calibrate)
        except Exception:
            try:
                self.left_arm.disconnect()
            except Exception:
                logger.exception("Failed to disconnect left SpaceMouse during rollback")
            raise

    def disconnect(self) -> None:
        errors: list[tuple[str, BaseException]] = []
        for label, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if not arm.is_connected:
                continue
            try:
                arm.disconnect()
            except Exception as exc:
                errors.append((label, exc))

        if errors:
            details = ", ".join(f"{label}: {exc}" for label, exc in errors)
            raise RuntimeError(f"BimanualSpaceMouse disconnect errors: {details}")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def seed_from_robot(self, robot) -> None:
        """Seed both arms' integrated poses from the robot's live EE state.

        Args:
            robot: A connected ``BimanualFranka`` instance.  Its
                ``robot_manager`` is queried for the current kinematic state of
                the ``"l"`` and ``"r"`` arms.
        """
        kin = robot.robot_manager.current_kinematic_state_batch(list(robot.active_arms))
        for arm_key, spacemouse in (("l", self.left_arm), ("r", self.right_arm)):
            if arm_key not in kin:
                logger.warning("BimanualSpaceMouse.seed_from_robot: arm '%s' not found", arm_key)
                continue
            _, _, _, pos, rot, _ = kin[arm_key]
            spacemouse.seed_state(np.asarray(pos), np.asarray(rot))
            logger.info("Seeded %s arm SpaceMouse from EE pos=%s", arm_key, pos)

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left = self.left_arm.get_action()
        right = self.right_arm.get_action()
        # The gain channels are the controller's, not an arm's -- see _SHARED_KEYS.
        # Taken from the left device; both report the same constant 0.0, and a
        # bimanual rig has one OSC gain schedule, so there is nothing to reconcile.
        return {
            **{f"l_{k}": v for k, v in left.items() if k not in self._SHARED_KEYS},
            **{f"r_{k}": v for k, v in right.items() if k not in self._SHARED_KEYS},
            **{k: left[k] for k in self._SHARED_KEYS if k in left},
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError
