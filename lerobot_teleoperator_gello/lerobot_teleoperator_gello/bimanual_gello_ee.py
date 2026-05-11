"""Bimanual GELLO EE teleoperator: two GELLO EE leaders fused into one.

Action keys are emitted with ``l_`` / ``r_`` prefixes to match the schema
expected by BimanualFranka in EE-position mode
(e.g. ``l_x``, ``l_qw``, ``l_gripper`` … ``r_x``, ``r_qw``, ``r_gripper``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_bimanual_gello_ee import BimanualGelloEEConfig
from .config_gello import GelloLeaderFields
from .config_gello_ee import GelloEEConfig
from .gello_ee import GelloEE

logger = logging.getLogger(__name__)


def _make_gello_ee(parent: BimanualGelloEEConfig, side: str, fields: GelloLeaderFields) -> GelloEE:
    cfg = GelloEEConfig(
        id=f"{parent.id}_{side}" if parent.id else None,
        calibration_dir=parent.calibration_dir,
        **asdict(fields),
    )
    return GelloEE(cfg)


class BimanualGelloEE(Teleoperator):
    """Pair of GELLO EE leaders presenting absolute EE poses for bimanual teleoperation."""

    config_class = BimanualGelloEEConfig
    name = "bimanual_gello_ee"

    def __init__(self, config: BimanualGelloEEConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = _make_gello_ee(config, "left", config.left_arm_config)
        self.right_arm = _make_gello_ee(config, "right", config.right_arm_config)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"l_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"r_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

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
                logger.exception("Failed to disconnect left GELLO EE during rollback")
            raise

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def seed_from_robot(self, robot) -> None:
        """Log the EE poses from the robot alongside the GELLO FK for comparison.

        Useful during bringup to verify the GELLO FK matches the robot's actual
        EE state before starting teleoperation.

        Args:
            robot: A connected ``BimanualFranka`` instance.
        """
        kin = robot.robot_manager.current_kinematic_state_batch(list(robot.active_arms))
        for arm_key, gello in (("l", self.left_arm), ("r", self.right_arm)):
            if arm_key not in kin:
                logger.warning("BimanualGelloEE.seed_from_robot: arm '%s' not found", arm_key)
                continue
            _, _, _, pos, rot, _ = kin[arm_key]
            robot_pos = np.asarray(pos)
            robot_rot = np.asarray(rot)
            action = gello.get_action()
            gello_pos = np.array([action["x"], action["y"], action["z"]])
            gello_rot = np.array([action["qx"], action["qy"], action["qz"], action["qw"]])
            logger.info(
                "Arm '%s' — robot EE pos=%s quat=%s | GELLO FK pos=%s quat=%s",
                arm_key, robot_pos.round(4), robot_rot.round(4),
                gello_pos.round(4), gello_rot.round(4),
            )

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left = self.left_arm.get_action()
        right = self.right_arm.get_action()
        return {
            **{f"l_{k}": v for k, v in left.items()},
            **{f"r_{k}": v for k, v in right.items()},
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

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
            raise RuntimeError(f"BimanualGelloEE disconnect errors: {details}")
