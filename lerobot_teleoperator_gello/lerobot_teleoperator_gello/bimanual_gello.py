"""Bimanual GELLO teleoperator: two single-arm Gello leaders fused into one.

Action keys are emitted with ``l_`` / ``r_`` prefixes to match the schema
expected by BimanualFranka (e.g. ``l_joint_1`` … ``r_gripper``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_bimanual_gello import BimanualGelloConfig
from .config_gello import GelloConfig, GelloLeaderFields
from .gello import Gello

logger = logging.getLogger(__name__)


def _to_gello_config(parent: BimanualGelloConfig, side: str, child: GelloLeaderFields) -> GelloConfig:
    return GelloConfig(
        id=f"{parent.id}_{side}" if parent.id else None,
        calibration_dir=parent.calibration_dir,
        **asdict(child),
    )


class BimanualGello(Teleoperator):
    """Pair of Gello leaders presented as a single bimanual teleoperator."""

    config_class = BimanualGelloConfig
    name = "bimanual_gello"

    def __init__(self, config: BimanualGelloConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = Gello(_to_gello_config(config, "left", config.left_arm_config))
        self.right_arm = Gello(_to_gello_config(config, "right", config.right_arm_config))

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
                logger.exception("Failed to disconnect left GELLO during rollback")
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
            raise RuntimeError(f"BimanualGello disconnect errors: {details}")
