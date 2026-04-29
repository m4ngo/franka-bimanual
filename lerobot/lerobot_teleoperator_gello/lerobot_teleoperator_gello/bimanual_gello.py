"""Bimanual GELLO teleoperator: two single-arm :class:`Gello` leaders fused.

Mirrors the built-in ``bi_so_leader`` design (see
``lerobot.teleoperators.bi_so_leader``) so the bimanual rig can be driven via
the standard ``lerobot-teleoperate`` / ``lerobot-record`` entry points:

```
lerobot-teleoperate \\
    --robot.type=bimanual_franka \\
    ...bimanual robot args... \\
    --teleop.type=bimanual_gello \\
    --teleop.id=gello_teleop \\
    --teleop.left_arm_config.port=/dev/ttyUSB0 \\
    --teleop.right_arm_config.port=/dev/ttyUSB1
```

Per-arm calibration files default to ``{teleop.id}_left.json`` and
``{teleop.id}_right.json`` (same convention as ``bi_so_leader``). Action keys
are emitted with ``l_`` / ``r_`` prefixes (e.g. ``l_joint_1`` … ``r_gripper``)
to match the schema consumed by
:class:`lerobot_robot_bimanual_franka.BimanualFranka`.
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


def _to_gello_config(
    parent: BimanualGelloConfig, side: str, child: GelloLeaderFields
) -> GelloConfig:
    """Promote per-arm fields to a full :class:`GelloConfig` for one sub-leader."""
    child_id = f"{parent.id}_{side}" if parent.id else None
    return GelloConfig(
        id=child_id,
        calibration_dir=parent.calibration_dir,
        **asdict(child),
    )


class BimanualGello(Teleoperator):
    """Pair of :class:`Gello` leaders presented as a single bimanual teleop."""

    config_class = BimanualGelloConfig
    name = "bimanual_gello"

    def __init__(self, config: BimanualGelloConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = Gello(_to_gello_config(config, "left", config.left_arm_config))
        self.right_arm = Gello(_to_gello_config(config, "right", config.right_arm_config))

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

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
            # Roll back the left arm so the dynamixel buses do not stay half-open.
            try:
                self.left_arm.disconnect()
            except Exception:
                logger.exception("Failed to disconnect left GELLO during bring-up rollback")
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
        # No force feedback on GELLO yet (matches Gello.send_feedback).
        raise NotImplementedError

    def disconnect(self) -> None:
        # Tolerate partial connection state so we never leak a half-open bus.
        errors: list[tuple[str, BaseException]] = []
        for label, arm in (("left", self.left_arm), ("right", self.right_arm)):
            if not arm.is_connected:
                continue
            try:
                arm.disconnect()
            except Exception as exc:  # noqa: BLE001
                errors.append((label, exc))

        if errors:
            details = ", ".join(f"{label}: {exc}" for label, exc in errors)
            raise RuntimeError(f"BimanualGello disconnect errors: {details}")
