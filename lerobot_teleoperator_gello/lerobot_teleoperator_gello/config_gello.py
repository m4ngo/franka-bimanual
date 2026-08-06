"""Configuration dataclasses for the GELLO teleoperator plugin.

GelloLeaderFields is a plain dataclass (not a TeleoperatorConfig subclass) so it can be
embedded in BimanualGelloConfig without draccus recursing through the choice registry.
GelloConfig composes it with the standard TeleoperatorConfig metadata.
"""

from dataclasses import dataclass, field

import franka_config as fc
from lerobot.teleoperators.config import TeleoperatorConfig


def _gello(path: str):
    return fc.teleop(f"gello.{path}")


@dataclass
class GelloLeaderFields:
    """Hardware and calibration parameters for one GELLO leader arm.

    Defaults come from config/teleop.yaml; the right-hand device is the
    single-arm default.
    """

    port: str = field(default_factory=lambda: _gello("devices.right.port"))
    baudrate: int = field(default_factory=lambda: _gello("baudrate"))
    # Reference joint angles (rad) at the calibration home pose, one per motor in JOINT_NAMES order.
    calibration_position: list[float] = field(default_factory=lambda: list(_gello("calibration_position")))
    joint_signs: list[int] = field(default_factory=lambda: list(_gello("joint_signs")))
    gripper_travel_counts: int = field(default_factory=lambda: _gello("gripper_travel_counts"))
    smoothing: float = field(default_factory=lambda: _gello("smoothing"))
    use_async: bool = field(default_factory=lambda: _gello("use_async"))
    use_noise: bool = field(default_factory=lambda: _gello("use_noise"))

    @classmethod
    def for_side(cls, side: str, **overrides) -> "GelloLeaderFields":
        """Leader fields for the "left"/"right" GELLO out of config/teleop.yaml."""
        return cls(port=_gello(f"devices.{side}.port"), **overrides)


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig, GelloLeaderFields):
    """Standalone GELLO leader, registered as the ``"gello"`` teleoperator type."""

    # When set ("l" or "r"), Gello.get_action and Gello.action_features emit
    # keys prefixed with f"{side}_". Used for single-arm operation against a
    # bimanual follower (BimanualFranka with active_arms=(side,)).
    # Kept off GelloLeaderFields so BimanualGello's children always see None.
    side: str | None = None
