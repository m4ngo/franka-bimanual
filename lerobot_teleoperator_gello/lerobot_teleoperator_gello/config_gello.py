"""Configuration dataclasses for the GELLO teleoperator plugin.

GelloLeaderFields is a plain dataclass (not a TeleoperatorConfig subclass) so it can be
embedded in BimanualGelloConfig without draccus recursing through the choice registry.
GelloConfig composes it with the standard TeleoperatorConfig metadata.
"""

from dataclasses import dataclass, field
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class GelloLeaderFields:
    """Hardware and calibration parameters for one GELLO leader arm."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 57_600
    # Reference joint angles (rad) at the calibration home pose, one per motor in JOINT_NAMES order.
    calibration_position: list[float] = field(default_factory=lambda: [0, 0, 0, -1.57, 0, 1.57, 0, 0])
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, -1, 1, -1, 1, -1])
    gripper_travel_counts: int = 575  # closed-to-open travel in encoder counts
    smoothing: float = 0.99           # EMA alpha; 1.0 = no smoothing, 0.0 = max smoothing
    use_async: bool = True            # read motor states in a background thread


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig, GelloLeaderFields):
    """Standalone GELLO leader, registered as the ``"gello"`` teleoperator type."""

    # When set ("l" or "r"), Gello.get_action and Gello.action_features emit
    # keys prefixed with f"{side}_". Used for single-arm operation against a
    # bimanual follower (BimanualFranka with active_arms=(side,)).
    # Kept off GelloLeaderFields so BimanualGello's children always see None.
    side: str | None = None
