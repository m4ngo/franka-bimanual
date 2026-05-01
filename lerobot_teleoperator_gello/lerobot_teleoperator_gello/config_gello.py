"""Configuration dataclasses for the GELLO teleoperator plugin.

Defines two pieces:

* :class:`GelloLeaderFields` — a plain dataclass holding the per-arm GELLO
  hardware/calibration parameters. This is the type used for
  ``BimanualGelloConfig.left_arm_config`` / ``right_arm_config`` so that
  draccus does not treat it as a polymorphic ``TeleoperatorConfig`` choice
  (which would otherwise recurse infinitely through the choice registry).
* :class:`GelloConfig` — the registered ``"gello"`` teleoperator choice. It
  composes the leader fields with the standard ``id`` / ``calibration_dir``
  metadata from :class:`TeleoperatorConfig`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class GelloLeaderFields:
    """Hardware + calibration parameters for one GELLO leader arm.

    This is intentionally a plain dataclass (not a :class:`TeleoperatorConfig`
    subclass) so it can be embedded in higher-level configs (such as
    :class:`BimanualGelloConfig`) without being treated as a polymorphic
    choice by draccus.
    """

    # Serial port that the Dynamixel bus is attached to.
    port: str = "/dev/ttyUSB0"
    baudrate: int = 57_600

    # Reference joint angles (radians) at the calibration home pose. One entry
    # per motor in the same order as Gello.JOINT_NAMES.
    calibration_position: list[float] = field(
        default_factory=lambda: [0, 0, 0, -1.57, 0, 1.57, 0, 0]
    )
    # Per-motor direction: +1 to follow, -1 to invert.
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, -1, 1, -1, 1, -1])
    # Full closed-to-open travel of the gripper servo, in motor counts.
    gripper_travel_counts: int = 575

    # EMA smoothing factor in [0, 1]. 1.0 = no smoothing (instant update);
    # values closer to 0 smooth jitter but add latency.
    smoothing: float = 0.99
    # If True, read motor states in a background thread to hide USB latency.
    use_async: bool = True
    # calibration_dir: Path = Path("~/franka_ws/config/gello")  # Override default


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig, GelloLeaderFields):
    """Standalone GELLO leader, exposed as the ``"gello"`` teleoperator type."""
