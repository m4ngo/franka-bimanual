"""Configuration dataclass for the GELLO teleoperator plugin.

Defines serial port settings, calibration home pose, per-joint signs, and
optional smoothing/async parameters used by :class:`Gello`.
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
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
