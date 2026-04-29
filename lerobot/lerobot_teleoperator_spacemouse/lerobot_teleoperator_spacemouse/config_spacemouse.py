"""Configuration dataclass for the SpaceMouse teleoperator plugin.

Defines the hidraw device path, axis scaling and signs, and gripper travel
limits used by :class:`SpaceMouse`.
"""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig):
    # Path to the hidraw node for the SpaceMouse. Two SpaceMice plugged in
    # at the same time appear as separate /dev/hidrawN nodes.
    hidraw_path: str = "/dev/hidraw4"

    # Linear velocity (m/s) at full axis deflection. pyspacemouse already
    # normalizes axis values to [-1, 1], so this is the achievable peak.
    translation_scale: float = 0.15
    # Angular velocity (rad/s) at full axis deflection.
    rotation_scale: float = 0.6

    # Per-axis sign multipliers applied to the SpaceMouse output before
    # scaling. Each entry must be +1 or -1. Use these to flip an axis
    # whose physical orientation does not match the robot's base frame
    # convention. Order is (x, y, z) for translation and (roll, pitch,
    # yaw) for rotation, matching SpaceMouse.AXIS_NAMES.
    translation_signs: tuple[int, int, int] = (1, -1, 1)
    rotation_signs: tuple[int, int, int] = (1, 1, -1)

    # Gripper travel limits forwarded as the action's "gripper" value (mm).
    # The right button drives the target to gripper_max_mm (open); the left
    # button drives it to gripper_min_mm (close).
    gripper_min_mm: float = 10.0
    gripper_max_mm: float = 100.0
    # Gripper target on connect, before any button press.
    initial_gripper_mm: float = 100.0

    def __post_init__(self) -> None:
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()
        for name, signs in (
            ("translation_signs", self.translation_signs),
            ("rotation_signs", self.rotation_signs),
        ):
            if len(signs) != 3 or any(s not in (-1, 1) for s in signs):
                raise ValueError(
                    f"SpaceMouseConfig.{name} must be a 3-tuple of +1/-1, got {signs!r}"
                )
