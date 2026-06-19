"""Configuration dataclasses for the SpaceMouse teleoperator plugin.

``SpaceMouseLeaderFields`` is a plain dataclass holding the per-device
hardware parameters.  It can be embedded inside ``BimanualSpaceMouseConfig``
without draccus recursing through the TeleoperatorConfig choice registry.

``SpaceMouseConfig`` composes the standard ``TeleoperatorConfig`` metadata
with ``SpaceMouseLeaderFields`` for the single-arm case.
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class SpaceMouseLeaderFields:
    """Hardware parameters for one SpaceMouse device."""

    # Path to the hidraw node. Two SpaceMice appear as separate /dev/hidrawN.
    hidraw_path: str = "/dev/hidraw4"

    # Position increment (metres) per control tick at full axis deflection.
    # pyspacemouse normalises axis values to [-1, 1].
    translation_scale: float = 0.02
    # Rotation increment (radians) per control tick at full axis deflection.
    rotation_scale: float = 0.05

    prefix: str = ""
    use_delta: bool = False

    # Initial EE Cartesian position [x, y, z] in metres. Override with
    # SpaceMouse.seed_state() to sync to the arm's actual EE on startup.
    initial_pos: tuple[float, float, float] = field(default_factory=lambda: (0.5, 0.0, 0.5))
    # Initial EE orientation as a unit quaternion [qx, qy, qz, qw].
    initial_rot: tuple[float, float, float, float] = field(default_factory=lambda: (1.0, 0.0, 0.0, 0.0))

    # Per-axis sign multipliers (+1 or -1) to match the robot's base frame.
    # Order: (x, y, z) for translation and (roll, pitch, yaw) for rotation.
    translation_signs: tuple[int, int, int] = field(default_factory=lambda: (1, -1, 1))
    rotation_signs: tuple[int, int, int] = field(default_factory=lambda: (1, 1, -1))

    # Gripper travel limits (mm). Right button → open, left button → close.
    gripper_min_mm: float = 0.1
    gripper_max_mm: float = 0.9
    # Gripper target on connect, before any button press.
    initial_gripper_mm: float = 0.9


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig, SpaceMouseLeaderFields):
    """Single SpaceMouse leader, registered as the ``"spacemouse"`` teleoperator type."""

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
