"""Configuration dataclasses for the SpaceMouse teleoperator plugin.

``SpaceMouseLeaderFields`` is a plain dataclass holding the per-device
hardware parameters.  It can be embedded inside ``BimanualSpaceMouseConfig``
without draccus recursing through the TeleoperatorConfig choice registry.

``SpaceMouseConfig`` composes the standard ``TeleoperatorConfig`` metadata
with ``SpaceMouseLeaderFields`` for the single-arm case.
"""

from dataclasses import dataclass, field

import franka_config as fc
from lerobot.teleoperators.config import TeleoperatorConfig


def _sm(path: str):
    return fc.teleop(f"spacemouse.{path}")


def _initial_rot_xyzw() -> tuple[float, float, float, float]:
    """config/teleop.yaml stores wxyz; this field is xyzw."""
    return fc.quat_wxyz_to_xyzw(tuple(_sm("initial_quat_wxyz")))


@dataclass
class SpaceMouseLeaderFields:
    """Hardware parameters for one SpaceMouse device (config/teleop.yaml)."""

    # Path to the hidraw node. Two SpaceMice appear as separate /dev/hidrawN.
    hidraw_path: str = field(default_factory=lambda: _sm("devices.right.hidraw_path"))

    # Position increment (metres) per control tick at full axis deflection.
    # pyspacemouse normalises axis values to [-1, 1].
    translation_scale: float = field(default_factory=lambda: _sm("translation_scale"))
    # Rotation increment (radians) per control tick at full axis deflection.
    rotation_scale: float = field(default_factory=lambda: _sm("rotation_scale"))

    # Fraction of full deflection treated as zero, with the remainder rescaled
    # so full deflection still reaches 1.0. The puck cross-talks badly; see
    # spacemouse._apply_deadzone.
    deadzone: float = field(default_factory=lambda: _sm("deadzone"))

    prefix: str = ""
    use_delta: bool = False
    # use_noise: bool = False
    noise_pos_scale: float = field(default_factory=lambda: _sm("noise_pos_scale"))
    noise_rot_scale: float = field(default_factory=lambda: _sm("noise_rot_scale"))

    # Initial EE Cartesian position [x, y, z] in metres. Override with
    # SpaceMouse.seed_state() to sync to the arm's actual EE on startup.
    initial_pos: tuple[float, float, float] = field(
        default_factory=lambda: tuple(_sm("initial_pos"))
    )
    # Initial EE orientation as a unit quaternion [qx, qy, qz, qw].
    initial_rot: tuple[float, float, float, float] = field(default_factory=_initial_rot_xyzw)

    # Per-axis sign trims in BASE frame, applied AFTER spacemouse.py's
    # LINEAR_DEVICE_TO_BASE / ANGULAR_DEVICE_TO_BASE. The device mounting lives
    # in those matrices, not here. Order: (x, y, z) / (roll, pitch, yaw).
    translation_signs: tuple[int, int, int] = field(
        default_factory=lambda: tuple(_sm("translation_signs"))
    )
    rotation_signs: tuple[int, int, int] = field(
        default_factory=lambda: tuple(_sm("rotation_signs"))
    )

    # Gripper travel limits (mm). Right button → open, left button → close.
    gripper_min_mm: float = field(default_factory=lambda: _sm("gripper_min_mm"))
    gripper_max_mm: float = field(default_factory=lambda: _sm("gripper_max_mm"))
    # Gripper target on connect, before any button press.
    initial_gripper_mm: float = field(default_factory=lambda: _sm("initial_gripper_mm"))

    @classmethod
    def for_side(cls, side: str, **overrides) -> "SpaceMouseLeaderFields":
        """Leader fields for the "left"/"right" SpaceMouse out of config/teleop.yaml."""
        return cls(hidraw_path=_sm(f"devices.{side}.hidraw_path"), **overrides)


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
