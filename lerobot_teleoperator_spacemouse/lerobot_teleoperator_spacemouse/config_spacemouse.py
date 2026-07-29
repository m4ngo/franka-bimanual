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

    # Metres / radians per control tick at full axis deflection. These are
    # robosuite osc_pose.json's output_max, so full deflection is exactly a
    # normalized +/-1 policy action -- teleop and policy drive the controller
    # through identical units. Downstream clip_delta enforces the same bound, so
    # raising these past 0.05/0.5 only saturates earlier, it does not go faster.
    translation_scale: float = 0.05
    rotation_scale: float = 0.5

    # Fraction of full deflection treated as zero, with the remainder rescaled
    # so full deflection still reaches 1.0. The puck cross-talks: twisting to yaw
    # drives the linear axes to ~0.23, which is ~1.2 cm/tick of translation the
    # operator never commanded.
    deadzone: float = 0.08

    prefix: str = ""
    use_delta: bool = False
    # use_noise: bool = False
    noise_pos_scale: float = 0.05   # metres, added to position output each step
    noise_rot_scale: float = 0.03    # radians (axis-angle), added to rotation output each step

    # Initial EE Cartesian position [x, y, z] in metres. Override with
    # SpaceMouse.seed_state() to sync to the arm's actual EE on startup.
    initial_pos: tuple[float, float, float] = field(default_factory=lambda: (0.5, 0.0, 0.5))
    # Initial EE orientation as a unit quaternion [qx, qy, qz, qw].
    initial_rot: tuple[float, float, float, float] = field(default_factory=lambda: (1.0, 0.0, 0.0, 0.0))

    # Per-axis sign trims in BASE frame (x, y, z), applied after
    # spacemouse.DEVICE_TO_BASE. The device mounting lives in that matrix, not
    # here -- these exist only to flip an axis without editing it. Defaults are
    # identity because the matrix already encodes this rig's mounting.
    translation_signs: tuple[int, int, int] = field(default_factory=lambda: (1, 1, 1))
    # Yaw is inverted on this puck; the controller itself is correctly signed.
    rotation_signs: tuple[int, int, int] = field(default_factory=lambda: (1, 1, -1))

    # Gripper travel limits (mm). Right button → open, left button → close.
    gripper_min_mm: float = -1.0
    gripper_max_mm: float = 1.0
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
