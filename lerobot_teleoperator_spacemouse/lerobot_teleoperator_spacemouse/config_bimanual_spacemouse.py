"""Configuration for the bimanual SpaceMouse teleoperator.

Uses ``SpaceMouseLeaderFields`` (plain dataclass) rather than
``SpaceMouseConfig`` (TeleoperatorConfig subclass) to avoid draccus recursing
through the choice registry when building the CLI parser.

The two SpaceMice sit on different hidraw nodes; the paths come from
config/teleop.yaml (spacemouse.devices).
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_spacemouse import SpaceMouseLeaderFields


@TeleoperatorConfig.register_subclass("bimanual_spacemouse")
@dataclass
class BimanualSpaceMouseConfig(TeleoperatorConfig):
    """Pair of SpaceMouse leaders driving a bimanual follower (left + right arm)."""

    left_arm_config: SpaceMouseLeaderFields = field(
        default_factory=lambda: SpaceMouseLeaderFields.for_side("left")
    )
    right_arm_config: SpaceMouseLeaderFields = field(
        default_factory=lambda: SpaceMouseLeaderFields.for_side("right")
    )
