"""Configuration for the bimanual GELLO EE teleoperator.

Uses GelloLeaderFields (plain dataclass) rather than GelloEEConfig
(TeleoperatorConfig subclass) to avoid draccus recursing through the choice
registry when building the CLI parser.

Per-arm calibration files are derived from the bimanual id:
  ``{id}_left.json`` and ``{id}_right.json``
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_gello import GelloLeaderFields


@TeleoperatorConfig.register_subclass("bimanual_gello_ee")
@dataclass
class BimanualGelloEEConfig(TeleoperatorConfig):
    """Pair of GELLO EE leaders driving a bimanual follower (left + right arm)."""

    left_arm_config: GelloLeaderFields = field(default_factory=GelloLeaderFields)
    right_arm_config: GelloLeaderFields = field(default_factory=GelloLeaderFields)
