"""Configuration for the bimanual GELLO teleoperator.

Uses GelloLeaderFields (plain dataclass) rather than GelloConfig (TeleoperatorConfig subclass)
to avoid draccus recursing through the choice registry when building the CLI parser.

Per-arm calibration files are derived from the bimanual id:
  ``{id}_left.json`` and ``{id}_right.json``
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_gello import GelloLeaderFields


@TeleoperatorConfig.register_subclass("bimanual_gello")
@dataclass
class BimanualGelloConfig(TeleoperatorConfig):
    """Pair of GELLO leaders driving a bimanual follower (left + right arm)."""

    left_arm_config: GelloLeaderFields = field(default_factory=GelloLeaderFields)
    right_arm_config: GelloLeaderFields = field(default_factory=GelloLeaderFields)
