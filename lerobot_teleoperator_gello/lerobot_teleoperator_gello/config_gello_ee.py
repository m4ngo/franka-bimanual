"""Configuration for the GELLO EE teleoperator.

Hardware parameters are identical to the standard GELLO (GelloLeaderFields).
A separate config type is registered so the EE variant is its own entry in
the teleoperator registry.
"""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_gello import GelloLeaderFields


@TeleoperatorConfig.register_subclass("gello_ee")
@dataclass
class GelloEEConfig(TeleoperatorConfig, GelloLeaderFields):
    """Standalone GELLO EE leader, registered as ``"gello_ee"``."""
