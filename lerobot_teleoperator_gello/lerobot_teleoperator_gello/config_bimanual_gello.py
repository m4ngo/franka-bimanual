"""Configuration dataclass for the bimanual GELLO teleoperator plugin.

Pairs two :class:`GelloLeaderFields` instances (one per arm) so that
``lerobot-teleoperate`` and ``lerobot-record`` can drive a bimanual Franka
with two GELLO leaders through the standard
``--teleop.type=bimanual_gello`` CLI surface.

Note: the per-arm field type is :class:`GelloLeaderFields` (a plain dataclass)
rather than :class:`GelloConfig` (a ``TeleoperatorConfig`` choice). Embedding
a choice-registry type here would cause draccus to recurse through every
registered teleop type — including this one — and infinite-loop while
building the CLI parser.
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_gello import GelloLeaderFields


@TeleoperatorConfig.register_subclass("bimanual_gello")
@dataclass
class BimanualGelloConfig(TeleoperatorConfig):
    """Pair of GELLO leaders driving a bimanual follower (left + right arm).

    The bimanual ``id`` is appended with ``_left`` / ``_right`` to produce the
    per-arm calibration identifiers used by each underlying :class:`Gello`.
    For example, ``--teleop.id=gello_teleop`` resolves to calibration files
    ``gello_teleop_left.json`` and ``gello_teleop_right.json`` under
    ``$HF_LEROBOT_HOME/calibration/teleoperators/gello/``.
    """

    left_arm_config: GelloLeaderFields = field(default_factory=GelloLeaderFields)
    right_arm_config: GelloLeaderFields = field(default_factory=GelloLeaderFields)
