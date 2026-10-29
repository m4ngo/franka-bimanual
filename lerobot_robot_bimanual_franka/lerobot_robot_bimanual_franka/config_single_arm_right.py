"""`--robot.type=single_arm_right`: SingleArmFrankaConfig pinned to the
`single_arm_right` profile, kept so datasets and commands that name this type
still resolve. The arm identity lives in config/rig.yaml, never here.
"""

from dataclasses import dataclass

from lerobot.robots import RobotConfig

from .config_single_arm_franka import SingleArmFrankaConfig


@RobotConfig.register_subclass("single_arm_right")
@dataclass
class SingleArmRightConfig(SingleArmFrankaConfig):
    rig_profile: str = "single_arm_right"
