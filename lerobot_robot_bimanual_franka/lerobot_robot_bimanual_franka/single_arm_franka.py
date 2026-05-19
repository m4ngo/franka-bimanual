from .bimanual_franka import BimanualFranka
from .config_single_arm_franka import SingleArmFrankaConfig


class SingleArmFranka(BimanualFranka):
    config_class = SingleArmFrankaConfig
    name = "single_arm_franka"