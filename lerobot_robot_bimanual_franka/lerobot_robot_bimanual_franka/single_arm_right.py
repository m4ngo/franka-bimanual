from .config_single_arm_right import SingleArmRightConfig
from .single_arm_franka import SingleArmFranka


class SingleArmRight(SingleArmFranka):
    config_class = SingleArmRightConfig
    name = "single_arm_right"
