from .bimanual_franka import BimanualFranka
from .config_single_arm_franka import SingleArmFrankaConfig


class SingleArmFranka(BimanualFranka):
    config_class = SingleArmFrankaConfig
    name = "single_arm_franka"

    def __init__(self, config: SingleArmFrankaConfig):
        # The dataset's robot_type, which baselines/common.py resolves as a rig profile.
        self.name = config.rig_profile
        super().__init__(config)
