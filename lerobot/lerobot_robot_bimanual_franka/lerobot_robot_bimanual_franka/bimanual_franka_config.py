"""Configuration dataclass for the bimanual Franka robot plugin."""

from dataclasses import dataclass

from lerobot.robots import RobotConfig

# Arm identifier -> side. Ordered tuple preserves CLI / config ordering.
_VALID_ARMS: tuple[str, ...] = ("l", "r")


@RobotConfig.register_subclass("my_cool_robot")
@dataclass
class BimanualFrankaConfig(RobotConfig):
    l_server_ip: str
    l_robot_ip: str
    l_gripper_ip: str
    l_port: int
    r_server_ip: str
    r_robot_ip: str
    r_gripper_ip: str
    r_port: int
    use_ee_delta: bool
    active_arms: tuple[str, ...] = _VALID_ARMS

    def __post_init__(self):
        # Forward to parent __post_init__ if it defines one (RobotConfig may not).
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

        if not self.active_arms:
            raise ValueError("active_arms must contain at least one arm: 'l' and/or 'r'.")

        invalid = [arm for arm in self.active_arms if arm not in _VALID_ARMS]
        if invalid:
            raise ValueError(
                f"Invalid active arm identifiers: {invalid}. Allowed values are {_VALID_ARMS}."
            )

        # Deduplicate while preserving user-specified order.
        self.active_arms = tuple(dict.fromkeys(self.active_arms))
