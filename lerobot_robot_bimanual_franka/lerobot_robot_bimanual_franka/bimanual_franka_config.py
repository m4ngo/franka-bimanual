"""Configuration dataclass for the bimanual Franka robot plugin."""

from dataclasses import dataclass, field

from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore

# Arm identifier -> side. Ordered tuple preserves CLI / config ordering.
_VALID_ARMS: tuple[str, ...] = ("l", "r")

# _DEFAULT_CAMERAS: tuple[BimanualFrankaCameraConfig, ...] = (
#     BimanualFrankaCameraConfig(
#         name="gripper_bfs_23595723",
#         ip="192.168.0.142",
#         serial_number="BFS_23595723",
#     ),
#     BimanualFrankaCameraConfig(
#         name="workspace_framos_d71",
#         ip="192.168.0.116",
#         serial_number="FRAMOS_D71",
#     ),
#     BimanualFrankaCameraConfig(
#         name="gripper_bfs_23595719",
#         ip="192.168.1.138",
#         serial_number="BFS_23595719",
#     ),
#     BimanualFrankaCameraConfig(
#         name="gripper_bfs_23595720",
#         ip="192.168.1.139",
#         serial_number="BFS_23595720",
#     ),
#     BimanualFrankaCameraConfig(
#         name="gripper_bfs_23595724",
#         ip="192.168.1.143",
#         serial_number="BFS_23595724",
#     ),
#     BimanualFrankaCameraConfig(
#         name="workspace_framos_d63",
#         ip="192.168.1.102",
#         serial_number="FRAMOS_D63",
#     ),
# )


@RobotConfig.register_subclass("bimanual_franka")
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
    cameras: dict[str, ArvCameraConfig] = field(
        default_factory=lambda: {
            "cam_1": ArvCameraConfig(
                name="gripper_bfs_23595723",
                ip="192.168.0.142",
                fps=20,
                width=240,
                height=150,
            ),
            "cam_2": ArvCameraConfig(
                name="workspace_framos_d71",
                ip="192.168.0.116",
                fps=20,
                width=240,
                height=135,
            ),
            "cam_3": ArvCameraConfig(
                name="gripper_bfs_23595719",
                ip="192.168.1.138",
                fps=20,
                width=240,
                height=150,
            ),
            "cam_4": ArvCameraConfig(
                name="gripper_bfs_23595720",
                ip="192.168.1.139",
                fps=20,
                width=240,
                height=150,
            ),
            "cam_5": ArvCameraConfig(
                name="gripper_bfs_23595724",
                ip="192.168.1.143",
                fps=20,
                width=240,
                height=150,
            ),
            "cam_6": ArvCameraConfig(
                name="workspace_framos_d63",
                ip="192.168.1.102",
                fps=20,
                width=240,
                height=135,
            ),
        }
    )

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

        camera_names = [camera.name for camera in self.cameras.values()]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError("Camera names must be unique.")
