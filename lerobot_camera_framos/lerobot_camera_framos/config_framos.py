from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("framos_camera")
@dataclass
class FramosCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    serial_number: str = ""
    enable_color: bool = True
    enable_depth: bool = True
    color_width: int = 1280
    color_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    align_to: str = "color"
    color_format: str = "rgb8"
    depth_format: str = "z16"
    #: librealsense only accepts discrete FPS (typically 6/15/30/60/90 on D415e).
    #: If unset, FPS is snapped from `CameraConfig.fps` automatically in `FramosCamera`.
    streaming_fps: int | None = 30
    options: dict[str, float] = field(default_factory=dict)
    # Camera calibration defaults copied from frames/camera/matrices.txt.
    intrinsic_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (896.16738407, 0.0, 623.03492019),
        (0.0, 887.37167772, 353.61930747),
        (0.0, 0.0, 1.0),
    )
    distortion_coeffs: tuple[float, float, float, float, float] = (
        -3.49668586e-03,
        1.29032071e+00,
        -1.50837926e-03,
        -9.31210944e-05,
        -5.63551716e+00
    )
    r_cam_in_world: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (-0.84769453, 0.07912915, -0.52454986),
        (0.33176631, 0.85065211, -0.40782608),
        (0.41393851, -0.51973991, -0.74734553),
    )
    t_cam_in_world: tuple[float, float, float] = (0.47975414, 0.45897286, 0.94583442)
