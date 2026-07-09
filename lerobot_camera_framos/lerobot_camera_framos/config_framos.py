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
        (914.947789, 0.0, 647.36886029),
        (0.0, 915.51476285, 374.96911383),
        (0.0, 0.0, 1.0),
    )
    distortion_coeffs: tuple[float, float, float, float, float] = (
        9.37354420e-02,
        -2.74125058e-01,
        9.12541074e-06,
        1.35710217e-04,
        1.23260996e-01,
    )
    r_cam_in_world: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (-0.9370905, -0.03893257, 0.34690871),
        (-0.21090789, 0.85502599, -0.47375987),
        (-0.27817128, -0.51712166, -0.80944792),
    )
    t_cam_in_world: tuple[float, float, float] = (-0.33212947, 0.58415179, 0.94429804)
