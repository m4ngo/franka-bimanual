"""FRAMOS D400e camera config.

All defaults come from config/cameras.yaml via franka_config; nothing is
hardcoded here. Per-camera intrinsics/extrinsics are injected by
`FramosCameraConfig.for_camera("cam_2")`, which is what the robot configs use.
Extrinsics are stored world-relative (see config/world.yaml).
"""

from dataclasses import dataclass, field

import franka_config as fc
from lerobot.cameras.configs import CameraConfig

_IDENTITY_R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _default(key, fallback=None):
    return fc.framos_defaults().get(key, fallback)


@CameraConfig.register_subclass("framos_camera")
@dataclass
class FramosCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    serial_number: str = ""
    enable_color: bool = True
    enable_depth: bool = True
    color_width: int = field(default_factory=lambda: _default("color_width"))
    color_height: int = field(default_factory=lambda: _default("color_height"))
    depth_width: int = field(default_factory=lambda: _default("depth_width"))
    depth_height: int = field(default_factory=lambda: _default("depth_height"))
    align_to: str = field(default_factory=lambda: _default("align_to"))
    color_format: str = field(default_factory=lambda: _default("color_format"))
    depth_format: str = field(default_factory=lambda: _default("depth_format"))
    #: librealsense only accepts discrete FPS (typically 6/15/30/60/90 on D415e).
    #: If unset, FPS is snapped from `CameraConfig.fps` automatically in `FramosCamera`.
    streaming_fps: int | None = field(default_factory=lambda: fc.camera_stream_fps())
    options: dict[str, float] = field(default_factory=dict)
    # Populated from config/cameras.yaml by for_camera(); the bare defaults are
    # deliberately inert so an unconfigured camera can't silently deproject into
    # a plausible-looking but wrong world pose.
    intrinsic_matrix: tuple[tuple[float, float, float], ...] = _IDENTITY_R
    distortion_coeffs: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    #: Camera -> WORLD rotation and translation (world.yaml frame).
    r_cam_in_world: tuple[tuple[float, float, float], ...] = _IDENTITY_R
    t_cam_in_world: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def for_camera(cls, key: str, **overrides) -> "FramosCameraConfig":
        """Build the config for `key` (cam_1 … cam_6) from config/cameras.yaml."""
        spec = fc.camera(key)
        if not spec.is_framos:
            raise TypeError(f"{key} is a {spec.type} camera, not framos")
        kwargs = dict(
            name=spec.name,
            ip=spec.ip,
            serial_number=spec.serial_number,
            fps=spec.fps,
            width=spec.width,
            height=spec.height,
        )
        if spec.calibration is not None:
            pose = spec.calibration.cam_in_world
            kwargs.update(
                intrinsic_matrix=spec.calibration.intrinsic_matrix,
                distortion_coeffs=spec.calibration.distortion_coeffs,
                r_cam_in_world=tuple(tuple(float(v) for v in row) for row in pose.rotation),
                t_cam_in_world=tuple(float(v) for v in pose.translation),
            )
        kwargs.update(overrides)
        return cls(**kwargs)
