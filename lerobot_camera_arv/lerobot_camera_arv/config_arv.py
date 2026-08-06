"""Aravis GigE camera config. Defaults come from config/cameras.yaml."""

from dataclasses import dataclass, field

import franka_config as fc
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("arv_camera")
@dataclass
class ArvCameraConfig(CameraConfig):
    name: str = ""
    ip: str = ""
    pixel_format: str = field(default_factory=lambda: fc.arv_defaults()["pixel_format"])

    @classmethod
    def for_camera(cls, key: str, **overrides) -> "ArvCameraConfig":
        """Build the config for `key` (cam_1 … cam_6) from config/cameras.yaml."""
        spec = fc.camera(key)
        if spec.type != "arv":
            raise TypeError(f"{key} is a {spec.type} camera, not arv")
        kwargs = dict(
            name=spec.name,
            ip=spec.ip,
            fps=spec.fps,
            width=spec.width,
            height=spec.height,
        )
        kwargs.update(overrides)
        return cls(**kwargs)
