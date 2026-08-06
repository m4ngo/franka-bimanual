"""Bridge between config/*.yaml and the LeRobot dataclass configs.

Keeps `franka_config` free of any LeRobot/camera-plugin imports while giving
both robot configs one place to turn a rig profile into concrete camera and
arm settings.
"""

from __future__ import annotations

from typing import Any

import franka_config as fc  # type: ignore
from lerobot.cameras import CameraConfig
from lerobot_camera_arv import ArvCameraConfig  # type: ignore
from lerobot_camera_framos import FramosCameraConfig  # type: ignore

_CONFIG_CLASSES = {"arv": ArvCameraConfig, "framos": FramosCameraConfig}


def make_camera_config(key: str, **overrides) -> CameraConfig:
    """Build the plugin config for one camera id out of config/cameras.yaml."""
    spec = fc.camera(key)
    cls = _CONFIG_CLASSES.get(spec.type)
    if cls is None:
        raise TypeError(f"unsupported camera type {spec.type!r} for {key}")
    return cls.for_camera(key, **overrides)


def profile_cameras(profile_name: str) -> dict[str, CameraConfig]:
    """Observation-key -> camera config for a rig profile's colour cameras."""
    profile = fc.profile(profile_name)
    return {key: make_camera_config(key) for key in profile.cameras}


def profile_depth_cameras(profile_name: str) -> dict[str, CameraConfig]:
    """Observation-key -> depth-only camera config for a rig profile."""
    profile = fc.profile(profile_name)
    return {
        key: make_camera_config(key, enable_color=False)
        for key in profile.depth_cameras
    }


def profile_arm_fields(profile_name: str) -> dict[str, Any]:
    """`{key}_server_ip` / `{key}_robot_ip` / `{key}_gripper_ip` / `{key}_port`
    for every arm the profile exposes, ready to splat into a RobotConfig."""
    profile = fc.profile(profile_name)
    fields: dict[str, Any] = {}
    for key, arm_name in profile.arms.items():
        spec = fc.arm(arm_name)
        fields[f"{key}_server_ip"] = spec.server_ip
        fields[f"{key}_robot_ip"] = spec.robot_ip
        fields[f"{key}_gripper_ip"] = spec.gripper.ip
        fields[f"{key}_port"] = spec.rpyc_port
    return fields


def profile_arm_name(profile_name: str, key: str) -> str:
    """Physical arm ("left"/"right") behind an exposed key prefix."""
    return fc.profile(profile_name).arms[key]
