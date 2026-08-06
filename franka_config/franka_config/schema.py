"""Typed accessors over config/*.yaml.

Everything is cached, so repeated calls in a control loop are free. Values are
returned as plain tuples/floats so they can be dropped straight into the
LeRobot dataclass `default_factory` slots without sharing mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ._loader import config_dir, get, repo_root, section
from .transforms import Pose

# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def calib_origin_in_world() -> Pose:
    """Calibration-board frame expressed in world."""
    node = get("world.calib_origin_in_world")
    return Pose.from_quat_wxyz(node["quat_wxyz"], node["translation_m"])


@lru_cache(maxsize=None)
def robot_base_in_world(arm: str) -> Pose:
    """Robot base frame expressed in world: p_world = R @ p_base + t."""
    table = get("world.robot_base_in_world")
    if arm not in table:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(table)}")
    node = table[arm]
    return Pose.from_quat_wxyz(node["quat_wxyz"], node["translation_m"])


def robot_base_in_world_verified(arm: str) -> bool:
    return bool(get("world.robot_base_in_world")[arm].get("verified", False))


@lru_cache(maxsize=1)
def worktable_height_m() -> float:
    """Table surface in WORLD frame Z (what the safety brake compares against)."""
    return float(get("world.worktable.height_m"))


@lru_cache(maxsize=1)
def sim_world_alignment() -> Pose:
    """Real world -> sim world. Identity when the alignment is disabled."""
    node = get("world.sim_alignment")
    return Pose.from_quat_wxyz(node["world_quat_wxyz"], node["world_translation_m"])


@lru_cache(maxsize=1)
def sim_ee_convention() -> tuple[np.ndarray, np.ndarray]:
    """(rotvec_rad, pos_tool_m) mapping franka_fk output to the sim obs convention."""
    node = get("world.sim_alignment.ee_convention")
    return (
        np.asarray(node["rotvec_rad"], dtype=np.float64),
        np.asarray(node["pos_tool_m"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperSpec:
    kind: str
    ip: str

    @property
    def is_wsg(self) -> bool:
        return self.kind == "wsg"


@dataclass(frozen=True)
class EESphere:
    """Collision sphere approximating the end effector.

    `center_tool_m` is in the TOOL frame and rotates with the EE, so the
    volume stays correct when the gripper is tilted. The worktable brake uses
    the sphere's lowest point: `center_world_z - radius_m`.
    """

    center_tool_m: tuple[float, float, float]
    radius_m: float

    @classmethod
    def from_node(cls, node: dict) -> "EESphere":
        center = tuple(float(v) for v in node.get("center_tool_m", (0.0, 0.0, 0.0)))
        if len(center) != 3:
            raise ValueError(f"ee_sphere.center_tool_m must have 3 entries, got {center!r}")
        radius = float(node["radius_m"])
        if radius < 0.0:
            raise ValueError(f"ee_sphere.radius_m must be non-negative, got {radius}")
        return cls(center_tool_m=center, radius_m=radius)


@lru_cache(maxsize=1)
def default_ee_sphere() -> EESphere:
    """Shared EE collision sphere from control.yaml."""
    return EESphere.from_node(get("control.worktable_brake.ee_sphere"))


@lru_cache(maxsize=None)
def ee_sphere(arm: str) -> EESphere:
    """EE collision sphere for one arm — its own override, else the shared default."""
    node = get("arms.arms")[arm].get("ee_sphere")
    return EESphere.from_node(node) if node else default_ee_sphere()


@dataclass(frozen=True)
class ArmSpec:
    name: str
    nuc_host: str
    nuc_user: str
    server_ip: str
    robot_ip: str
    rpyc_port: int
    gripper_rpyc_port: int
    gripper: GripperSpec
    default_key: str
    ee_sphere: EESphere

    @property
    def ssh_target(self) -> str:
        return f"{self.nuc_user}@{self.server_ip}"


@lru_cache(maxsize=None)
def arm(name: str) -> ArmSpec:
    table = get("arms.arms")
    if name not in table:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(table)}")
    node = table[name]
    g = node["gripper"]
    return ArmSpec(
        name=name,
        nuc_host=node["nuc_host"],
        nuc_user=node["nuc_user"],
        server_ip=node["server_ip"],
        robot_ip=node["robot_ip"],
        rpyc_port=int(node["rpyc_port"]),
        gripper_rpyc_port=int(node["gripper_rpyc_port"]),
        gripper=GripperSpec(kind=g["kind"], ip=g["ip"]),
        default_key=node["default_key"],
        ee_sphere=ee_sphere(name),
    )


def arm_names() -> tuple[str, ...]:
    return tuple(get("arms.arms"))


@lru_cache(maxsize=1)
def home_poses_dir() -> Path:
    return repo_root() / get("arms.home_poses.dir")


@lru_cache(maxsize=1)
def default_home_pose_name() -> str:
    return str(get("arms.home_poses.default"))


def load_home_pose(name: str | None = None) -> dict[str, Any]:
    """Load home_poses/<name>.json. The JSON files are the only source of home q."""
    import json

    name = name or default_home_pose_name()
    path = home_poses_dir() / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no home pose {name!r} at {path}")
    return json.loads(path.read_text())


def home_q(name: str | None = None, key: str = "r") -> np.ndarray:
    """Joint targets for one arm key out of a saved home pose."""
    pose = load_home_pose(name)
    field = f"{key}_q"
    if field not in pose:
        raise KeyError(f"home pose {name!r} has no {field!r} (keys: {sorted(pose)})")
    return np.asarray(pose[field], dtype=np.float64)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic_matrix: tuple[tuple[float, ...], ...]
    distortion_coeffs: tuple[float, ...]
    cam_in_calib: Pose
    source: str

    @property
    def cam_in_world(self) -> Pose:
        return calib_origin_in_world().compose(self.cam_in_calib)


@dataclass(frozen=True)
class CameraSpec:
    key: str
    type: str
    name: str
    ip: str
    role: str
    mount: str
    width: int
    height: int
    fps: int
    serial_number: str = ""
    depth: bool = False
    calibration: CameraCalibration | None = None
    extra: dict[str, Any] | None = None

    @property
    def is_framos(self) -> bool:
        return self.type == "framos"


def _camera_defaults(cam_type: str) -> dict[str, Any]:
    defaults = get("cameras.defaults")
    merged = {k: v for k, v in defaults.items() if not isinstance(v, dict)}
    merged.update(defaults.get(cam_type, {}))
    return merged


@lru_cache(maxsize=None)
def camera(key: str) -> CameraSpec:
    table = get("cameras.cameras")
    if key not in table:
        raise KeyError(f"unknown camera {key!r}; known: {sorted(table)}")
    node = dict(table[key])
    cam_type = node.pop("type")
    defaults = _camera_defaults(cam_type)

    calib_node = node.pop("calibration", None)
    calib = None
    if calib_node:
        calib = CameraCalibration(
            intrinsic_matrix=tuple(tuple(float(v) for v in row)
                                   for row in calib_node["intrinsic_matrix"]),
            distortion_coeffs=tuple(float(v) for v in calib_node["distortion_coeffs"]),
            cam_in_calib=Pose.from_matrix_translation(
                calib_node["r_cam_in_calib"], calib_node["t_cam_in_calib"]
            ),
            source=str(calib_node.get("source", "")),
        )

    extra = {k: v for k, v in defaults.items()
             if k not in ("width", "height", "fps") and k not in node}
    return CameraSpec(
        key=key,
        type=cam_type,
        name=node.pop("name"),
        ip=node.pop("ip"),
        role=node.pop("role", "unknown"),
        mount=node.pop("mount", "unknown"),
        width=int(node.pop("width", defaults["width"])),
        height=int(node.pop("height", defaults["height"])),
        fps=int(node.pop("fps", defaults["fps"])),
        serial_number=str(node.pop("serial_number", "")),
        depth=bool(node.pop("depth", False)),
        calibration=calib,
        extra=extra,
    )


def camera_keys() -> tuple[str, ...]:
    return tuple(get("cameras.cameras"))


def camera_ips() -> tuple[str, ...]:
    return tuple(camera(k).ip for k in camera_keys())


def framos_defaults() -> dict[str, Any]:
    return dict(get("cameras.defaults.framos"))


def arv_defaults() -> dict[str, Any]:
    return dict(get("cameras.defaults.arv"))


# ---------------------------------------------------------------------------
# Rig profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    arms: dict[str, str]           # key prefix -> arm name in arms.yaml
    cameras: tuple[str, ...]
    depth_cameras: tuple[str, ...]
    depth: bool
    depth_center_arm: str
    control_mode: str
    #: Which physical leader in teleop.yaml the operator holds. Independent of
    #: which arm the profile drives; None for profiles with no teleoperator.
    teleop_device: str | None = None

    def arm_spec(self, key: str) -> ArmSpec:
        return arm(self.arms[key])

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.arms)


@lru_cache(maxsize=None)
def profile(name: str) -> ProfileSpec:
    table = get("rig.profiles")
    if name not in table:
        raise KeyError(f"unknown rig profile {name!r}; known: {sorted(table)}")
    node = table[name]
    return ProfileSpec(
        name=name,
        arms=dict(node["arms"]),
        cameras=tuple(node["cameras"]),
        depth_cameras=tuple(node.get("depth_cameras", ())),
        depth=bool(node.get("depth", False)),
        depth_center_arm=str(node.get("depth_center_arm", next(iter(node["arms"])))),
        control_mode=str(node["control_mode"]),
        teleop_device=node.get("teleop_device"),
    )


def profile_names() -> tuple[str, ...]:
    return tuple(get("rig.profiles"))


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


def control(path: str, default: Any = ...) -> Any:
    """Shorthand for get("control.<path>")."""
    return get(f"control.{path}", default)


@lru_cache(maxsize=1)
def control_fps() -> int:
    return int(get("control.rates.control_fps"))


@lru_cache(maxsize=1)
def camera_stream_fps() -> int:
    return int(get("control.rates.camera_stream_fps"))


@lru_cache(maxsize=1)
def home_fps() -> int:
    value = get("control.rates.home_fps")
    return control_fps() if value is None else int(value)


@lru_cache(maxsize=1)
def num_joints() -> int:
    return int(get("control.franka.num_joints"))


def teleop(path: str, default: Any = ...) -> Any:
    return get(f"teleop.{path}", default)


def policy(path: str, default: Any = ...) -> Any:
    return get(f"policy.{path}", default)


def calibration(path: str, default: Any = ...) -> Any:
    return get(f"calibration.{path}", default)


__all__ = [
    "ArmSpec", "CameraCalibration", "CameraSpec", "EESphere", "GripperSpec", "Pose",
    "ProfileSpec", "arm", "arm_names", "arv_defaults", "calib_origin_in_world",
    "calibration", "camera", "camera_ips", "camera_keys", "camera_stream_fps",
    "config_dir", "control", "control_fps", "default_ee_sphere",
    "default_home_pose_name", "ee_sphere", "framos_defaults", "home_fps",
    "home_poses_dir", "home_q", "load_home_pose", "num_joints", "policy", "profile",
    "profile_names", "repo_root", "robot_base_in_world",
    "robot_base_in_world_verified", "section", "sim_ee_convention",
    "sim_world_alignment", "teleop", "worktable_height_m",
]
