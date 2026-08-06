"""Centralized environment configuration for the TRI bimanual FR3 workspace.

Every world/robot/camera/control constant lives in `config/*.yaml` at the
workspace root. This package is the only way code should read them.

    from franka_config import arm, camera, control_fps, robot_base_in_world

Import-time cost is one YAML parse per file, cached process-wide.
`FRANKA_CONFIG_DIR` overrides which directory is read.
"""

from ._loader import all_sections, config_dir, get, reload, repo_root, section
from .schema import (
    ArmSpec,
    CameraCalibration,
    CameraSpec,
    EESphere,
    GripperSpec,
    ProfileSpec,
    arm,
    arm_names,
    arv_defaults,
    calib_origin_in_world,
    calibration,
    camera,
    camera_ips,
    camera_keys,
    camera_stream_fps,
    control,
    control_fps,
    default_ee_sphere,
    default_home_pose_name,
    ee_sphere,
    framos_defaults,
    home_fps,
    home_poses_dir,
    home_q,
    load_home_pose,
    num_joints,
    policy,
    profile,
    profile_names,
    robot_base_in_world,
    robot_base_in_world_verified,
    sim_ee_convention,
    sim_world_alignment,
    teleop,
    worktable_height_m,
)
from .transforms import (
    Pose,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)

__all__ = [
    "ArmSpec", "CameraCalibration", "CameraSpec", "EESphere", "GripperSpec", "Pose",
    "ProfileSpec", "all_sections", "arm", "arm_names", "arv_defaults",
    "calib_origin_in_world", "calibration", "camera", "camera_ips", "camera_keys",
    "camera_stream_fps", "config_dir", "control", "control_fps", "default_ee_sphere",
    "default_home_pose_name", "ee_sphere", "framos_defaults", "get", "home_fps",
    "home_poses_dir", "home_q", "load_home_pose", "matrix_to_quat_wxyz", "num_joints",
    "policy", "profile", "profile_names", "quat_wxyz_to_matrix", "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz", "reload", "repo_root", "robot_base_in_world",
    "robot_base_in_world_verified", "section", "sim_ee_convention",
    "sim_world_alignment", "teleop", "worktable_height_m",
]
