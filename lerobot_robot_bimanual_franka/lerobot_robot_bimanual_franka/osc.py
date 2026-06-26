"""Operational-space control matching multi-fast robosuite OSC (osc.py).

Robosuite outputs joint torques:
  F = kp[:3] * pos_err - kd[:3] * ee_vel_lin
  T = kp[3:] * ori_err - kd[3:] * ee_vel_ang
  tau = J^T [Lambda_pos F; Lambda_ori T]   (uncouple_pos_ori=True)

With a Cartesian-velocity interface the same task-space PD and goals produce the
same end-effector motion when we command the operational-space twist directly:
  v_ee = [F; T]   (base frame, m/s and rad/s)

This holds because J_pos @ J_bar_pos = I and J_ori @ J_bar_ori = I for the
dynamically-consistent pseudoinverse, so J @ (J_bar_pos F + J_bar_ori T) = [F; T].

Gains are in velocity units (1/s): osc_kp_base scales like a task-space velocity
gain, not robosuite's default kp=150 N/m.  Goal updates and error terms match
osc.py exactly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.types import RobotAction

from .franka_process import KinematicSnapshot

EE_DELTA_ROT_KEYS: tuple[str, ...] = ("rx", "ry", "rz")
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")

_OSC_STIFFNESS_EXP_SCALE = 10.0
_OSC_DAMPING_EXP_SCALE = 10.0


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 1.0 / n
    return np.array([
        [1.0 - 2 * s * (y * y + z * z), 2 * s * (x * y - z * w), 2 * s * (x * z + y * w)],
        [2 * s * (x * y + z * w), 1.0 - 2 * s * (x * x + z * z), 2 * s * (y * z - x * w)],
        [2 * s * (x * z - y * w), 2 * s * (y * z + x * w), 1.0 - 2 * s * (x * x + y * y)],
    ], dtype=np.float64)


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Axis-angle error; matches robosuite control_utils.orientation_error."""
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(aa))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    s, c = np.sin(angle), np.cos(angle)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return c * np.eye(3) + s * k + (1 - c) * np.outer(axis, axis)


def scale_delta_action(action: np.ndarray, output_max: float, input_max: float = 1.0) -> np.ndarray:
    """robosuite base_controller.scale_action (symmetric input/output)."""
    scale = abs(2 * output_max) / abs(2 * input_max)
    return np.clip(action, -input_max, input_max) * scale


def parse_variable_impedance(
    action: RobotAction,
    kp_base: float,
    damping_ratio: float,
    kp_ori_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Variable impedance: exp-scale scalar kp/kd → 6-DOF (robosuite variable mode)."""
    kp_norm = float(np.clip(action.get("kp", 0.0), -1.0, 1.0))
    damp_norm = float(np.clip(action.get("kd", 0.0), -1.0, 1.0))
    kp = np.full(6, float(np.power(_OSC_STIFFNESS_EXP_SCALE, kp_norm) * kp_base))
    kp[3:] *= kp_ori_ratio
    zeta = np.full(6, float(np.power(_OSC_DAMPING_EXP_SCALE, damp_norm) * damping_ratio))
    kd = 2.0 * np.sqrt(kp) * zeta
    return kp, kd


def task_space_pd(
    goal_pos: np.ndarray,
    goal_ori_mat: np.ndarray,
    ee_pos: np.ndarray,
    ee_rot_xyzw: np.ndarray,
    ee_twist: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Task-space PD → 6D EE twist.  Matches osc.py run_controller force/torque."""
    ee_ori = quat_xyzw_to_mat(np.asarray(ee_rot_xyzw, dtype=np.float64))
    pos_err = np.asarray(goal_pos, dtype=np.float64) - np.asarray(ee_pos, dtype=np.float64)
    ori_err = orientation_error(goal_ori_mat, ee_ori)
    twist = np.asarray(ee_twist, dtype=np.float64)

    lin = kp[:3] * pos_err - kd[:3] * twist[:3]
    ang = kp[3:] * ori_err - kd[3:] * twist[3:]
    return np.concatenate([lin, ang])


def update_goal_delta(
    action: RobotAction,
    arm: str,
    snap: KinematicSnapshot,
    goal_ori_mat: dict[str, np.ndarray],
    out_max_pos: float,
    out_max_rot: float,
    dpos: np.ndarray,
    drot: np.ndarray,
) -> np.ndarray:
    """robosuite set_goal with control_delta=True."""
    _, _, _, _, ee_pos, ee_rot_xyzw, _ = snap
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    ee_ori = quat_xyzw_to_mat(np.asarray(ee_rot_xyzw, dtype=np.float64))

    raw_dpos = np.array([action[f"{arm}_x"], action[f"{arm}_y"], action[f"{arm}_z"]], dtype=np.float64)
    raw_drot = np.array([action[f"{arm}_{k}"] for k in EE_DELTA_ROT_KEYS], dtype=np.float64)
    scaled_dpos = scale_delta_action(raw_dpos, out_max_pos) + dpos
    scaled_drot = scale_delta_action(raw_drot, out_max_rot) + drot

    goal_pos = ee_pos + scaled_dpos
    if sum(0.0 if math.isclose(v, 0.0) else 1.0 for v in scaled_drot) > 0.0:
        goal_ori_mat[arm] = axis_angle_to_mat(scaled_drot) @ ee_ori
    return goal_pos


def update_goal_absolute(
    action: RobotAction,
    arm: str,
    snap: KinematicSnapshot,
    goal_ori_mat: dict[str, np.ndarray],
    dpos: np.ndarray,
    drot: np.ndarray,
    ignore_action: bool,
) -> np.ndarray:
    _, _, _, _, ee_pos, ee_rot_xyzw, _ = snap
    if ignore_action:
        goal_ori_mat[arm] = quat_xyzw_to_mat(np.asarray(ee_rot_xyzw, dtype=np.float64))
        return np.asarray(ee_pos, dtype=np.float64).copy()

    target = np.fromiter(
        (action[f"{arm}_{k}"] for k in EE_AXIS_KEYS), dtype=np.float64, count=len(EE_AXIS_KEYS),
    )
    target[3:] /= max(float(np.linalg.norm(target[3:])), 1e-12)
    goal_ori_mat[arm] = axis_angle_to_mat(drot) @ quat_xyzw_to_mat(target[3:])
    return target[:3] + dpos


def ee_velocity_delta(
    action: RobotAction,
    arm: str,
    snap: KinematicSnapshot,
    goal_ori_mat: dict[str, np.ndarray],
    kp: np.ndarray,
    kd: np.ndarray,
    out_max_pos: float,
    out_max_rot: float,
    dpos: np.ndarray,
    drot: np.ndarray,
) -> np.ndarray:
    goal_pos = update_goal_delta(
        action, arm, snap, goal_ori_mat, out_max_pos, out_max_rot, dpos, drot,
    )
    _, _, _, _, ee_pos, ee_rot_xyzw, ee_twist = snap
    return task_space_pd(goal_pos, goal_ori_mat[arm], ee_pos, ee_rot_xyzw, ee_twist, kp, kd)


def ee_velocity_absolute(
    action: RobotAction,
    arm: str,
    snap: KinematicSnapshot,
    goal_ori_mat: dict[str, np.ndarray],
    kp: np.ndarray,
    kd: np.ndarray,
    dpos: np.ndarray,
    drot: np.ndarray,
    ignore_action: bool,
) -> np.ndarray:
    goal_pos = update_goal_absolute(action, arm, snap, goal_ori_mat, dpos, drot, ignore_action)
    _, _, _, _, ee_pos, ee_rot_xyzw, ee_twist = snap
    return task_space_pd(goal_pos, goal_ori_mat[arm], ee_pos, ee_rot_xyzw, ee_twist, kp, kd)
