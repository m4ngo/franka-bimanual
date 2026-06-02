"""Visualize a recorded LeRobot episode in Rerun for obs/action sync debugging.

Lays out, side-by-side:
- All six camera streams, captioned with their BimanualFrankaConfig names so
  the user can identify third-person vs left/right wrist by eye.
- Two 3D views (one per arm) with a stick-figure FR3 from franka_fk_chain
  plus the commanded EE target as a transform marker.
- A time-series panel comparing commanded vs observed gripper for each arm.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_bimanual_franka import BimanualFrankaConfig
from lerobot_teleoperator_gello.franka_fk import franka_fk_chain

_WORLD_IN_ROBOT_TRANSLATION_M = np.array((0.669, 0.003, 0.120), dtype=np.float64)
_WORLD_IN_ROBOT_QUAT_WXYZ = np.array((-0.376557, 0.0, 0.0, 0.926393), dtype=np.float64)
_WORLD_FROM_ROBOT_ROT = Rotation.from_quat(
    (_WORLD_IN_ROBOT_QUAT_WXYZ[1], _WORLD_IN_ROBOT_QUAT_WXYZ[2], _WORLD_IN_ROBOT_QUAT_WXYZ[3], _WORLD_IN_ROBOT_QUAT_WXYZ[0])
).inv()


def _camera_names() -> dict[str, str]:
    """cam_N → human name from BimanualFrankaConfig's default factory."""
    field = next(f for f in dataclasses.fields(BimanualFrankaConfig) if f.name == "cameras")
    default_factory = field.default_factory
    if default_factory is dataclasses.MISSING:
        return {}
    return {k: v.name for k, v in default_factory().items()}


def _is_depth_name(name: str) -> bool:
    return name.startswith("depth_")


def _arm_prefixes(names: Sequence[str]) -> list[str]:
    prefixes: list[str] = []
    for name in names:
        if _is_depth_name(name):
            continue
        prefix, _, suffix = name.partition("_")
        if prefix in ("l", "r") and suffix:
            if prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes


def _action_mode(action_names: Sequence[str]) -> str:
    for name in action_names:
        if name.endswith("_x"):
            return "ee"
    return "joint"


def _robot_to_world_points(points: np.ndarray) -> np.ndarray:
    return _WORLD_FROM_ROBOT_ROT.apply(np.asarray(points, dtype=np.float64) - _WORLD_IN_ROBOT_TRANSLATION_M)


def _robot_to_world_pose(translation: np.ndarray, rotation_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_translation = _robot_to_world_points(np.asarray(translation, dtype=np.float64))
    world_rotation = _WORLD_FROM_ROBOT_ROT * Rotation.from_quat(np.asarray(rotation_xyzw, dtype=np.float64))
    return world_translation, world_rotation.as_quat()


def _build_blueprint(cam_keys: list[str], cam_names: dict[str, str]) -> rrb.Blueprint:
    cam_views = [
        rrb.Spatial2DView(origin=k, name=f"{k}: {cam_names.get(k, k)}")
        for k in cam_keys
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*cam_views),
            rrb.Spatial3DView(
                origin="/",
                name="3D scene",
                line_grid=rrb.LineGrid3D(visible=True),
                eye_controls=rrb.EyeControls3D(),
            ),
            rrb.TimeSeriesView(origin="gripper", name="Gripper cmd vs obs"),
            row_shares=[2, 4, 1],
        ),
        collapse_panels=True,
    )


def _log_arm(side: str, q: np.ndarray, ee_pos: np.ndarray, ee_quat: np.ndarray) -> None:
    chain = franka_fk_chain(q)
    points = np.vstack([np.zeros((1, 3)), chain[:, :3, 3]])  # base + 7 joints + EE
    points = _robot_to_world_points(points)
    ee_pos_world, ee_quat_world = _robot_to_world_pose(ee_pos, ee_quat)
    rr.log(f"{side}_arm/skeleton",
           rr.LineStrips3D([points], colors=[(180, 180, 220)], radii=0.005))
    rr.log(f"{side}_arm/joints",
           rr.Points3D(points, colors=(200, 80, 80), radii=0.015))
    rr.log(f"{side}_arm/ee_target",
        rr.Transform3D(translation=ee_pos_world, rotation=rr.Quaternion(xyzw=ee_quat_world)))
    rr.log(f"{side}_arm/ee_target/origin",
           rr.Points3D([[0.0, 0.0, 0.0]], colors=(80, 200, 80), radii=0.02))


def _log_depth(depth: np.ndarray | torch.Tensor) -> None:
    points = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return
    rr.log("depth", rr.Points3D(points, colors=(120, 180, 255), radii=0.004))


def _extract_depth_points(state: np.ndarray | torch.Tensor, state_names: list[str]) -> np.ndarray | None:
    values = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
    depth_indices = [index for index, name in enumerate(state_names) if _is_depth_name(name)]
    if len(depth_indices) < 3:
        return None
    depth_values = values[depth_indices]
    if depth_values.size % 3 != 0:
        return None
    return depth_values.reshape(-1, 3)


def _arm_block(names: Sequence[str], arm: str) -> tuple[int, int] | None:
    start = next((i for i, name in enumerate(names) if name.startswith(f"{arm}_")), None)
    if start is None:
        return None
    end = start
    while end < len(names) and names[end].startswith(f"{arm}_"):
        end += 1
    return start, end


def visualize(repo_id: str, episode_index: int, compress: bool, spawn: bool) -> None:
    dataset = LeRobotDataset(repo_id, episodes=[episode_index])
    cam_keys = list(dataset.meta.camera_keys)
    cam_names = _camera_names()
    action_names = list(dataset.meta.names["action"])
    state_names = list(dataset.meta.names["observation.state"])
    arm_prefixes = _arm_prefixes(action_names)
    mode = _action_mode(action_names)
    has_depth = any(_is_depth_name(name) for name in state_names)

    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.send_blueprint(_build_blueprint(cam_keys, cam_names))

    first_index: int | None = None
    for i in range(len(dataset)):
        frame = dataset[i]
        idx = int(frame["index"])
        if first_index is None:
            first_index = idx
        rr.set_time("frame_index", sequence=idx - first_index)
        rr.set_time("timestamp", timestamp=float(frame["timestamp"]))

        for key in cam_keys:
            img = (frame[key] * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            rr.log(key, rr.Image(img).compress() if compress else rr.Image(img))

        action = frame["action"].numpy()
        state = frame["observation.state"].numpy()

        if has_depth:
            depth_points = _extract_depth_points(state, state_names)
            if depth_points is not None:
                _log_depth(depth_points)

        for arm in arm_prefixes:
            action_block = _arm_block(action_names, arm)
            state_block = _arm_block(state_names, arm)
            if action_block is None or state_block is None:
                continue

            action_start, action_end = action_block
            state_start, state_end = state_block
            action_slice = action[action_start:action_end]
            state_slice = state[state_start:state_end]

            q = state_slice[:7]
            g_obs = float(state_slice[-1])
            g_cmd = float(action_slice[-1])

            if mode == "ee" and action_slice.size >= 8:
                ee_pos = action_slice[:3]
                ee_quat = action_slice[3:7]
                _log_arm(arm, q, ee_pos, ee_quat)
            else:
                _log_arm(arm, q, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

            rr.log(f"gripper/{arm}_cmd", rr.Scalars(g_cmd))
            rr.log(f"gripper/{arm}_obs", rr.Scalars(g_obs))
            for j in range(7):
                rr.log(f"joints/{arm}_{j + 1}", rr.Scalars(float(q[j])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="HuggingFace dataset repo id")
    parser.add_argument("episode_index", type=int, help="Episode index to visualize")
    parser.add_argument("--no-compress", dest="compress", action="store_false",
                        help="Log raw images instead of compressed (uses more memory)")
    parser.add_argument("--no-spawn", dest="spawn", action="store_false",
                        help="Don't spawn a local Rerun viewer (e.g. when running headless)")
    args = parser.parse_args()
    visualize(args.repo_id, args.episode_index, args.compress, args.spawn)


if __name__ == "__main__":
    main()
