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

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_bimanual_franka import BimanualFrankaConfig
from lerobot_teleoperator_gello.franka_fk import franka_fk_chain


def _camera_names() -> dict[str, str]:
    """cam_N → human name from BimanualFrankaConfig's default factory."""
    field = next(f for f in dataclasses.fields(BimanualFrankaConfig) if f.name == "cameras")
    return {k: v.name for k, v in field.default_factory().items()}


def _build_blueprint(cam_keys: list[str], cam_names: dict[str, str]) -> rrb.Blueprint:
    cam_views = [
        rrb.Spatial2DView(origin=k, name=f"{k}: {cam_names.get(k, k)}")
        for k in cam_keys
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*cam_views),
            rrb.Horizontal(
                rrb.Spatial3DView(origin="l_arm", name="Left arm"),
                rrb.Spatial3DView(origin="r_arm", name="Right arm"),
            ),
            rrb.TimeSeriesView(origin="gripper", name="Gripper cmd vs obs"),
            row_shares=[2, 3, 1],
        ),
        collapse_panels=True,
    )


def _log_arm(side: str, q: np.ndarray, ee_pos: np.ndarray, ee_quat: np.ndarray) -> None:
    chain = franka_fk_chain(q)
    points = np.vstack([np.zeros((1, 3)), chain[:, :3, 3]])  # base + 7 joints + EE
    rr.log(f"{side}_arm/skeleton",
           rr.LineStrips3D([points], colors=[(180, 180, 220)], radii=0.005))
    rr.log(f"{side}_arm/joints",
           rr.Points3D(points, colors=(200, 80, 80), radii=0.015))
    rr.log(f"{side}_arm/ee_target",
           rr.Transform3D(translation=ee_pos, rotation=rr.Quaternion(xyzw=ee_quat)))
    rr.log(f"{side}_arm/ee_target/origin",
           rr.Points3D([[0.0, 0.0, 0.0]], colors=(80, 200, 80), radii=0.02))


def visualize(repo_id: str, episode_index: int, compress: bool, spawn: bool) -> None:
    dataset = LeRobotDataset(repo_id, episodes=[episode_index])
    cam_keys = list(dataset.meta.camera_keys)
    cam_names = _camera_names()

    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn)
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

        for side, off in (("l", 0), ("r", 8)):
            ee_pos = action[off:off + 3]
            ee_quat = action[off + 3:off + 7]
            g_cmd = float(action[off + 7])

            q = state[off:off + 7]
            g_obs = float(state[off + 7])

            _log_arm(side, q, ee_pos, ee_quat)
            rr.log(f"gripper/{side}_cmd", rr.Scalars(g_cmd))
            rr.log(f"gripper/{side}_obs", rr.Scalars(g_obs))
            for j in range(7):
                rr.log(f"joints/{side}_{j + 1}", rr.Scalars(float(q[j])))


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
