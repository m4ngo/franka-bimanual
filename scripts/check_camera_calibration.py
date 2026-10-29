#!/usr/bin/env python3
"""Live point clouds from the FRAMOS cameras in world frame, as a Plotly HTML.

A calibration check: each camera's cloud is drawn in its own colour over the
geometry the config says should be there -- the extrinsics board outline at
`world.calib_origin_in_world`, the robot bases, and each camera's pose. A good
calibration puts the board's points flat inside the outline and makes clouds
from different cameras overlap. The board-height residual printed per camera
is the same check as a number.

Usage:
    python scripts/check_camera_calibration.py                    # depth cameras of the default rig
    python scripts/check_camera_calibration.py --cameras cam_7 cam_2
    python scripts/check_camera_calibration.py --arm              # also draw each arm from its joint state

`--arm` connects to the NUC servers, which starts torque control holding the
arm where it is. Leave it off to check the cameras alone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import franka_config as fc
import numpy as np
import plotly.graph_objects as go

from lerobot_camera_framos import FramosCamera, FramosCameraConfig

_CAMERA_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")
_AXIS_COLOURS = ("red", "green", "blue")


def _axes_traces(pose: fc.Pose, name: str, length: float) -> list[go.Scatter3d]:
    traces = []
    for i, colour in enumerate(_AXIS_COLOURS):
        end = pose.apply(np.eye(3)[i] * length)
        traces.append(go.Scatter3d(
            x=[pose.translation[0], end[0]], y=[pose.translation[1], end[1]],
            z=[pose.translation[2], end[2]],
            mode="lines", line=dict(color=colour, width=6),
            name=name, legendgroup=name, showlegend=i == 0,
        ))
    return traces


def _board_outline(calib_in_world: fc.Pose) -> tuple[go.Scatter3d, float]:
    board = fc.calibration("extrinsics_board")
    hx = 0.5 * board["squares_x"] * board["square_length_m"]
    hy = 0.5 * board["squares_y"] * board["square_length_m"]
    corners = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0], [-hx, -hy, 0]])
    pts = calib_in_world.apply(corners)
    trace = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        line=dict(color="black", width=8), name="extrinsics board (config)",
    )
    return trace, max(hx, hy)


def _board_residual(cloud: np.ndarray, calib_in_world: fc.Pose, half: float) -> str:
    """Height of the cloud above the board plane, over the board's footprint."""
    local = calib_in_world.inverse().apply(cloud.astype(np.float64))
    on_board = np.all(np.abs(local[:, :2]) <= 0.8 * half, axis=1)
    if on_board.sum() < 100:
        return "board not in view (fewer than 100 points over its footprint)"
    h = local[on_board, 2]
    return (f"height above board plane over its footprint: median {1000 * np.median(h):+.1f} mm, "
            f"IQR {1000 * (np.percentile(h, 75) - np.percentile(h, 25)):.1f} mm "
            f"({on_board.sum()} points)")


def _arm_traces(profile: fc.ProfileSpec) -> list[go.Scatter3d]:
    from lerobot_robot_bimanual_franka.franka_process import RobotDriver
    from lerobot_teleoperator_gello.franka_fk import franka_fk_chain

    traces = []
    for key in profile.keys:
        spec = profile.arm_spec(key)
        driver = RobotDriver(spec.server_ip, spec.robot_ip, spec.rpyc_port)
        try:
            q = driver.get_kinematic_state()[0]
        finally:
            driver.shutdown()
        chain = franka_fk_chain(q)
        pts = fc.robot_base_in_world(profile.arms[key]).apply(
            np.vstack([np.zeros((1, 3)), chain[:, :3, 3]]))
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines+markers",
            line=dict(color="dimgray", width=10), marker=dict(size=5, color="dimgray"),
            name=f"{profile.arms[key]} arm (FK)",
        ))
    return traces


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rig", default=fc.default_single_arm_profile(),
                   help="rig profile whose depth cameras and arms are drawn (default: %(default)s)")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="camera ids to capture (default: the rig's depth_cameras)")
    p.add_argument("--arm", action="store_true",
                   help="read each arm's joints over RPyC and draw it (starts torque control)")
    p.add_argument("--radius", type=float, default=1.2,
                   help="keep points within this world-axis box half-extent of the board centre (m)")
    p.add_argument("--max-points", type=int, default=60000,
                   help="per-camera random subsample, to keep the HTML responsive")
    p.add_argument("--warmup-frames", type=int, default=15,
                   help="frames to discard so auto-exposure and depth settle")
    p.add_argument("--out", type=Path, default=None,
                   help="output HTML (default: ~/franka_data/calib_check/<timestamp>.html)")
    args = p.parse_args()

    profile = fc.profile(args.rig)
    keys = args.cameras or list(profile.depth_cameras)
    if not keys:
        p.error(f"rig {args.rig!r} has no depth_cameras; pass --cameras")
    out = args.out or Path.home() / "franka_data" / "calib_check" / (
        time.strftime("%Y%m%d_%H%M%S") + "_" + "_".join(keys) + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)

    calib_in_world = fc.calib_origin_in_world()
    board_trace, board_half = _board_outline(calib_in_world)
    traces: list[go.Scatter3d] = [board_trace]
    traces += _axes_traces(fc.Pose.identity(), "world origin", 0.2)
    traces += _axes_traces(calib_in_world, "calibration origin", 0.1)
    for key in profile.keys:
        traces += _axes_traces(fc.robot_base_in_world(profile.arms[key]),
                               f"{profile.arms[key]} base", 0.15)

    rng = np.random.default_rng(0)
    cams: list[tuple[str, FramosCamera]] = []
    try:
        for key in keys:
            if fc.camera(key).calibration is None:
                print(f"warning: {key} has no calibration; it will sit at the world origin")
            cam = FramosCamera(FramosCameraConfig.for_camera(key))
            cam.connect()
            cams.append((key, cam))
        for _, cam in cams:
            for _ in range(args.warmup_frames):
                cam.read()

        for i, (key, cam) in enumerate(cams):
            cam.read()
            cloud = cam.get_full_point_cloud()
            cloud = cloud[np.max(np.abs(cloud - calib_in_world.translation), axis=1) <= args.radius]
            print(f"{key}: {len(cloud)} points in the box; {_board_residual(cloud, calib_in_world, board_half)}")
            if len(cloud) > args.max_points:
                cloud = cloud[rng.choice(len(cloud), args.max_points, replace=False)]
            colour = _CAMERA_COLOURS[i % len(_CAMERA_COLOURS)]
            traces.append(go.Scatter3d(
                x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], mode="markers",
                marker=dict(size=1.5, color=colour, opacity=0.6), name=f"{key} cloud",
            ))
            pose = fc.camera(key).calibration.cam_in_world if fc.camera(key).calibration else fc.Pose.identity()
            traces += _axes_traces(pose, f"{key} pose (z = optical axis)", 0.1)
    finally:
        for _, cam in cams:
            cam.disconnect()

    if args.arm:
        traces += _arm_traces(profile)

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Calibration check: {', '.join(keys)} ({args.rig})",
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
