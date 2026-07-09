"""Residual-policy episode visualizer — interactive Plotly HTML.

Call ``EpisodeRecorder.record()`` at each control step and
``EpisodeRecorder.record_chunk()`` each time inference runs a new action chunk,
then ``save_episode_html(recorder, path)`` after the loop to write a
self-contained animated HTML containing:

    - 3D arm skeleton animated over the episode (FK chain)
    - Actual EE trail — where the arm physically was (thin gray, cumulative)
    - Base chunk forecast — full projected trajectory from the base policy at
      the most recent inference (dashed orange); updated each time a new chunk
      is inferred, stays fixed within a chunk execution window.
    - Total chunk forecast — base + residual projected trajectory (solid blue);
      shows what the residual-modified plan looks like over the whole chunk.
    - Time-series panels (right column): kp, kd, commanded gripper

Terminology
-----------
actual EE position : FK(joint_angles from obs) — where the arm physically is.
base forecast      : ee_pos_at_inference + cumsum(commanded_delta × EE_PD_KP × kp_gain × dt)
                     over the first _RESIDUAL_HORIZON steps — the expected actual EE
                     trajectory scaled to match arm movement (not raw commanded deltas,
                     which are ~10× larger due to the PD velocity scaling).
total forecast     : same, but residual position corrections are included for the
                     first _CHUNK_EXEC steps, using the residual policy's kp_gain.

The HTML is self-contained via ``include_plotlyjs="cdn"`` and mirrors the
animated-slider pattern from multi-fast/utils/distill/pcd_viz.py.
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_gello.franka_fk import franka_fk_chain

# Default world→robot transform (matches BimanualFrankaConfig defaults).
_DEFAULT_WORLD_IN_ROBOT_T = (0.669, 0.003, 0.120)
_DEFAULT_WORLD_IN_ROBOT_Q_WXYZ = (-0.376557, 0.0, 0.0, 0.926393)
_AXIS_COLORS = (
    "rgba(220, 40, 40, 0.92)",
    "rgba(40, 170, 80, 0.92)",
    "rgba(50, 90, 235, 0.92)",
)
_FORECAST_AXIS_LENGTH = 0.038
_CURRENT_AXIS_COLORS = _AXIS_COLORS
_TOTAL_AXIS_COLORS = (
    "rgba(190, 55, 210, 0.82)",
    "rgba(45, 155, 180, 0.82)",
    "rgba(70, 115, 255, 0.82)",
)
_BASE_AXIS_COLORS = (
    "rgba(235, 95, 35, 0.82)",
    "rgba(95, 180, 70, 0.82)",
    "rgba(55, 120, 220, 0.82)",
)


# ---------------------------------------------------------------------------
# Data recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    """Accumulates per-step state and per-inference chunk forecasts for visualization.

    Designed to be created by the caller before entering _run_episode and
    accessed afterwards (even if the episode was interrupted early).

    Two recording methods:
        record()       — called every control step (joint angles, gains, gripper …).
        record_chunk() — called once per inference event with the full projected
                         chunk trajectory (base forecast and base+residual forecast).
    """

    def __init__(self) -> None:
        self.joint_angles: list[np.ndarray] = []       # (7,) per step
        self.actual_ee_pos: list[np.ndarray] = []      # (3,) actual EE via FK
        self.base_desired_pos: list[np.ndarray] = []   # (3,) kept for compat; not used by viz
        self.total_desired_pos: list[np.ndarray] = []  # (3,) kept for compat; not used by viz
        self.kp: list[float] = []
        self.kd: list[float] = []
        self.gripper: list[float] = []
        self.point_clouds: list[np.ndarray | None] = []  # (N, 3) world-space, or None
        # Per-inference chunk forecast events, sorted by ascending step index.
        self.chunk_events: list[dict] = []

    def record(
        self,
        q: np.ndarray,
        actual_ee_pos: np.ndarray,
        base_desired_pos: np.ndarray,
        total_desired_pos: np.ndarray,
        kp: float,
        kd: float,
        gripper: float,
        point_cloud: np.ndarray | None = None,
    ) -> None:
        """Record one control step.

        Args:
            q:                 (7,) joint angles in rad (from obs).
            actual_ee_pos:     Actual EE position [x, y, z] (m) — FK(q).
            base_desired_pos:  Base-policy absolute EE target [x, y, z] (m).
            total_desired_pos: Effective PD target [x, y, z] (m) = base + residual dpos.
            kp:                Proportional gain (normalised, range [-1, 1]).
            kd:                Derivative gain   (normalised, range [-1, 1]).
            gripper:           Commanded gripper normalised to [0, 1].
            point_cloud:       (N, 3) point cloud in world space, or None.
        """
        self.joint_angles.append(np.asarray(q, dtype=np.float64).copy())
        self.actual_ee_pos.append(np.asarray(actual_ee_pos[:3], dtype=np.float32).copy())
        self.base_desired_pos.append(np.asarray(base_desired_pos[:3], dtype=np.float32).copy())
        self.total_desired_pos.append(np.asarray(total_desired_pos[:3], dtype=np.float32).copy())
        self.kp.append(float(kp))
        self.kd.append(float(kd))
        self.gripper.append(float(gripper))
        self.point_clouds.append(
            np.asarray(point_cloud, dtype=np.float32).copy() if point_cloud is not None else None
        )

    def record_chunk(
        self,
        step: int,
        ee_pos: np.ndarray,
        base_traj: np.ndarray,
        total_traj: np.ndarray,
        base_traj_pose: np.ndarray | None = None,
        total_traj_pose: np.ndarray | None = None,
    ) -> None:
        """Record one inference event with the full projected chunk trajectory.

        Should be called immediately after inference runs (chunk_used == 0) and
        before the corresponding record() call for that step.

        Args:
            step:       Episode step index (== len(recorder) at call time).
            ee_pos:     (3,) EE position [x, y, z] (m) at inference time.
            base_traj:  (_RESIDUAL_HORIZON+1, 3) expected actual EE positions in
                        metres — ee_pos prepended, then cumulative sum of
                        (commanded_delta × EE_PD_KP × kp_gain × dt) per step.
                        Scaled to match the actual EE trail, not raw deltas.
            total_traj: (_RESIDUAL_HORIZON+1, 3) same as base_traj but with the
                        residual position correction added (using residual kp_gain)
                        for the first _CHUNK_EXEC steps.
            base_traj_pose:
                        Optional (_RESIDUAL_HORIZON+1, 7) [xyz + xyzw] pose
                        trajectory for the base forecast.
            total_traj_pose:
                        Optional (_RESIDUAL_HORIZON+1, 7) [xyz + xyzw] pose
                        trajectory for the total forecast.
        """
        base_traj_pose_arr = None if base_traj_pose is None else np.asarray(base_traj_pose, dtype=np.float32).copy()
        total_traj_pose_arr = None if total_traj_pose is None else np.asarray(total_traj_pose, dtype=np.float32).copy()
        self.chunk_events.append({
            "step": int(step),
            "ee_pos": np.asarray(ee_pos[:3], dtype=np.float32).copy(),
            "base_traj": np.asarray(base_traj, dtype=np.float32).copy(),
            "total_traj": np.asarray(total_traj, dtype=np.float32).copy(),
            "base_traj_pose": base_traj_pose_arr,
            "total_traj_pose": total_traj_pose_arr,
        })

    def __len__(self) -> int:
        return len(self.joint_angles)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skeleton_pts(q: np.ndarray) -> np.ndarray:
    """Return (9, 3) skeleton positions: robot base origin + 7 joint frames + EE."""
    chain = franka_fk_chain(q)  # (8, 4, 4)
    return np.vstack([np.zeros((1, 3)), chain[:, :3, 3]])  # (9, 3)


def _fk_pose(q: np.ndarray) -> np.ndarray:
    """Return [x, y, z, qx, qy, qz, qw] for the EE pose implied by q."""
    chain = franka_fk_chain(q)
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = chain[7, :3, 3]
    pose[3:] = Rotation.from_matrix(chain[7, :3, :3]).as_quat().astype(np.float32)
    return pose


def _poses_from_qs(qs: list[np.ndarray]) -> np.ndarray:
    """Return (T, 7) EE poses for a list of joint-angle vectors."""
    return np.array([_fk_pose(q) for q in qs], dtype=np.float32)


def _propagate_pose_traj(
    start_pose: np.ndarray,
    pos_deltas: np.ndarray,
    rot_deltas: np.ndarray,
) -> np.ndarray:
    """Integrate a local-frame pose trajectory from position and rotvec deltas.

    The rotational delta is composed on the right: R_next = R_current * R_delta.
    That matches the EE-local increment convention used by the controller.
    """
    start_pose = np.asarray(start_pose, dtype=np.float64)
    pos_deltas = np.asarray(pos_deltas, dtype=np.float64)
    rot_deltas = np.asarray(rot_deltas, dtype=np.float64)
    pose_traj = np.empty((len(pos_deltas) + 1, 7), dtype=np.float32)
    pose_traj[0, :3] = start_pose[:3]
    pose_traj[0, 3:] = start_pose[3:7]

    current_pos = start_pose[:3].copy()
    current_rot = Rotation.from_quat(start_pose[3:7])
    for i, (dpos, drot) in enumerate(zip(pos_deltas, rot_deltas), start=1):
        current_pos = current_pos + dpos
        current_rot = current_rot * Rotation.from_rotvec(drot)
        pose_traj[i, :3] = current_pos
        pose_traj[i, 3:] = current_rot.as_quat()
    return pose_traj


def _pose_axis_segments(poses: np.ndarray, axis_idx: int, length: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return line-segment coordinates for a colored local axis over a pose list."""
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) == 0:
        return [], [], []

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for pose in poses:
        origin = pose[:3]
        rot = Rotation.from_quat(pose[3:7]).as_matrix()
        tip = origin + rot[:, axis_idx] * length
        xs.extend([float(origin[0]), float(tip[0]), None])
        ys.extend([float(origin[1]), float(tip[1]), None])
        zs.extend([float(origin[2]), float(tip[2]), None])
    return xs, ys, zs


def _pose_axes_traces(
    poses: np.ndarray,
    name_prefix: str,
    length: float,
    colors: tuple[str, str, str] = _AXIS_COLORS,
    width: int = 4,
    opacity: float = 1.0,
) -> list[go.Scatter3d]:
    """Build three local x/y/z axis traces for a pose trajectory."""
    traces: list[go.Scatter3d] = []
    for axis_idx, color in enumerate(colors):
        xs, ys, zs = _pose_axis_segments(poses, axis_idx, length)
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                name=f"{name_prefix} axis {axis_idx}",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def _skeleton_trace(pts: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines+markers",
        line=dict(color="dimgray", width=6),
        marker=dict(size=5, color="dimgray"),
        name="skeleton",
    )


def _trail_trace(
    pos: np.ndarray,
    color: str,
    name: str,
    dash: str = "solid",
    width: int = 5,
    marker_size: int = 3,
) -> go.Scatter3d:
    """3D polyline for an EE trajectory trail."""
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="lines+markers",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=marker_size, color=color),
        name=name,
    )


def _metric_trace(
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    name: str,
) -> go.Scatter:
    return go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=color, width=2),
        name=name,
    )


def _build_world_to_robot(
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) for p_robot = R @ p_world + t.

    Args:
        translation:  (tx, ty, tz) in metres.
        quat_wxyz:    Unit quaternion (w, x, y, z).
    """
    w, x, y, z = quat_wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy expects xyzw
    t = np.array(translation, dtype=np.float64)
    return R, t


def _apply_world_to_robot(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply rigid transform to the xyz columns; pass any extra columns (e.g. RGB) through.

    pts: (N, 3) or (N, 6) — xyz [+ rgb].  Returns same shape.
    """
    xyz_out = (R @ pts[:, :3].T).T + t
    if pts.shape[1] == 3:
        return xyz_out
    return np.concatenate([xyz_out, pts[:, 3:]], axis=1)


def _pcd_marker(pts: np.ndarray) -> dict:
    """Build a Plotly marker dict for a point cloud array.

    (N, 6) xyzrgb → per-point hex colors at full opacity (colors are vivid enough
    on their own; blending with the background washes them out).
    (N, 3) xyz    → flat dimgray at 0.65 opacity (ghost-style).
    """
    if pts.shape[1] >= 6:
        rgb = (pts[:, 3:6] * 255.0).clip(0, 255).astype(np.uint8)
        packed = (rgb[:, 0].astype(np.int32) << 16) | (rgb[:, 1].astype(np.int32) << 8) | rgb[:, 2].astype(np.int32)
        colors = [f"#{c:06x}" for c in packed]
        return dict(size=6, color=colors, opacity=1.0)
    return dict(size=6, color="dimgray", opacity=0.65)


def _pcd_trace(pts: np.ndarray) -> go.Scatter3d:
    """Scatter3d for a single point cloud frame."""
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=_pcd_marker(pts),
        name="point cloud",
    )


_EMPTY_FORECAST = np.zeros((1, 3), dtype=np.float32)


def _active_chunk_event(chunk_events: list[dict], step_idx: int) -> "dict | None":
    """Return the most recent chunk event whose step <= step_idx, or None."""
    active = None
    for ev in chunk_events:
        if ev["step"] <= step_idx:
            active = ev
        else:
            break
    return active


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_episode_html(
    recorder: EpisodeRecorder,
    path: str,
    title: str = "Residual episode",
    frame_stride: int = 1,
    fps: float = 20.0,
    world_in_robot_translation: tuple[float, float, float] = _DEFAULT_WORLD_IN_ROBOT_T,
    world_in_robot_quat_wxyz: tuple[float, float, float, float] = _DEFAULT_WORLD_IN_ROBOT_Q_WXYZ,
    pcd_max_pts: int = 3000,
) -> None:
    """Write a self-contained animated Plotly HTML from recorded episode data.

    Layout:
        Left 65 %  — animated 3D arm skeleton + actual EE trail (thin gray,
                       cumulative) + chunk forecasts updated at each inference:
                       total chunk forecast (solid blue, base + residual) and
                       base chunk forecast (dashed orange, base only).
                       Optional point cloud when recorder.point_clouds has data.
        Right 35 % — stacked time-series: kp (row 1), kd (row 2), gripper (row 3).

    Trace index map (for go.Frame updates):
        0  skeleton            — animated arm FK chain
        1  actual trail        — cumulative actual EE path (FK from obs)
        2  total chunk forecast — base + residual projected trajectory for active chunk
        3  base chunk forecast  — base-policy-only projected trajectory for active chunk
        4  kp metric
        5  kp_true metric
        6  kd metric
        7  kd_true metric
        8  gripper metric
        9  point cloud         — only present when recorder.point_clouds contains data
        (static zero-lines appended last, not in any frame update)

    Args:
        recorder:                  Populated EpisodeRecorder.
        path:                      Output .html path; parent directories are created.
        title:                     Figure title.
        frame_stride:              Emit every Nth step (default 1 = all).
                                   Use 2–4 to reduce file size for long episodes.
        fps:                       Playback speed in frames per second (default 20).
        world_in_robot_translation: (tx, ty, tz) metres — translation part of the
                                   world→robot rigid transform.  Defaults to
                                   BimanualFrankaConfig values.
        world_in_robot_quat_wxyz:  (w, x, y, z) unit quaternion — rotation part of
                                   the world→robot rigid transform.  Defaults to
                                   BimanualFrankaConfig values.
        pcd_max_pts:               Max points to render per frame (uniformly
                                   subsampled).  Reduces HTML size for dense clouds.
    """
    T_full = len(recorder)
    if T_full == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    frame_duration_ms = int(round(1000.0 / max(fps, 1.0)))

    # --- apply stride --------------------------------------------------------
    indices = list(range(0, T_full, max(1, frame_stride)))
    T = len(indices)

    joint_angles     = [recorder.joint_angles[i]      for i in indices]
    actual_pos       = np.array([recorder.actual_ee_pos[i]     for i in indices])  # (T, 3)
    total_des_pos    = np.array([recorder.total_desired_pos[i] for i in indices])  # (T, 3)
    base_des_pos     = np.array([recorder.base_desired_pos[i]  for i in indices])  # (T, 3)
    kp_arr           = np.array([recorder.kp[i]      for i in indices])
    kd_arr           = np.array([recorder.kd[i]      for i in indices])
    grip_arr         = np.array([recorder.gripper[i] for i in indices])
    ts               = np.array(indices, dtype=np.float32)

    # --- point clouds: transform world→robot and subsample -------------------
    R_w2r, t_w2r = _build_world_to_robot(world_in_robot_translation, world_in_robot_quat_wxyz)
    raw_pcds = [recorder.point_clouds[i] for i in indices]  # (T,) list of (N,3) or None
    has_pcd = any(p is not None and len(p) > 0 for p in raw_pcds)

    pcd_robot: list[np.ndarray] = []
    if has_pcd:
        empty = np.zeros((0, 3), dtype=np.float32)
        for pts in raw_pcds:
            if pts is None or len(pts) == 0:
                pcd_robot.append(empty)
                continue
            transformed = _apply_world_to_robot(pts.astype(np.float64), R_w2r, t_w2r).astype(np.float32)
            if len(transformed) > pcd_max_pts:
                idx = np.random.choice(len(transformed), pcd_max_pts, replace=False)
                transformed = transformed[idx]
            pcd_robot.append(transformed)

    # --- forward kinematics --------------------------------------------------
    fk_frames = [(_skeleton_pts(q), _fk_pose(q)) for q in joint_angles]
    skeletons = [item[0] for item in fk_frames]  # T × (9, 3)
    ee_poses = np.array([item[1] for item in fk_frames], dtype=np.float32)  # T × (7,)

    # --- chunk forecast events -----------------------------------------------
    chunk_events = recorder.chunk_events  # sorted ascending by step

    # --- global 3D bbox (prevents camera rescaling between animation frames) -
    pcd_for_bbox = [p[:, :3] for p in pcd_robot if len(p) > 0] if has_pcd else []
    chunk_pts = []
    for ev in chunk_events:
        chunk_pts.append(ev["base_traj"])
        chunk_pts.append(ev["total_traj"])
    all_xyz = np.concatenate(
        [actual_pos] + skeletons + chunk_pts + pcd_for_bbox if chunk_pts
        else [actual_pos] + skeletons + pcd_for_bbox,
        axis=0,
    )
    mn, mx = all_xyz.min(0), all_xyz.max(0)
    pad = max(float((mx - mn).max()) * 0.05, _FORECAST_AXIS_LENGTH * 1.5)
    x_range = [float(mn[0] - pad), float(mx[0] + pad)]
    y_range = [float(mn[1] - pad), float(mx[1] + pad)]
    z_range = [float(mn[2] - pad), float(mx[2] + pad)]
    extents = np.array(
        [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]],
        dtype=np.float32,
    )
    ext_max = float(extents.max())
    aspectratio = dict(
        x=float(extents[0] / ext_max),
        y=float(extents[1] / ext_max),
        z=float(extents[2] / ext_max),
    )

    # --- subplot layout ------------------------------------------------------
    # Col 1 (65 %): 3D scene spanning all 3 rows.
    # Col 2 (35 %): three stacked 2D panels — kp, kd, gripper.
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "xy"}],
            [None,                             {"type": "xy"}],
            [None,                             {"type": "xy"}],
        ],
        column_widths=[0.65, 0.35],
        row_heights=[0.33, 0.33, 0.34],
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    # Initial chunk forecast: use the first event covering step 0.
    _init_ev = _active_chunk_event(chunk_events, indices[0])
    _init_total_fcast = _init_ev["total_traj"] if _init_ev else _EMPTY_FORECAST
    _init_base_fcast  = _init_ev["base_traj"]  if _init_ev else _EMPTY_FORECAST
    _init_total_pose = _init_ev["total_traj_pose"] if (_init_ev and _init_ev.get("total_traj_pose") is not None) else np.zeros((0, 7), dtype=np.float32)
    _init_base_pose = _init_ev["base_traj_pose"] if (_init_ev and _init_ev.get("base_traj_pose") is not None) else np.zeros((0, 7), dtype=np.float32)

    # --- initial traces (step 0) — order must match go.Frame list below ------
    # Trace 0: skeleton
    fig.add_trace(_skeleton_trace(skeletons[0]), row=1, col=1)
    # Trace 1: actual EE trail
    fig.add_trace(
        _trail_trace(actual_pos[:1], color="slategray", name="actual EE",
                     width=3, marker_size=2),
        row=1, col=1,
    )
    # Trace 2: total chunk forecast (base + residual projected trajectory)
    fig.add_trace(
        _trail_trace(_init_total_fcast, color="royalblue", name="total chunk forecast"),
        row=1, col=1,
    )
    # Trace 3: base chunk forecast (base-policy projected trajectory)
    fig.add_trace(
        _trail_trace(_init_base_fcast, color="darkorange", name="base chunk forecast",
                     dash="dash"),
        row=1, col=1,
    )
    # Trace 4-6: live EE orientation triad.
    for axis_trace in _pose_axes_traces(ee_poses[:1], "current ee", _FORECAST_AXIS_LENGTH, colors=_CURRENT_AXIS_COLORS, width=5, opacity=0.95):
        fig.add_trace(axis_trace, row=1, col=1)
    # Trace 7-9: total forecast orientation triad trail.
    for axis_trace in _pose_axes_traces(_init_total_pose, "total forecast", _FORECAST_AXIS_LENGTH, colors=_TOTAL_AXIS_COLORS, width=4, opacity=0.68):
        fig.add_trace(axis_trace, row=1, col=1)
    # Trace 10-12: base forecast orientation triad trail.
    for axis_trace in _pose_axes_traces(_init_base_pose, "base forecast", _FORECAST_AXIS_LENGTH, colors=_BASE_AXIS_COLORS, width=4, opacity=0.68):
        fig.add_trace(axis_trace, row=1, col=1)
    # Traces 13-14: kp_gain (solid) and kp_true (dotted)
    fig.add_trace(_metric_trace(ts[:1], kp_arr[:1], "crimson", "kp_gain"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**kp_arr[:1], mode="lines",
                             line=dict(color="crimson", width=2, dash="dot"), name="kp_true"),
                  row=1, col=2)
    # Traces 15-16: kd_gain (solid) and kd_true (dotted)
    fig.add_trace(_metric_trace(ts[:1], kd_arr[:1], "seagreen", "kd_gain"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**(kd_arr[:1] * 2 * np.sqrt(kp_arr[:1])), mode="lines",
                             line=dict(color="seagreen", width=2, dash="dot"), name="kd_true"),
                  row=2, col=2)
    # Trace 17: gripper
    fig.add_trace(
        _metric_trace(ts[:1], grip_arr[:1], "darkorchid", "gripper"),
        row=3, col=2,
    )
    # Trace 18 (optional): point cloud at step 0 in robot space
    if has_pcd:
        fig.add_trace(_pcd_trace(pcd_robot[0] if len(pcd_robot[0]) > 0 else np.zeros((1, 3), dtype=np.float32)), row=1, col=1)

    # --- animation frames ----------------------------------------------------
    # Traces 0-17 are always animated; trace 18 (point cloud) is added when present.
    # Index mapping:
    #   0 skeleton  1 actual  2 total  3 base  4-6 current EE axes
    #   7-9 total forecast axes  10-12 base forecast axes
    #   13 kp_gain  14 kp_true  15 kd_gain  16 kd_true  17 gripper  [18 point cloud]
    animated_traces = list(range(18))
    if has_pcd:
        animated_traces.append(18)

    frames = []
    for t in range(T):
        skel = skeletons[t]
        ap   = actual_pos[:t + 1]
        kp_t = kp_arr[:t + 1]
        kd_t = kd_arr[:t + 1]
        current_pose = ee_poses[t:t + 1]

        # Chunk forecast: use the most recent inference event at or before this step.
        ev = _active_chunk_event(chunk_events, indices[t])
        total_fcast = ev["total_traj"] if ev else _EMPTY_FORECAST
        base_fcast  = ev["base_traj"]  if ev else _EMPTY_FORECAST
        total_pose = ev["total_traj_pose"] if (ev and ev.get("total_traj_pose") is not None) else np.zeros((0, 7), dtype=np.float32)
        base_pose = ev["base_traj_pose"] if (ev and ev.get("base_traj_pose") is not None) else np.zeros((0, 7), dtype=np.float32)

        frame_data: list[object] = [
            # 0: skeleton
            go.Scatter3d(
                x=skel[:, 0], y=skel[:, 1], z=skel[:, 2],
                mode="lines+markers",
                line=dict(color="dimgray", width=6),
                marker=dict(size=5, color="dimgray"),
            ),
            # 1: actual EE trail 0..t (cumulative history)
            go.Scatter3d(
                x=ap[:, 0], y=ap[:, 1], z=ap[:, 2],
                mode="lines+markers",
                line=dict(color="slategray", width=3),
                marker=dict(size=2, color="slategray"),
            ),
            # 2: total chunk forecast (base + residual full projected trajectory)
            go.Scatter3d(
                x=total_fcast[:, 0], y=total_fcast[:, 1], z=total_fcast[:, 2],
                mode="lines+markers",
                line=dict(color="royalblue", width=5),
                marker=dict(size=3, color="royalblue"),
            ),
            # 3: base chunk forecast (base-policy full projected trajectory)
            go.Scatter3d(
                x=base_fcast[:, 0], y=base_fcast[:, 1], z=base_fcast[:, 2],
                mode="lines+markers",
                line=dict(color="darkorange", width=4, dash="dash"),
                marker=dict(size=3, color="darkorange"),
            ),
        ]
        frame_data.extend(_pose_axes_traces(current_pose, "current ee", _FORECAST_AXIS_LENGTH, colors=_CURRENT_AXIS_COLORS, width=5, opacity=0.95))
        frame_data.extend(_pose_axes_traces(total_pose, "total forecast", _FORECAST_AXIS_LENGTH, colors=_TOTAL_AXIS_COLORS, width=4, opacity=0.68))
        frame_data.extend(_pose_axes_traces(base_pose, "base forecast", _FORECAST_AXIS_LENGTH, colors=_BASE_AXIS_COLORS, width=4, opacity=0.68))
        frame_data.extend([
            # 13: kp_gain 0..t
            go.Scatter(x=ts[:t + 1], y=kp_t, mode="lines",
                       line=dict(color="crimson", width=2)),
            # 14: kp_true 0..t
            go.Scatter(x=ts[:t + 1], y=10**kp_t, mode="lines",
                       line=dict(color="crimson", width=2, dash="dot")),
            # 15: kd_gain 0..t
            go.Scatter(x=ts[:t + 1], y=kd_t, mode="lines",
                       line=dict(color="seagreen", width=2)),
            # 16: kd_true 0..t
            go.Scatter(x=ts[:t + 1], y=10**(kd_t * 2 * np.sqrt(kp_t)), mode="lines",
                       line=dict(color="seagreen", width=2, dash="dot")),
            # 17: gripper 0..t
            go.Scatter(x=ts[:t + 1], y=grip_arr[:t + 1], mode="lines",
                       line=dict(color="darkorchid", width=2)),
        ])
        if has_pcd:
            pts = pcd_robot[t]
            if len(pts) == 0:
                pts = np.zeros((1, 3), dtype=np.float32)
            # 18: point cloud at step t
            frame_data.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=_pcd_marker(pts),
            ))
        frames.append(
            go.Frame(data=frame_data, traces=animated_traces, name=str(t))
        )
    fig.frames = frames

    # --- layout --------------------------------------------------------------
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=10, t=50, b=60),
        # 3D scene: fixed ranges + manual aspect so the view is stable.
        scene=dict(
            xaxis=dict(range=x_range, autorange=False, title="x (m)"),
            yaxis=dict(range=y_range, autorange=False, title="y (m)"),
            zaxis=dict(range=z_range, autorange=False, title="z (m)"),
            aspectmode="manual",
            aspectratio=aspectratio,
        ),
        # Play / Pause buttons.
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0, y=0.0, xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        # Step slider.
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="step: "),
            pad=dict(t=40),
            steps=[dict(
                method="animate",
                args=[[str(t)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
                label=str(indices[t]),
            ) for t in range(T)],
        )],
    )

    # Fixed axis ranges for metric subplots.
    # kp/kd panels show both the normalised gain [-1,1] and the true value (10^gain),
    # so use autorange for those two; gripper is always [0,1].
    x_end = float(ts[-1]) + 1
    fig.update_yaxes(title_text="kp",      row=1, col=2)
    fig.update_yaxes(title_text="kd",      row=2, col=2)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="gripper", row=3, col=2)
    fig.update_xaxes(range=[float(ts[0]), x_end], title_text="step", row=1, col=2)
    fig.update_xaxes(range=[float(ts[0]), x_end], title_text="step", row=2, col=2)
    fig.update_xaxes(range=[float(ts[0]), x_end], title_text="step", row=3, col=2)

    # Zero-reference lines for kp and kd (static — appended after the 7 animated
    # traces so go.Frame updates don't touch them).
    _zero_line = dict(color="black", width=1, dash="dot")
    fig.add_trace(
        go.Scatter(x=[float(ts[0]), x_end], y=[0.0, 0.0], mode="lines",
                   line=_zero_line, showlegend=False, hoverinfo="skip"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[float(ts[0]), x_end], y=[0.0, 0.0], mode="lines",
                   line=_zero_line, showlegend=False, hoverinfo="skip"),
        row=2, col=2,
    )

    fig.write_html(path, include_plotlyjs="cdn")


def save_rollout_html(
    recorder: EpisodeRecorder,
    path: str,
    title: str = "Base policy rollout",
    frame_stride: int = 1,
    fps: float = 20.0,
    world_in_robot_translation: tuple[float, float, float] = _DEFAULT_WORLD_IN_ROBOT_T,
    world_in_robot_quat_wxyz: tuple[float, float, float, float] = _DEFAULT_WORLD_IN_ROBOT_Q_WXYZ,
    pcd_max_pts: int = 3000,
) -> None:
    """Write a self-contained animated Plotly HTML for a base-policy-only rollout.

    Simpler than save_episode_html: shows the actual EE trail (cumulative, thin
    gray) and the base chunk forecast (royalblue) — the base policy's full projected
    trajectory updated each time a new chunk is inferred.  Use this when there is no
    residual policy.

    Layout:
        Left 65 %  — animated 3D arm skeleton + actual EE trail (thin gray) +
                       base chunk forecast (solid royalblue, updated per inference).
                       Optional point cloud when data is present.
        Right 35 % — stacked time-series: kp (row 1), kd (row 2), gripper (row 3).

    Args:
        recorder:                  Populated EpisodeRecorder.
        path:                      Output .html path; parent directories are created.
        title:                     Figure title.
        frame_stride:              Emit every Nth step (default 1 = all).
        fps:                       Playback speed in frames per second.
        world_in_robot_translation: (tx, ty, tz) metres — world→robot transform.
        world_in_robot_quat_wxyz:  (w, x, y, z) — rotation part of world→robot.
        pcd_max_pts:               Max points to render per frame.
    """
    T_full = len(recorder)
    if T_full == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    frame_duration_ms = int(round(1000.0 / max(fps, 1.0)))

    indices = list(range(0, T_full, max(1, frame_stride)))
    T = len(indices)

    joint_angles   = [recorder.joint_angles[i]     for i in indices]
    actual_pos     = np.array([recorder.actual_ee_pos[i]    for i in indices])  # (T, 3)
    commanded_pos  = np.array([recorder.base_desired_pos[i] for i in indices])  # (T, 3)
    kp_arr         = np.array([recorder.kp[i]      for i in indices])
    kd_arr         = np.array([recorder.kd[i]      for i in indices])
    grip_arr       = np.array([recorder.gripper[i] for i in indices])
    ts             = np.array(indices, dtype=np.float32)

    R_w2r, t_w2r = _build_world_to_robot(world_in_robot_translation, world_in_robot_quat_wxyz)
    raw_pcds = [recorder.point_clouds[i] for i in indices]
    has_pcd = any(p is not None and len(p) > 0 for p in raw_pcds)

    pcd_robot: list[np.ndarray] = []
    if has_pcd:
        empty = np.zeros((0, 3), dtype=np.float32)
        for pts in raw_pcds:
            if pts is None or len(pts) == 0:
                pcd_robot.append(empty)
                continue
            transformed = _apply_world_to_robot(pts.astype(np.float64), R_w2r, t_w2r).astype(np.float32)
            if len(transformed) > pcd_max_pts:
                idx = np.random.choice(len(transformed), pcd_max_pts, replace=False)
                transformed = transformed[idx]
            pcd_robot.append(transformed)

    fk_frames = [(_skeleton_pts(q), _fk_pose(q)) for q in joint_angles]
    skeletons = [item[0] for item in fk_frames]
    ee_poses = np.array([item[1] for item in fk_frames], dtype=np.float32)

    chunk_events = recorder.chunk_events  # sorted ascending by step

    pcd_for_bbox = [p[:, :3] for p in pcd_robot if len(p) > 0] if has_pcd else []
    chunk_pts = [ev["base_traj"] for ev in chunk_events]
    all_xyz = np.concatenate(
        [actual_pos] + skeletons + chunk_pts + pcd_for_bbox if chunk_pts
        else [actual_pos] + skeletons + pcd_for_bbox,
        axis=0,
    )
    mn, mx = all_xyz.min(0), all_xyz.max(0)
    pad = max(float((mx - mn).max()) * 0.05, _FORECAST_AXIS_LENGTH * 1.5)
    x_range = [float(mn[0] - pad), float(mx[0] + pad)]
    y_range = [float(mn[1] - pad), float(mx[1] + pad)]
    z_range = [float(mn[2] - pad), float(mx[2] + pad)]
    extents = np.array(
        [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]],
        dtype=np.float32,
    )
    ext_max = float(extents.max())
    aspectratio = dict(
        x=float(extents[0] / ext_max),
        y=float(extents[1] / ext_max),
        z=float(extents[2] / ext_max),
    )

    _init_ev = _active_chunk_event(chunk_events, indices[0])
    _init_base_fcast = _init_ev["base_traj"] if _init_ev else _EMPTY_FORECAST
    _init_base_pose = _init_ev["base_traj_pose"] if (_init_ev and _init_ev.get("base_traj_pose") is not None) else np.zeros((0, 7), dtype=np.float32)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "xy"}],
            [None,                             {"type": "xy"}],
            [None,                             {"type": "xy"}],
        ],
        column_widths=[0.65, 0.35],
        row_heights=[0.33, 0.33, 0.34],
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    # Trace 0: skeleton
    fig.add_trace(_skeleton_trace(skeletons[0]), row=1, col=1)
    # Trace 1: actual EE trail
    fig.add_trace(
        _trail_trace(actual_pos[:1], color="slategray", name="actual EE", width=3, marker_size=2),
        row=1, col=1,
    )
    # Trace 2: base chunk forecast (full projected trajectory from most recent inference)
    fig.add_trace(
        _trail_trace(_init_base_fcast, color="royalblue", name="base chunk forecast"),
        row=1, col=1,
    )
    # Trace 3-5: live EE orientation triad.
    for axis_trace in _pose_axes_traces(ee_poses[:1], "current ee", _FORECAST_AXIS_LENGTH, colors=_CURRENT_AXIS_COLORS, width=5, opacity=0.95):
        fig.add_trace(axis_trace, row=1, col=1)
    # Trace 6-8: base forecast orientation triad trail.
    for axis_trace in _pose_axes_traces(_init_base_pose, "base forecast", _FORECAST_AXIS_LENGTH, colors=_BASE_AXIS_COLORS, width=4, opacity=0.68):
        fig.add_trace(axis_trace, row=1, col=1)
    # Traces 9-10: kp
    fig.add_trace(_metric_trace(ts[:1], kp_arr[:1], "crimson", "kp_gain"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**kp_arr[:1], mode="lines",
                             line=dict(color="crimson", width=2, dash="dot"), name="kp_true"),
                  row=1, col=2)
    # Traces 11-12: kd
    fig.add_trace(_metric_trace(ts[:1], kd_arr[:1], "seagreen", "kd_gain"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**(kd_arr[:1] * 2 * np.sqrt(kp_arr[:1])), mode="lines",
                             line=dict(color="seagreen", width=2, dash="dot"), name="kd_true"),
                  row=2, col=2)
    # Trace 13: gripper
    fig.add_trace(_metric_trace(ts[:1], grip_arr[:1], "darkorchid", "gripper"), row=3, col=2)
    # Trace 14 (optional): point cloud
    if has_pcd:
        fig.add_trace(_pcd_trace(pcd_robot[0] if len(pcd_robot[0]) > 0 else np.zeros((1, 3), dtype=np.float32)), row=1, col=1)

    animated_traces = list(range(14))
    if has_pcd:
        animated_traces.append(14)

    frames = []
    for t in range(T):
        skel = skeletons[t]
        ap   = actual_pos[:t + 1]
        kp_t = kp_arr[:t + 1]
        kd_t = kd_arr[:t + 1]
        current_pose = ee_poses[t:t + 1]

        ev = _active_chunk_event(chunk_events, indices[t])
        base_fcast = ev["base_traj"] if ev else _EMPTY_FORECAST
        base_pose = ev["base_traj_pose"] if (ev and ev.get("base_traj_pose") is not None) else np.zeros((0, 7), dtype=np.float32)

        frame_data: list[object] = [
            go.Scatter3d(
                x=skel[:, 0], y=skel[:, 1], z=skel[:, 2],
                mode="lines+markers",
                line=dict(color="dimgray", width=6),
                marker=dict(size=5, color="dimgray"),
            ),
            go.Scatter3d(
                x=ap[:, 0], y=ap[:, 1], z=ap[:, 2],
                mode="lines+markers",
                line=dict(color="slategray", width=3),
                marker=dict(size=2, color="slategray"),
            ),
            go.Scatter3d(
                x=base_fcast[:, 0], y=base_fcast[:, 1], z=base_fcast[:, 2],
                mode="lines+markers",
                line=dict(color="royalblue", width=5),
                marker=dict(size=3, color="royalblue"),
            ),
        ]
        frame_data.extend(_pose_axes_traces(current_pose, "current ee", _FORECAST_AXIS_LENGTH, colors=_CURRENT_AXIS_COLORS, width=5, opacity=0.95))
        frame_data.extend(_pose_axes_traces(base_pose, "base forecast", _FORECAST_AXIS_LENGTH, colors=_BASE_AXIS_COLORS, width=4, opacity=0.68))
        frame_data.extend([
            go.Scatter(x=ts[:t + 1], y=kp_t, mode="lines",
                       line=dict(color="crimson", width=2)),
            go.Scatter(x=ts[:t + 1], y=10**kp_t, mode="lines",
                       line=dict(color="crimson", width=2, dash="dot")),
            go.Scatter(x=ts[:t + 1], y=kd_t, mode="lines",
                       line=dict(color="seagreen", width=2)),
            go.Scatter(x=ts[:t + 1], y=10**(kd_t * 2 * np.sqrt(kp_t)), mode="lines",
                       line=dict(color="seagreen", width=2, dash="dot")),
            go.Scatter(x=ts[:t + 1], y=grip_arr[:t + 1], mode="lines",
                       line=dict(color="darkorchid", width=2)),
        ])
        if has_pcd:
            pts = pcd_robot[t]
            if len(pts) == 0:
                pts = np.zeros((1, 3), dtype=np.float32)
            frame_data.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=_pcd_marker(pts),
            ))
        frames.append(go.Frame(data=frame_data, traces=animated_traces, name=str(t)))
    fig.frames = frames

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=10, t=50, b=60),
        scene=dict(
            xaxis=dict(range=x_range, autorange=False, title="x (m)"),
            yaxis=dict(range=y_range, autorange=False, title="y (m)"),
            zaxis=dict(range=z_range, autorange=False, title="z (m)"),
            aspectmode="manual",
            aspectratio=aspectratio,
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0, y=0.0, xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=frame_duration_ms, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="step: "),
            pad=dict(t=40),
            steps=[dict(
                method="animate",
                args=[[str(t)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
                label=str(indices[t]),
            ) for t in range(T)],
        )],
    )

    x_end = float(ts[-1]) + 1
    fig.update_yaxes(title_text="kp",      row=1, col=2)
    fig.update_yaxes(title_text="kd",      row=2, col=2)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="gripper", row=3, col=2)
    fig.update_xaxes(range=[float(ts[0]), x_end], title_text="step", row=1, col=2)
    fig.update_xaxes(range=[float(ts[0]), x_end], title_text="step", row=2, col=2)
    fig.update_xaxes(range=[float(ts[0]), x_end], title_text="step", row=3, col=2)

    _zero_line = dict(color="black", width=1, dash="dot")
    fig.add_trace(
        go.Scatter(x=[float(ts[0]), x_end], y=[0.0, 0.0], mode="lines",
                   line=_zero_line, showlegend=False, hoverinfo="skip"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[float(ts[0]), x_end], y=[0.0, 0.0], mode="lines",
                   line=_zero_line, showlegend=False, hoverinfo="skip"),
        row=2, col=2,
    )

    fig.write_html(path, include_plotlyjs="cdn")
