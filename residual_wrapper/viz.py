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
        """
        self.chunk_events.append({
            "step": int(step),
            "ee_pos": np.asarray(ee_pos[:3], dtype=np.float32).copy(),
            "base_traj": np.asarray(base_traj, dtype=np.float32).copy(),
            "total_traj": np.asarray(total_traj, dtype=np.float32).copy(),
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
        return dict(size=2, color=colors, opacity=1.0)
    return dict(size=2, color="dimgray", opacity=0.65)


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
    skeletons = [_skeleton_pts(q) for q in joint_angles]  # T × (9, 3)

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
    pad = float((mx - mn).max()) * 0.05
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
    # Traces 4-5: kp_gain (solid) and kp_true (dotted)
    fig.add_trace(_metric_trace(ts[:1], kp_arr[:1], "crimson", "kp_gain"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**kp_arr[:1], mode="lines",
                             line=dict(color="crimson", width=2, dash="dot"), name="kp_true"),
                  row=1, col=2)
    # Traces 6-7: kd_gain (solid) and kd_true (dotted)
    fig.add_trace(_metric_trace(ts[:1], kd_arr[:1], "seagreen", "kd_gain"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**(kd_arr[:1] * 2 * np.sqrt(kp_arr[:1])), mode="lines",
                             line=dict(color="seagreen", width=2, dash="dot"), name="kd_true"),
                  row=2, col=2)
    # Trace 8: gripper
    fig.add_trace(
        _metric_trace(ts[:1], grip_arr[:1], "darkorchid", "gripper"),
        row=3, col=2,
    )
    # Trace 9 (optional): point cloud at step 0 in robot space
    if has_pcd:
        fig.add_trace(_pcd_trace(pcd_robot[0] if len(pcd_robot[0]) > 0 else np.zeros((1, 3), dtype=np.float32)), row=1, col=1)

    # --- animation frames ----------------------------------------------------
    # Traces 0-8 are always animated; trace 9 (point cloud) is added when present.
    # Index mapping:
    #   0 skeleton  1 actual  2 total  3 base  4 kp_gain  5 kp_true
    #   6 kd_gain   7 kd_true 8 gripper  [9 point cloud]
    animated_traces = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    if has_pcd:
        animated_traces.append(9)

    frames = []
    for t in range(T):
        skel = skeletons[t]
        ap   = actual_pos[:t + 1]
        kp_t = kp_arr[:t + 1]
        kd_t = kd_arr[:t + 1]

        # Chunk forecast: use the most recent inference event at or before this step.
        ev = _active_chunk_event(chunk_events, indices[t])
        total_fcast = ev["total_traj"] if ev else _EMPTY_FORECAST
        base_fcast  = ev["base_traj"]  if ev else _EMPTY_FORECAST

        frame_data = [
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
            # 4: kp_gain 0..t
            go.Scatter(x=ts[:t + 1], y=kp_t, mode="lines",
                       line=dict(color="crimson", width=2)),
            # 5: kp_true 0..t
            go.Scatter(x=ts[:t + 1], y=10**kp_t, mode="lines",
                       line=dict(color="crimson", width=2, dash="dot")),
            # 6: kd_gain 0..t
            go.Scatter(x=ts[:t + 1], y=kd_t, mode="lines",
                       line=dict(color="seagreen", width=2)),
            # 7: kd_true 0..t
            go.Scatter(x=ts[:t + 1], y=10**(kd_t * 2 * np.sqrt(kp_t)), mode="lines",
                       line=dict(color="seagreen", width=2, dash="dot")),
            # 8: gripper 0..t
            go.Scatter(x=ts[:t + 1], y=grip_arr[:t + 1], mode="lines",
                       line=dict(color="darkorchid", width=2)),
        ]
        if has_pcd:
            pts = pcd_robot[t]
            if len(pts) == 0:
                pts = np.zeros((1, 3), dtype=np.float32)
            # 9: point cloud at step t
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

    skeletons = [_skeleton_pts(q) for q in joint_angles]

    chunk_events = recorder.chunk_events  # sorted ascending by step

    pcd_for_bbox = [p[:, :3] for p in pcd_robot if len(p) > 0] if has_pcd else []
    chunk_pts = [ev["base_traj"] for ev in chunk_events]
    all_xyz = np.concatenate(
        [actual_pos] + skeletons + chunk_pts + pcd_for_bbox if chunk_pts
        else [actual_pos] + skeletons + pcd_for_bbox,
        axis=0,
    )
    mn, mx = all_xyz.min(0), all_xyz.max(0)
    pad = float((mx - mn).max()) * 0.05
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
    # Traces 3-4: kp
    fig.add_trace(_metric_trace(ts[:1], kp_arr[:1], "crimson", "kp_gain"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**kp_arr[:1], mode="lines",
                             line=dict(color="crimson", width=2, dash="dot"), name="kp_true"),
                  row=1, col=2)
    # Traces 5-6: kd
    fig.add_trace(_metric_trace(ts[:1], kd_arr[:1], "seagreen", "kd_gain"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ts[:1], y=10**(kd_arr[:1] * 2 * np.sqrt(kp_arr[:1])), mode="lines",
                             line=dict(color="seagreen", width=2, dash="dot"), name="kd_true"),
                  row=2, col=2)
    # Trace 7: gripper
    fig.add_trace(_metric_trace(ts[:1], grip_arr[:1], "darkorchid", "gripper"), row=3, col=2)
    # Trace 8 (optional): point cloud
    if has_pcd:
        fig.add_trace(_pcd_trace(pcd_robot[0] if len(pcd_robot[0]) > 0 else np.zeros((1, 3), dtype=np.float32)), row=1, col=1)

    animated_traces = [0, 1, 2, 3, 4, 5, 6, 7]
    if has_pcd:
        animated_traces.append(8)

    frames = []
    for t in range(T):
        skel = skeletons[t]
        ap   = actual_pos[:t + 1]
        kp_t = kp_arr[:t + 1]
        kd_t = kd_arr[:t + 1]

        ev = _active_chunk_event(chunk_events, indices[t])
        base_fcast = ev["base_traj"] if ev else _EMPTY_FORECAST

        frame_data = [
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
        ]
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
