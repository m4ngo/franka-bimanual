"""Residual-policy episode visualizer — interactive Plotly HTML.

Call ``EpisodeRecorder.record()`` at each control step, then
``save_episode_html(recorder, path)`` after the loop to write a
self-contained animated HTML containing:

    - 3D arm skeleton animated over the episode (FK chain)
    - Actual EE trail — where the arm actually was (thin gray)
    - Total desired trail — base + residual target (solid blue)
    - Base desired trail  — base-policy-only target (dashed orange)
    - Time-series panels (right column): kp, kd, commanded gripper

Terminology
-----------
actual EE position  : FK(joint_angles from obs) — where the arm physically is.
base desired        : base_chunk[step, :3] — absolute EE target from the base policy (metres).
total desired       : base_chunk[step, :3] + dpos — effective PD target after residual delta
                      (because _ee_pd tracks pos_error = base_target - current + dpos).

The HTML is self-contained via ``include_plotlyjs="cdn"`` and mirrors the
animated-slider pattern from multi-fast/utils/distill/pcd_viz.py.
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lerobot_teleoperator_gello.franka_fk import franka_fk_chain


# ---------------------------------------------------------------------------
# Data recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    """Accumulates per-step state for post-episode visualization.

    Designed to be created by the caller before entering _run_episode and
    accessed afterwards (even if the episode was interrupted early).
    """

    def __init__(self) -> None:
        self.joint_angles: list[np.ndarray] = []      # (7,) per step
        self.actual_ee_pos: list[np.ndarray] = []     # (3,) actual EE via FK
        self.base_desired_pos: list[np.ndarray] = []  # (3,) base-policy absolute target
        self.total_desired_pos: list[np.ndarray] = [] # (3,) base + residual delta target
        self.kp: list[float] = []
        self.kd: list[float] = []
        self.gripper: list[float] = []

    def record(
        self,
        q: np.ndarray,
        actual_ee_pos: np.ndarray,
        base_desired_pos: np.ndarray,
        total_desired_pos: np.ndarray,
        kp: float,
        kd: float,
        gripper: float,
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
        """
        self.joint_angles.append(np.asarray(q, dtype=np.float64).copy())
        self.actual_ee_pos.append(np.asarray(actual_ee_pos[:3], dtype=np.float32).copy())
        self.base_desired_pos.append(np.asarray(base_desired_pos[:3], dtype=np.float32).copy())
        self.total_desired_pos.append(np.asarray(total_desired_pos[:3], dtype=np.float32).copy())
        self.kp.append(float(kp))
        self.kd.append(float(kd))
        self.gripper.append(float(gripper))

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_episode_html(
    recorder: EpisodeRecorder,
    path: str,
    title: str = "Residual episode",
    frame_stride: int = 1,
) -> None:
    """Write a self-contained animated Plotly HTML from recorded episode data.

    Layout:
        Left 65 %  — animated 3D arm skeleton + four EE trajectory trails:
                       actual (thin gray), total desired (solid blue),
                       base desired (dashed orange).
        Right 35 % — stacked time-series: kp (row 1), kd (row 2), gripper (row 3).

    Trace index map (for go.Frame updates):
        0  skeleton        — animated arm FK chain
        1  actual trail    — actual EE path (FK from obs joint angles)
        2  total desired   — base + residual delta target
        3  base desired    — base-policy-only absolute target
        4  kp metric
        5  kd metric
        6  gripper metric
        7  kp zero-line    (static, not animated)
        8  kd zero-line    (static, not animated)

    Args:
        recorder:     Populated EpisodeRecorder.
        path:         Output .html path; parent directories are created.
        title:        Figure title.
        frame_stride: Emit every Nth step in the animation (default 1 = all steps).
                      Use 2–4 to reduce file size for long episodes.
    """
    T_full = len(recorder)
    if T_full == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

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

    # --- forward kinematics --------------------------------------------------
    skeletons = [_skeleton_pts(q) for q in joint_angles]  # T × (9, 3)

    # --- global 3D bbox (prevents camera rescaling between animation frames) -
    all_xyz = np.concatenate(
        [actual_pos, total_des_pos, base_des_pos] + skeletons,
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

    # --- initial traces (step 0) — order must match go.Frame list below ------
    # Trace 0: skeleton
    fig.add_trace(_skeleton_trace(skeletons[0]), row=1, col=1)
    # Trace 1: actual EE trail
    fig.add_trace(
        _trail_trace(actual_pos[:1], color="slategray", name="actual EE",
                     width=3, marker_size=2),
        row=1, col=1,
    )
    # Trace 2: total desired trail (base + residual)
    fig.add_trace(
        _trail_trace(total_des_pos[:1], color="royalblue", name="total desired"),
        row=1, col=1,
    )
    # Trace 3: base desired trail
    fig.add_trace(
        _trail_trace(base_des_pos[:1], color="darkorange", name="base desired", dash="dash"),
        row=1, col=1,
    )
    # Trace 4: kp
    fig.add_trace(_metric_trace(ts[:1], kp_arr[:1], "crimson", "kp"), row=1, col=2)
    # Trace 5: kd
    fig.add_trace(_metric_trace(ts[:1], kd_arr[:1], "seagreen", "kd"), row=2, col=2)
    # Trace 6: gripper
    fig.add_trace(
        _metric_trace(ts[:1], grip_arr[:1], "darkorchid", "gripper"),
        row=3, col=2,
    )

    # --- animation frames ----------------------------------------------------
    frames = []
    for t in range(T):
        skel  = skeletons[t]
        ap    = actual_pos[:t + 1]
        tp    = total_des_pos[:t + 1]
        bp    = base_des_pos[:t + 1]
        frame_data = [
            # 0: skeleton
            go.Scatter3d(
                x=skel[:, 0], y=skel[:, 1], z=skel[:, 2],
                mode="lines+markers",
                line=dict(color="dimgray", width=6),
                marker=dict(size=5, color="dimgray"),
            ),
            # 1: actual EE trail 0..t
            go.Scatter3d(
                x=ap[:, 0], y=ap[:, 1], z=ap[:, 2],
                mode="lines+markers",
                line=dict(color="slategray", width=3),
                marker=dict(size=2, color="slategray"),
            ),
            # 2: total desired trail 0..t
            go.Scatter3d(
                x=tp[:, 0], y=tp[:, 1], z=tp[:, 2],
                mode="lines+markers",
                line=dict(color="royalblue", width=5),
                marker=dict(size=3, color="royalblue"),
            ),
            # 3: base desired trail 0..t
            go.Scatter3d(
                x=bp[:, 0], y=bp[:, 1], z=bp[:, 2],
                mode="lines+markers",
                line=dict(color="darkorange", width=5, dash="dash"),
                marker=dict(size=3, color="darkorange"),
            ),
            # 4: kp 0..t
            go.Scatter(x=ts[:t + 1], y=kp_arr[:t + 1], mode="lines",
                       line=dict(color="crimson", width=2)),
            # 5: kd 0..t
            go.Scatter(x=ts[:t + 1], y=kd_arr[:t + 1], mode="lines",
                       line=dict(color="seagreen", width=2)),
            # 6: gripper 0..t
            go.Scatter(x=ts[:t + 1], y=grip_arr[:t + 1], mode="lines",
                       line=dict(color="darkorchid", width=2)),
        ]
        frames.append(
            go.Frame(data=frame_data, traces=[0, 1, 2, 3, 4, 5, 6], name=str(t))
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
                        frame=dict(duration=80, redraw=True),
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
    x_end = float(ts[-1]) + 1
    fig.update_yaxes(range=[-1.15, 1.15], title_text="kp",      row=1, col=2)
    fig.update_yaxes(range=[-1.15, 1.15], title_text="kd",      row=2, col=2)
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
