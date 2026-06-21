"""Shared visualization helpers for sysid scripts.

Separated so viz_sysid.py can produce the same plots as sysid.py without
importing robot-specific dependencies.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent

# Direct file import so this module works from any CWD.
_fk_path = _HERE.parent / "lerobot_teleoperator_gello" / "lerobot_teleoperator_gello" / "franka_fk.py"
_spec = importlib.util.spec_from_file_location("franka_fk", _fk_path)
_fk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fk_mod)
franka_fk_chain = _fk_mod.franka_fk_chain

# Offset subtracted from world-frame eef_pos to recover robot base frame.
WORLD_FRAME_OFFSET = np.array([-0.66, 0.0, 0.912])

def save_comparison_html(
    ref: dict[str, np.ndarray],
    recorded: dict[str, np.ndarray],
    path: str,
    title: str = "Sysid: reference vs replayed trajectory",
    fps: float = 20.0,
    frame_stride: int = 1,
) -> None:
    """Write an animated Plotly HTML comparing reference and replayed EE paths.

    Layout
    ------
    Left (65 %)  : animated arm skeleton + trails:
                     reference EE path (growing, dashed orange — current step)
                     replayed EE path (growing, solid blue — current step)
    Right (35 %) : three stacked time-series panels (static full data + moving cursor):
                     row 1  per-axis position error (x/y/z)
                     row 2  L2 position error norm
                     row 3  per-joint qpos (ref dashed, replayed solid; 7 joints each)
    """
    ref_pos = ref["eef_pos"] - WORLD_FRAME_OFFSET      # (T_ref, 3) — robot frame
    rep_pos = recorded["eef_pos"]                       # (T_rep, 3)
    T_rep = len(rep_pos)
    T_ref = len(ref_pos)
    ts_full = np.arange(T_rep, dtype=np.float32) / fps

    # Position error
    T_min = min(T_ref, T_rep)
    pos_err      = rep_pos[:T_min] - ref_pos[:T_min]   # (T_min, 3)
    pos_err_norm = np.linalg.norm(pos_err, axis=1)     # (T_min,)
    mean_err = float(pos_err_norm.mean())
    max_err  = float(pos_err_norm.max())

    ref_qpos = ref.get("qpos")       # (T_ref, 7) or None
    rep_qpos = recorded.get("qpos")  # (T_rep, 7)

    # Animation indices — driven by the replayed trajectory length
    indices  = list(range(0, T_rep, max(1, frame_stride)))
    T        = len(indices)
    ts_anim  = np.array(indices, dtype=np.float32) / fps
    frame_ms = int(round(1000.0 / max(fps, 1.0)))

    # Down-sampled data for animated trails
    rep_pos_s = rep_pos[indices]                        # (T, 3)
    rep_q_s   = [recorded["qpos"][i] for i in indices] if rep_qpos is not None else None

    # Reference trail: clamp each animation index to T_ref so the reference
    # trail grows in sync but never overshoots its own length.
    ref_indices_s = [min(i, T_ref - 1) for i in indices]
    ref_pos_s     = ref_pos[ref_indices_s]              # (T, 3) — may plateau at end
    ref_q_s       = ([ref_qpos[min(i, T_ref - 1)] for i in indices]
                     if ref_qpos is not None else None)

    # Pre-compute arm skeletons in robot frame
    def _build_skeletons(q_list):
        skels = []
        for q in q_list:
            chain = franka_fk_chain(q)                          # (8,4,4) robot frame
            pts_r = np.vstack([np.zeros((1, 3)), chain[:, :3, 3]])  # (9,3)
            skels.append(pts_r)
        return skels

    rep_skeletons = _build_skeletons(rep_q_s) if rep_q_s is not None else None
    ref_skeletons = _build_skeletons(ref_q_s) if ref_q_s is not None else None

    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Static traces (never updated by frames)
    # -----------------------------------------------------------------------
    # Ghost trails: full paths shown at low opacity for spatial context
    fig.add_trace(go.Scatter3d(
        x=ref_pos[:, 0], y=ref_pos[:, 1], z=ref_pos[:, 2],
        mode="lines",
        line=dict(color="darkorange", width=2),
        opacity=0.15,
        name="reference EE (full)",
        showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=rep_pos[:, 0], y=rep_pos[:, 1], z=rep_pos[:, 2],
        mode="lines",
        line=dict(color="royalblue", width=2),
        opacity=0.15,
        name="replayed EE (full)",
        showlegend=True,
    ), row=1, col=1)

    joint_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
    ts_err = ts_full[:T_min]
    for i, (ax, c) in enumerate(zip("xyz", ["crimson", "seagreen", "steelblue"])):
        fig.add_trace(go.Scatter(
            x=ts_err, y=pos_err[:, i],
            mode="lines", line=dict(color=c, width=2),
            name=f"err_{ax}",
        ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ts_err, y=np.zeros(T_min),
        mode="lines", line=dict(color="black", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=ts_err, y=pos_err_norm,
        mode="lines", line=dict(color="darkorchid", width=2),
        name="L2 err (m)",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[ts_err[0], ts_err[-1]],
        y=[mean_err, mean_err],
        mode="lines", line=dict(color="darkorchid", width=1, dash="dash"),
        name=f"mean {mean_err*1000:.1f} mm",
        showlegend=False, hoverinfo="skip",
    ), row=2, col=2)

    if ref_qpos is not None and rep_qpos is not None:
        T_q = min(len(ref_qpos), T_rep)
        ts_q = ts_full[:T_q]
        for j in range(7):
            c = joint_colors[j]
            fig.add_trace(go.Scatter(
                x=ts_q, y=ref_qpos[:T_q, j],
                mode="lines", line=dict(color=c, width=1.5, dash="dash"),
                name=f"ref q{j+1}", legendgroup=f"q{j+1}",
            ), row=3, col=2)
            fig.add_trace(go.Scatter(
                x=ts_q, y=rep_qpos[:T_q, j],
                mode="lines", line=dict(color=c, width=2),
                name=f"rep q{j+1}", legendgroup=f"q{j+1}",
            ), row=3, col=2)
    else:
        vel_norm = np.linalg.norm(recorded["eef_lin_vel"], axis=1)
        fig.add_trace(go.Scatter(
            x=ts_full, y=vel_norm,
            mode="lines", line=dict(color="steelblue", width=2),
            name="|EE vel| (m/s)",
        ), row=3, col=2)

    # -----------------------------------------------------------------------
    # Animated traces (updated by each go.Frame)
    # -----------------------------------------------------------------------
    anim_idxs: list[int] = []

    def _add_anim(trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        anim_idxs.append(len(fig.data) - 1)

    # --- Reference skeleton (animated) ---
    ref_skel0 = ref_skeletons[0] if ref_skeletons else np.zeros((2, 3))
    _add_anim(go.Scatter3d(
        x=ref_skel0[:, 0], y=ref_skel0[:, 1], z=ref_skel0[:, 2],
        mode="lines+markers",
        line=dict(color="darkorange", width=4),
        marker=dict(size=4, color="darkorange"),
        name="ref skeleton",
    ), 1, 1)

    # --- Replayed skeleton (animated) ---
    rep_skel0 = rep_skeletons[0] if rep_skeletons else np.zeros((2, 3))
    _add_anim(go.Scatter3d(
        x=rep_skel0[:, 0], y=rep_skel0[:, 1], z=rep_skel0[:, 2],
        mode="lines+markers",
        line=dict(color="dimgray", width=6),
        marker=dict(size=5, color="dimgray"),
        name="rep skeleton",
    ), 1, 1)

    # --- Reference EE growing trail (animated) ---
    _add_anim(go.Scatter3d(
        x=ref_pos_s[:1, 0], y=ref_pos_s[:1, 1], z=ref_pos_s[:1, 2],
        mode="lines+markers",
        line=dict(color="darkorange", width=4, dash="dash"),
        marker=dict(size=3, color="darkorange"),
        name="reference EE",
    ), 1, 1)

    # --- Replayed EE growing trail (animated) ---
    _add_anim(go.Scatter3d(
        x=rep_pos_s[:1, 0], y=rep_pos_s[:1, 1], z=rep_pos_s[:1, 2],
        mode="lines+markers",
        line=dict(color="royalblue", width=4),
        marker=dict(size=3, color="royalblue"),
        name="replayed EE",
    ), 1, 1)

    def _yrange(arr, pad=0.05):
        lo, hi = float(arr.min()), float(arr.max())
        span = max(hi - lo, 0.01)
        return lo - span * pad, hi + span * pad

    yr_err = _yrange(pos_err)
    yr_l2  = (0.0, float(pos_err_norm.max()) * 1.1 + 1e-4)
    yr_q   = _yrange(rep_qpos) if rep_qpos is not None else (-1.0, 1.0)

    def _cursor(t_val, y_lo, y_hi):
        return go.Scatter(
            x=[t_val, t_val], y=[y_lo, y_hi],
            mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        )

    _add_anim(_cursor(0.0, *yr_err), 1, 2)
    _add_anim(_cursor(0.0, *yr_l2),  2, 2)
    _add_anim(_cursor(0.0, *yr_q),   3, 2)

    # -----------------------------------------------------------------------
    # Animation frames
    # -----------------------------------------------------------------------
    frames = []
    for fi, t_val in enumerate(ts_anim):
        fd: list = []

        # Reference skeleton
        ref_skel = ref_skeletons[fi] if ref_skeletons else np.zeros((2, 3))
        fd.append(go.Scatter3d(
            x=ref_skel[:, 0], y=ref_skel[:, 1], z=ref_skel[:, 2],
            mode="lines+markers",
            line=dict(color="darkorange", width=4),
            marker=dict(size=4, color="darkorange"),
        ))

        # Replayed skeleton
        rep_skel = rep_skeletons[fi] if rep_skeletons else np.zeros((2, 3))
        fd.append(go.Scatter3d(
            x=rep_skel[:, 0], y=rep_skel[:, 1], z=rep_skel[:, 2],
            mode="lines+markers",
            line=dict(color="dimgray", width=6),
            marker=dict(size=5, color="dimgray"),
        ))

        # Reference EE growing trail
        rp_ref = ref_pos_s[:fi + 1]
        fd.append(go.Scatter3d(
            x=rp_ref[:, 0], y=rp_ref[:, 1], z=rp_ref[:, 2],
            mode="lines+markers",
            line=dict(color="darkorange", width=4, dash="dash"),
            marker=dict(size=3, color="darkorange"),
        ))

        # Replayed EE growing trail
        rp_rep = rep_pos_s[:fi + 1]
        fd.append(go.Scatter3d(
            x=rp_rep[:, 0], y=rp_rep[:, 1], z=rp_rep[:, 2],
            mode="lines+markers",
            line=dict(color="royalblue", width=4),
            marker=dict(size=3, color="royalblue"),
        ))

        fd.append(_cursor(float(t_val), *yr_err))
        fd.append(_cursor(float(t_val), *yr_l2))
        fd.append(_cursor(float(t_val), *yr_q))

        frames.append(go.Frame(data=fd, traces=anim_idxs, name=str(fi)))

    fig.frames = frames

    # -----------------------------------------------------------------------
    # 3D scene bounds
    # -----------------------------------------------------------------------
    all_xyz = np.concatenate(
        [ref_pos, rep_pos]
        + (ref_skeletons or [])
        + (rep_skeletons or []),
        axis=0,
    )
    mn, mx  = all_xyz.min(0), all_xyz.max(0)
    pad     = float(max((mx - mn).max() * 0.05, 0.01))
    extents = mx - mn
    ext_max = float(extents.max()) or 1.0
    aspect  = dict(x=float(extents[0]/ext_max), y=float(extents[1]/ext_max), z=float(extents[2]/ext_max))

    t_end = float(ts_full[-1]) + 0.1
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>max err {max_err*1000:.1f} mm | mean err {mean_err*1000:.1f} mm</sup>",
            x=0.5, xanchor="center",
        ),
        showlegend=True,
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.7)", font=dict(size=9)),
        margin=dict(l=0, r=10, t=60, b=60),
        scene=dict(
            xaxis=dict(range=[float(mn[0]-pad), float(mx[0]+pad)], autorange=False, title="x (m)"),
            yaxis=dict(range=[float(mn[1]-pad), float(mx[1]+pad)], autorange=False, title="y (m)"),
            zaxis=dict(range=[float(mn[2]-pad), float(mx[2]+pad)], autorange=False, title="z (m)"),
            aspectmode="manual", aspectratio=aspect,
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.0, y=0.0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(
                    frame=dict(duration=frame_ms, redraw=True),
                    fromcurrent=True, transition=dict(duration=0),
                )]),
                dict(label="Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate",
                )]),
            ],
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="step: "),
            pad=dict(t=40),
            steps=[dict(
                method="animate",
                args=[[str(fi)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
                label=str(indices[fi]),
            ) for fi in range(T)],
        )],
    )

    fig.update_yaxes(title_text="pos error (m)", row=1, col=2, range=list(yr_err))
    fig.update_yaxes(title_text="L2 error (m)",  row=2, col=2, range=list(yr_l2))
    fig.update_yaxes(title_text="q (rad)",        row=3, col=2)
    for r in (1, 2, 3):
        fig.update_xaxes(title_text="time (s)", row=r, col=2, range=[0.0, t_end])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"visualization saved to {path}")