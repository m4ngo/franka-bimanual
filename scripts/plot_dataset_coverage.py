#!/usr/bin/env python3
"""Coverage figure for one episode HDF5 file: where the end-effector went, how
it was oriented, and how each joint moved over time.

Every episode in the file (EPISODE_HDF5.md) is drawn together, so the union is
the dataset's coverage -- what a plant fit or a policy trained on it has
actually seen. One colour per episode across every panel; clicking an episode
in the legend hides it everywhere.

  python scripts/plot_dataset_coverage.py multi-fast/cfg/sysid/9-15-sysid-dataset.hdf5
  python scripts/plot_dataset_coverage.py <file.hdf5> --output coverage.html

Panels
  top left    every EE path in 3D, in the file's own frame, start of each marked
  top right   EE orientation as a rotation vector about the base axes, relative
              to the dataset's mean orientation; the cloud's extent is the
              rotation coverage
  middle      each joint's span in the dataset against the FR3 range, with
              every episode's own span drawn inside it
  bottom      joint angles over time, episodes laid end to end with a strip
              naming each one, one row per joint so each joint's excitation is
              visible on its own scale

The same numbers print to stdout. Sim-format files (the sysid collect layout)
carry the same field names and are read the same way; their rate comes from
`t_sim` or, failing that, the configured control rate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "multi-fast"))

import franka_config as fc  # noqa: E402
from utils.sysid import episode_hdf5  # noqa: E402

NUM_JOINTS = 7

# Up to eight episodes get a fixed categorical hue each; past that, episodes are
# coloured by their order on a sequential ramp so the legend still reads.
_EPISODE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_RANGE_COLOR = "#e4e3df"
_SPAN_COLOR = "#52514e"
_BOUNDARY_COLOR = "#b3b2ab"


# ---------------------------------------------------------------------- io

def load(path: Path) -> list[dict]:
    """One dict per episode: name, qpos (T,7), eef_pos (T,3), eef_quat (T,4)
    xyzw, fps, and the frame the poses are in."""
    out = []
    for name, arrays, attrs, _ in episode_hdf5.read_episodes(path):
        attrs = episode_hdf5.with_defaults(attrs)
        missing = [k for k in ("qpos", "eef_pos", "eef_quat") if k not in arrays]
        if missing:
            raise ValueError(f"{path}:{name} has no {', '.join(missing)}")
        out.append({
            "name": name,
            "qpos": np.asarray(arrays["qpos"], dtype=np.float64).reshape(-1, NUM_JOINTS),
            "eef_pos": np.asarray(arrays["eef_pos"], dtype=np.float64).reshape(-1, 3),
            "eef_quat": np.asarray(arrays["eef_quat"], dtype=np.float64).reshape(-1, 4),
            "fps": _episode_fps(arrays, attrs),
            "frame": attrs.get("frame", "?"),
        })
    if not out:
        raise ValueError(f"{path} holds no episodes")
    return out


def _episode_fps(arrays: dict, attrs: dict) -> float:
    if "fps" in attrs:
        return float(attrs["fps"])
    t = arrays.get("t_sim")
    if t is not None and len(t) > 1:
        dt = np.median(np.diff(np.asarray(t, dtype=np.float64)))
        if dt > 0:
            return 1.0 / dt
    return float(fc.control_fps())


def joint_limits() -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(fc.control("franka.joint_position_min_rad"), dtype=np.float64)
    hi = np.asarray(fc.control("franka.joint_position_max_rad"), dtype=np.float64)
    return lo, hi


# ---------------------------------------------------------------------- math

def mean_quat(q_xyzw: np.ndarray) -> np.ndarray:
    """Chordal mean of unit quaternions: the top eigenvector of sum(q q^T),
    which is unaffected by the sign of each q."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True).clip(1e-9)
    w, v = np.linalg.eigh(q.T @ q)
    return v[:, np.argmax(w)]


def rotvecs_about_base(q_xyzw: np.ndarray, q_ref_xyzw: np.ndarray) -> np.ndarray:
    """(T,3) rotation vectors taking the reference orientation to each sample,
    expressed in the base frame: rotvec(R(t) R_ref^-1)."""
    return (Rotation.from_quat(q_xyzw) * Rotation.from_quat(q_ref_xyzw).inv()).as_rotvec()


def episode_colors(n: int) -> list[str]:
    if n <= len(_EPISODE_COLORS):
        return list(_EPISODE_COLORS[:n])
    return plotly.colors.sample_colorscale("Viridis", np.linspace(0.0, 0.85, n))


def _axis_refs(fig: go.Figure, row: int, col: int = 1) -> tuple[str, str]:
    """("x3", "y3")-style references for one xy subplot, read off the axes'
    anchors rather than assumed from the row: scenes take no xy numbers."""
    sp = fig.get_subplot(row, col)
    return sp.yaxis.anchor, sp.xaxis.anchor


# ---------------------------------------------------------------------- figure

def build_figure(episodes: list[dict], lo: np.ndarray, hi: np.ndarray, title: str) -> go.Figure:
    colors = episode_colors(len(episodes))
    q_ref = mean_quat(np.concatenate([e["eef_quat"] for e in episodes]))
    strip_row = 3
    joint_rows = list(range(4, 4 + NUM_JOINTS))
    fig = make_subplots(
        rows=3 + NUM_JOINTS, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "xy", "colspan": 2}, None],
               [{"type": "xy", "colspan": 2}, None]]
              + [[{"type": "xy", "colspan": 2}, None] for _ in range(NUM_JOINTS)],
        row_heights=[0.30, 0.14, 0.025] + [0.0765] * NUM_JOINTS,
        vertical_spacing=0.016,
        horizontal_spacing=0.04,
        subplot_titles=("EE position", "EE orientation (rotation from the mean)",
                        "joint span in the dataset (rad) against the FR3 range")
                       + ("",) * (1 + NUM_JOINTS),
    )
    for a in fig.layout.annotations:
        a.font.size = 12
    range_x, range_y = _axis_refs(fig, 2)
    joint_refs = [_axis_refs(fig, r) for r in joint_rows]

    t0 = 0.0
    for i, (e, c) in enumerate(zip(episodes, colors)):
        name, T = e["name"], len(e["qpos"])
        pos, rot = e["eef_pos"], rotvecs_about_base(e["eef_quat"], q_ref)
        hover = "%{text}<br>x %{x:.3f}  y %{y:.3f}  z %{z:.3f}<extra></extra>"
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines",
            line=dict(color=c, width=3), text=[name] * T, hovertemplate=hover,
            name=name, legendgroup=name, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=pos[:1, 0], y=pos[:1, 1], z=pos[:1, 2], mode="markers",
            marker=dict(color=c, size=4), text=[name + " start"], hovertemplate=hover,
            name=name, legendgroup=name, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=rot[:, 0], y=rot[:, 1], z=rot[:, 2], mode="lines",
            line=dict(color=c, width=3), text=[name] * T, hovertemplate=hover,
            name=name, legendgroup=name, showlegend=False,
        ), row=1, col=2)

        # This episode's span per joint, as a thin line inside the joint's bar.
        y_off = -0.22 + 0.44 * (i + 0.5) / len(episodes)
        for j in range(NUM_JOINTS):
            fig.add_trace(go.Scatter(
                x=[e["qpos"][:, j].min(), e["qpos"][:, j].max()], y=[j + y_off] * 2,
                mode="lines", line=dict(color=c, width=2),
                hovertemplate=f"{name}<br>joint {j + 1}: %{{x:.3f}} rad<extra></extra>",
                name=name, legendgroup=name, showlegend=False,
            ), row=2, col=1)

        t = t0 + np.arange(T) / e["fps"]
        for j, r in enumerate(joint_rows):
            fig.add_trace(go.Scatter(
                x=t, y=e["qpos"][:, j], mode="lines", line=dict(color=c, width=1.2),
                hovertemplate=f"{name}<br>t %{{x:.2f}} s<br>joint {j + 1}: %{{y:.3f}} rad<extra></extra>",
                name=name, legendgroup=name, showlegend=False,
            ), row=r, col=1)
        # The strip names each episode's stretch of the time axis; add_vline(row=)
        # trips over the 3D traces, so the boundaries are shapes by axis name.
        fig.add_trace(go.Bar(
            base=[t0], x=[T / e["fps"]], y=[0], orientation="h", width=1.0,
            marker=dict(color=c, line=dict(width=0)), text=name, textposition="inside",
            insidetextanchor="start", textfont=dict(size=9, color="white"), constraintext="inside",
            hovertemplate=f"{name}<br>%{{base:.1f}} .. %{{customdata:.1f}} s<extra></extra>",
            customdata=[t0 + T / e["fps"]],
            name=name, legendgroup=name, showlegend=False,
        ), row=strip_row, col=1)
        if t0 > 0:
            for xr, yr in joint_refs:
                fig.add_shape(type="line", xref=xr, yref=f"{yr} domain", x0=t0, x1=t0, y0=0, y1=1,
                              line=dict(color=_BOUNDARY_COLOR, width=1, dash="dot"))
        t0 = t[-1] + 1.0 / e["fps"]

    # The mean orientation is the origin of the rotation panel.
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0], mode="markers",
        marker=dict(color=_SPAN_COLOR, size=5, symbol="x"),
        hovertemplate="mean orientation<extra></extra>", showlegend=False,
    ), row=1, col=2)

    # Joint range bars: the FR3 range in grey with the dataset's union span on top.
    q_all = np.concatenate([e["qpos"] for e in episodes])
    q_lo, q_hi = q_all.min(axis=0), q_all.max(axis=0)
    idx = np.arange(NUM_JOINTS)
    fig.add_trace(go.Bar(
        base=lo, x=hi - lo, y=idx, orientation="h", width=0.66,
        marker=dict(color=_RANGE_COLOR, line=dict(width=0)),
        customdata=np.stack([idx + 1, hi], axis=1),
        hovertemplate="joint %{customdata[0]}: FR3 range %{base:.3f} .. %{customdata[1]:.3f} rad"
                      "<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        base=q_lo, x=q_hi - q_lo, y=idx, orientation="h", width=0.5,
        marker=dict(color=_SPAN_COLOR, opacity=0.35, line=dict(width=0)),
        customdata=np.stack([idx + 1, q_hi, (q_hi - q_lo) / (hi - lo)], axis=1),
        hovertemplate="joint %{customdata[0]}: dataset %{base:.3f} .. %{customdata[1]:.3f} rad"
                      "<br>%{customdata[2]:.1%} of range<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    for j in range(NUM_JOINTS):
        fig.add_annotation(x=hi[j], y=j, xref=range_x, yref=range_y, xanchor="left",
                           text=f" {100 * (q_hi[j] - q_lo[j]) / (hi[j] - lo[j]):.0f}%",
                           showarrow=False, font=dict(size=10, color=_SPAN_COLOR))
    fig.update_yaxes(tickvals=idx, ticktext=[f"j{j + 1}" for j in idx], autorange="reversed",
                     row=2, col=1)

    # The strip and every joint row share one time axis.
    fig.update_xaxes(matches=joint_refs[-1][0], showticklabels=False, row=strip_row, col=1)
    fig.update_yaxes(visible=False, range=[-0.5, 0.5], row=strip_row, col=1)
    for j, r in enumerate(joint_rows):
        if r != joint_rows[-1]:
            fig.update_xaxes(matches=joint_refs[-1][0], showticklabels=False, row=r, col=1)
        fig.update_yaxes(title_text=f"j{j + 1} (rad)", row=r, col=1)
    fig.update_xaxes(title_text="time (s), episodes end to end", row=joint_rows[-1], col=1)

    frame = ", ".join(sorted({e["frame"] for e in episodes}))
    n_steps = sum(len(e["qpos"]) for e in episodes)
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{len(episodes)} episodes, {n_steps} steps, frame {frame}</sup>",
                   x=0.5, xanchor="center"),
        height=1900, barmode="overlay", template="plotly_white", uniformtext=dict(mode="hide", minsize=7),
        legend=dict(font=dict(size=9), itemsizing="constant", groupclick="togglegroup"),
        margin=dict(l=40, r=20, t=80, b=40),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)", aspectmode="data"),
        scene2=dict(xaxis_title="about x (rad)", yaxis_title="about y (rad)",
                    zaxis_title="about z (rad)", aspectmode="data"),
    )
    return fig


# ---------------------------------------------------------------------- summary

def print_summary(path: Path, episodes: list[dict], lo: np.ndarray, hi: np.ndarray) -> None:
    pos = np.concatenate([e["eef_pos"] for e in episodes])
    quat = np.concatenate([e["eef_quat"] for e in episodes])
    q = np.concatenate([e["qpos"] for e in episodes])
    rot = rotvecs_about_base(quat, mean_quat(quat))
    duration = sum(len(e["qpos"]) / e["fps"] for e in episodes)
    print(f"{path}: {len(episodes)} episodes, {len(q)} steps, {duration:.1f} s, "
          f"frame {', '.join(sorted({e['frame'] for e in episodes}))}")
    for e in episodes:
        print(f"  {e['name']:<32} {len(e['qpos']):5d} steps  {len(e['qpos']) / e['fps']:6.1f} s  @ {e['fps']:g} Hz")

    print(f"\n  {'':<20}{'min':>9}{'max':>9}{'span':>9}")
    for i, ax in enumerate("xyz"):
        print(f"  {f'EE {ax} (m)':<20}{pos[:, i].min():9.3f}{pos[:, i].max():9.3f}{np.ptp(pos[:, i]):9.3f}")
    for i, ax in enumerate("xyz"):
        print(f"  {f'rot about {ax} (rad)':<20}{rot[:, i].min():9.3f}{rot[:, i].max():9.3f}{np.ptp(rot[:, i]):9.3f}")
    print(f"  largest rotation from the mean orientation: {np.linalg.norm(rot, axis=1).max():.3f} rad")

    print(f"\n  {'joint':<7}{'min':>9}{'max':>9}{'span':>9}   FR3 range          covered")
    for j in range(NUM_JOINTS):
        span = np.ptp(q[:, j])
        print(f"  j{j + 1:<6}{q[:, j].min():9.3f}{q[:, j].max():9.3f}{span:9.3f}   "
              f"[{lo[j]:7.3f}, {hi[j]:7.3f}]   {100 * span / (hi[j] - lo[j]):5.1f}%")


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="episode HDF5 file (EPISODE_HDF5.md layout, or the sysid collect layout)")
    ap.add_argument("--output", default=None,
                    help="HTML to write (default: <dataset>_coverage.html beside the file)")
    args = ap.parse_args()

    path = Path(args.dataset).expanduser()
    if not path.is_file():
        ap.error(f"{path} not found")
    out = Path(args.output).expanduser() if args.output else path.with_name(path.stem + "_coverage.html")

    episodes = load(path)
    lo, hi = joint_limits()
    print_summary(path, episodes, lo, hi)
    fig = build_figure(episodes, lo, hi, path.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"\ncoverage figure saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
