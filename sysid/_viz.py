"""Shared visualization helpers for sysid scripts.

Separated so viz_sysid.py can produce the same plots as sysid.py without
importing robot-specific dependencies.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent

# Direct file import so this module works from any CWD.
_fk_path = _HERE.parent / "lerobot_teleoperator_gello" / "lerobot_teleoperator_gello" / "franka_fk.py"
_spec = importlib.util.spec_from_file_location("franka_fk", _fk_path)
_fk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fk_mod)
franka_fk_chain = _fk_mod.franka_fk_chain

# Offset subtracted from world-frame eef_pos to recover robot base frame.
WORLD_FRAME_OFFSET = np.array([-0.66, 0.0, 0.912])

# --- EE orientation triad settings (mirrors residual wrapper's viz) ---------
_FORECAST_AXIS_LENGTH = 0.038
_REF_AXIS_COLORS = (
    "rgba(235, 95, 35, 0.82)",
    "rgba(95, 180, 70, 0.82)",
    "rgba(55, 120, 220, 0.82)",
)
_REP_AXIS_COLORS = (
    "rgba(220, 40, 40, 0.92)",
    "rgba(40, 170, 80, 0.92)",
    "rgba(50, 90, 235, 0.92)",
)


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


def _normalized_quats(quats_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_xyzw, dtype=np.float64)
    return q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)


def _relative_rotvec_series(quats_xyzw: np.ndarray, quat0_xyzw: np.ndarray | None = None) -> np.ndarray:
    """(T, 4) xyzw quats -> (T, 3) rotation-vectors relative to ``quat0_xyzw``
    (default: the first sample).

    rotvec(t) = as_rotvec(R0^-1 · R(t)) — the rotation accumulated since the
    start, expressed in the start frame. Start-relative series from sim and
    real are directly comparable as long as both trajectories begin at the
    same pose (both are homed to the reference qpos[0]).
    """
    q = _normalized_quats(quats_xyzw)
    q0 = q[0] if quat0_xyzw is None else _normalized_quats(np.asarray(quat0_xyzw)[None, :])[0]
    return (Rotation.from_quat(q0).inv() * Rotation.from_quat(q)).as_rotvec()


def _geodesic_angles(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """(T,) geodesic angle (rad) between two (T, 4) xyzw quaternion series.

    2·arccos(|q1·q2|) — robust to the quaternion double cover.
    """
    a = _normalized_quats(q1_xyzw)
    b = _normalized_quats(q2_xyzw)
    dot = np.clip(np.abs((a * b).sum(axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _fault_steps(fault_count: np.ndarray) -> np.ndarray:
    """Step indices where the cumulative recovery counter incremented."""
    fc = np.asarray(fault_count, dtype=np.int64).ravel()
    return np.flatnonzero(np.diff(np.concatenate([[0], fc])) > 0)


def _align_goal_quats(ref_quat: np.ndarray, goal_quat: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the constant frame offset between eef_goal_quat and eef_quat.

    The sim collector records the OSC goal in the controller's EEF-site frame,
    which on the Panda differs from the recorded eef_quat observable by a
    constant body-frame rotation (measured: 90° about z). Estimated from t=0 —
    valid for probe trajectories whose first commanded delta is ~zero (sines
    start at zero amplitude); for a dataset whose first action carries a real
    rotation delta, up to that one step's delta is absorbed into the estimate.

    Returns (goal_quat corrected into the eef_quat frame family, offset in deg).
    """
    gq = Rotation.from_quat(_normalized_quats(goal_quat))
    C = Rotation.from_quat(_normalized_quats(ref_quat[:1])[0]).inv() * gq[0]
    corrected = (gq * C.inv()).as_quat()
    return corrected, float(np.degrees(C.magnitude()))


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
    colors: tuple[str, str, str],
    width: int = 4,
    opacity: float = 1.0,
    dash: str = "solid",
    legendgroup: str | None = None,
) -> list[go.Scatter3d]:
    """Build three local x/y/z axis traces for a pose trajectory.

    All triads share the same RGB axis colors so x/y/z correspondence holds
    across series; groups are told apart by color family and by toggling.
    When legendgroup is set, the first axis trace carries a legend entry that
    toggles the whole triad (plotly's default groupclick).
    """
    traces: list[go.Scatter3d] = []
    for axis_idx, color in enumerate(colors):
        xs, ys, zs = _pose_axis_segments(poses, axis_idx, length)
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
                name=name_prefix if (legendgroup and axis_idx == 0) else f"{name_prefix} axis {axis_idx}",
                legendgroup=legendgroup,
                showlegend=bool(legendgroup) and axis_idx == 0,
                hoverinfo="skip",
            )
        )
    return traces


def compute_trajectory_errors(
    ref: dict[str, np.ndarray],
    recorded: dict[str, np.ndarray],
    name: str = "",
    joint_vel_l2_max: float = 2.0,   # safety.JOINT_VELOCITY_MAX; literal because this module stays free of robot imports
    saturation_level: float = 0.9,
) -> dict:
    """Position/rotation tracking errors plus run-health diagnostics.

    Returns a dict suitable for inclusion in the errors JSON:
        name, n_steps,
        position_error_m / rotation_error_rad — absolute real-vs-sim errors {mean, max, rms, final}
        start_aligned      — same errors with the t=0 offset removed (homing residual excluded)
        initial_offset     — the t=0 position/rotation offset itself
        goal_decomposition — real and sim lag vs the sim OSC's internal goal (needs ref eef_goal_quat)
        timing_dt_s        — realized control-period stats from recorded t_sim
        qvel_l2            — peak joint-velocity L2 norm and the fraction of steps within
                             `saturation_level` of the safety clamp (those steps identify the
                             safety screen, not the controller)
    """
    ref_pos = ref["eef_pos"] - WORLD_FRAME_OFFSET  # robot frame
    rep_pos = recorded["eef_pos"]
    T_min = min(len(ref_pos), len(rep_pos))

    pos_err_vec  = rep_pos[:T_min] - ref_pos[:T_min]   # (T_min, 3)
    pos_err_norm = np.linalg.norm(pos_err_vec, axis=1)  # (T_min,)
    pos_aligned  = np.linalg.norm(pos_err_vec - pos_err_vec[0], axis=1)

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean":  float(arr.mean()),
            "max":   float(arr.max()),
            "rms":   float(np.sqrt((arr ** 2).mean())),
            "final": float(arr[-1]),
        }

    rot_stats = None
    rot_aligned_stats = None
    initial_rot = None
    goal_decomp = None
    ref_quat = ref.get("eef_quat")
    rep_quat = recorded.get("eef_quat")
    if ref_quat is not None and rep_quat is not None:
        T_q = min(len(ref_quat), len(rep_quat))
        rot_err = _geodesic_angles(ref_quat[:T_q], rep_quat[:T_q])
        rot_stats = _stats(rot_err)
        initial_rot = float(rot_err[0])

        # Start-aligned: angle between the two start-relative rotations, so the
        # homing residual at t=0 doesn't contaminate the tracking comparison.
        def _rel(qs: np.ndarray) -> Rotation:
            q = _normalized_quats(qs[:T_q])
            return Rotation.from_quat(q[0]).inv() * Rotation.from_quat(q)

        rot_aligned_stats = _stats((_rel(ref_quat).inv() * _rel(rep_quat)).magnitude())

        goal_quat = ref.get("eef_goal_quat")
        if goal_quat is not None:
            goal_quat, frame_offset_deg = _align_goal_quats(ref_quat, goal_quat)
            T_g = min(T_q, len(goal_quat))
            goal_decomp = {
                "real_vs_goal_rad": _stats(_geodesic_angles(rep_quat[:T_g], goal_quat[:T_g])),
                "sim_vs_goal_rad":  _stats(_geodesic_angles(ref_quat[:T_g], goal_quat[:T_g])),
                "frame_offset_deg": frame_offset_deg,
            }

    timing = None
    t_sim = recorded.get("t_sim")
    if t_sim is not None and len(t_sim) > 1:
        dts = np.diff(np.asarray(t_sim, dtype=np.float64).ravel())
        timing = {
            "mean": float(dts.mean()),
            "p95":  float(np.percentile(dts, 95)),
            "max":  float(dts.max()),
        }

    qvel_info = None
    qvel = recorded.get("qvel")
    if qvel is not None and len(qvel):
        qn = np.linalg.norm(np.asarray(qvel, dtype=np.float64), axis=1)
        qvel_info = {
            "max": float(qn.max()),
            "saturated_fraction": float((qn > saturation_level * joint_vel_l2_max).mean()),
        }

    faults = None
    if recorded.get("fault_count") is not None and len(recorded["fault_count"]):
        steps = _fault_steps(recorded["fault_count"])
        faults = {"count": len(steps), "step_indices": [int(s) for s in steps]}

    return {
        "name":               name,
        "n_steps":            int(T_min),
        "position_error_m":   _stats(pos_err_norm),
        "rotation_error_rad": rot_stats,
        "start_aligned": {
            "position_error_m":   _stats(pos_aligned),
            "rotation_error_rad": rot_aligned_stats,
        },
        "initial_offset": {
            "position_m":   float(pos_err_norm[0]),
            "rotation_rad": initial_rot,
        },
        "goal_decomposition": goal_decomp,
        "timing_dt_s": timing,
        "qvel_l2": qvel_info,
        "faults": faults,
    }


def save_errors_json(
    trajectory_errors: list[dict],
    path: str,
) -> None:
    """Write per-trajectory and aggregate error stats to a JSON file.

    trajectory_errors: list of dicts returned by compute_trajectory_errors.
    """
    def _agg(key: str, sub: str) -> dict | None:
        vals = [t[key][sub] for t in trajectory_errors if t[key] is not None]
        if not vals:
            return None
        arr = np.array(vals)
        return {
            "mean":  float(arr.mean()),
            "max":   float(arr.max()),
            "total": float(arr.sum()),
        }

    pos_agg = {k: _agg("position_error_m", k) for k in ("mean", "max", "rms", "final")}
    rot_vals = [t["rotation_error_rad"] for t in trajectory_errors if t["rotation_error_rad"] is not None]
    rot_agg: dict | None = None
    if rot_vals:
        rot_agg = {k: _agg("rotation_error_rad", k) for k in ("mean", "max", "rms", "final")}

    # Run-level reflex rollup: which episodes were fault-corrupted, at a glance.
    faulty = {t["name"]: t["faults"]["count"]
              for t in trajectory_errors if t.get("faults") and t["faults"]["count"] > 0}

    payload = {
        "trajectories": trajectory_errors,
        "aggregate": {
            "n_trajectories":    len(trajectory_errors),
            "position_error_m":  pos_agg,
            "rotation_error_rad": rot_agg,
            "faults": {
                "episodes_with_faults": len(faulty),
                "total_recoveries": sum(faulty.values()),
                "by_episode": faulty,
            },
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"error stats saved to {path}")


def save_aggregate_html(
    items: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]],
    path: str,
    title: str = "Sysid aggregate: sim (orange) vs real (blue)",
    fps: float = 20.0,
) -> None:
    """One static overview across all episodes of a run.

    items: list of (episode_name, ref_traj, recorded_traj).

    Layout
    ------
    Left        : 3D overlay of every EE path — sim orange, real blue
                  (Scatter3d has no dash support, so color/opacity carry the coding).
    Right row 1 : start-relative rotation magnitude per episode, sim dashed / real solid.
    Right row 2 : real-vs-sim geodesic rotation error per episode (the drift-curve
                  family), one qualitative color per episode.
    One legend entry per episode toggles all of its traces across panels.
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None,                             {"type": "xy"}],
        ],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        subplot_titles=(None, "start-relative |rotation| (rad)", "rotation error real↔sim (rad)"),
    )
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
               "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
               "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02"]

    for idx, (name, ref, recorded) in enumerate(items):
        ref_pos = ref["eef_pos"] - WORLD_FRAME_OFFSET
        rep_pos = recorded["eef_pos"]
        fig.add_trace(go.Scatter3d(
            x=ref_pos[:, 0], y=ref_pos[:, 1], z=ref_pos[:, 2],
            mode="lines", line=dict(color="darkorange", width=3), opacity=0.45,
            name=name, legendgroup=name, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=rep_pos[:, 0], y=rep_pos[:, 1], z=rep_pos[:, 2],
            mode="lines", line=dict(color="royalblue", width=3), opacity=0.45,
            name=name, legendgroup=name, showlegend=False,
        ), row=1, col=1)

        ref_quat = ref.get("eef_quat")
        rep_quat = recorded.get("eef_quat")
        if ref_quat is None or rep_quat is None:
            continue
        m_ref = np.linalg.norm(_relative_rotvec_series(ref_quat), axis=1)
        m_rep = np.linalg.norm(_relative_rotvec_series(rep_quat), axis=1)
        fig.add_trace(go.Scatter(
            x=np.arange(len(m_ref), dtype=np.float32) / fps, y=m_ref,
            mode="lines", line=dict(color="darkorange", width=1, dash="dash"),
            opacity=0.6, name=name, legendgroup=name, showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=np.arange(len(m_rep), dtype=np.float32) / fps, y=m_rep,
            mode="lines", line=dict(color="royalblue", width=1.5),
            opacity=0.7, name=name, legendgroup=name, showlegend=False,
        ), row=1, col=2)

        T_min = min(len(ref_quat), len(rep_quat))
        gap = _geodesic_angles(rep_quat[:T_min], ref_quat[:T_min])
        fig.add_trace(go.Scatter(
            x=np.arange(T_min, dtype=np.float32) / fps, y=gap,
            mode="lines", line=dict(color=palette[idx % len(palette)], width=1.5),
            name=name, legendgroup=name, showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{len(items)} episodes</sup>", x=0.5, xanchor="center"),
        legend=dict(font=dict(size=9), itemsizing="constant"),
        margin=dict(l=0, r=10, t=70, b=40),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
    )
    fig.update_yaxes(title_text="|rotvec| (rad)",  row=1, col=2)
    fig.update_yaxes(title_text="rot error (rad)", row=2, col=2)
    for r in (1, 2):
        fig.update_xaxes(title_text="time (s)", row=r, col=2)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"aggregate visualization saved to {path}")


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
                     sim OSC goal path (static, dotted — when ref has eef_goal_pos)
                     reference EE orientation triad (animated, orange-family axes)
                     replayed EE orientation triad (animated, red/green/blue axes)
    Right (35 %) : four stacked time-series panels (static full data + moving cursor):
                     row 1  start-relative rotation per axis (real solid, sim dashed,
                            sim OSC goal dotted — when ref has eef_goal_quat)
                     row 2  rotation error angles: real↔sim gap, real↔goal, sim↔goal
                     row 3  per-axis position error (x/y/z) + L2 norm
                     row 4  per-joint qpos (ref dashed, replayed solid; 7 joints each)
    """
    ref_pos = ref["eef_pos"] - WORLD_FRAME_OFFSET      # (T_ref, 3) — robot frame
    rep_pos = recorded["eef_pos"]                       # (T_rep, 3)
    T_rep = len(rep_pos)
    T_ref = len(ref_pos)
    ts_full = np.arange(T_rep, dtype=np.float32) / fps
    ts_ref_full = np.arange(T_ref, dtype=np.float32) / fps

    # Position error
    T_min = min(T_ref, T_rep)
    pos_err      = rep_pos[:T_min] - ref_pos[:T_min]   # (T_min, 3)
    pos_err_norm = np.linalg.norm(pos_err, axis=1)     # (T_min,)
    mean_err = float(pos_err_norm.mean())
    max_err  = float(pos_err_norm.max())

    # Rotation series (start-relative) and sim-OSC-goal overlays
    ref_quat  = ref.get("eef_quat")
    rep_quat  = recorded.get("eef_quat")
    goal_quat = ref.get("eef_goal_quat")
    goal_pos  = ref.get("eef_goal_pos")
    goal_pos  = goal_pos - WORLD_FRAME_OFFSET if goal_pos is not None else None

    have_rot = ref_quat is not None and rep_quat is not None
    rot_ref = rot_rep = rot_goal = None
    rot_err_gap = rot_err_real_goal = rot_err_sim_goal = None
    rot_subtitle = ""
    if have_rot:
        rot_ref = _relative_rotvec_series(ref_quat)     # (T_ref, 3)
        rot_rep = _relative_rotvec_series(rep_quat)     # (T_rep, 3)
        rot_err_gap = _geodesic_angles(rep_quat[:T_min], ref_quat[:T_min])
        if goal_quat is not None:
            # Undo the collector's constant goal-frame offset, then express the
            # goal relative to the reference start so ref/goal share a base.
            goal_quat, _ = _align_goal_quats(ref_quat, goal_quat)
            rot_goal = _relative_rotvec_series(goal_quat, quat0_xyzw=ref_quat[0])
            T_gr = min(T_min, len(goal_quat))
            rot_err_real_goal = _geodesic_angles(rep_quat[:T_gr], goal_quat[:T_gr])
            T_gs = min(T_ref, len(goal_quat))
            rot_err_sim_goal = _geodesic_angles(ref_quat[:T_gs], goal_quat[:T_gs])
        rot_subtitle = (f" | rot err max {np.degrees(rot_err_gap.max()):.1f}°"
                        f" mean {np.degrees(rot_err_gap.mean()):.1f}°")

    fault_steps = (_fault_steps(recorded["fault_count"])
                   if recorded.get("fault_count") is not None else np.array([], dtype=np.int64))
    fault_note = f" | {len(fault_steps)} reflex recoveries" if len(fault_steps) else ""

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

    # Pre-compute EE orientation poses (quat) in robot frame, in sync with the
    # skeletons above, for the animated orientation triads.
    ref_poses = _poses_from_qs(ref_q_s) if ref_q_s is not None else None
    rep_poses = _poses_from_qs(rep_q_s) if rep_q_s is not None else None

    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 4}, {"type": "xy"}],
            [None,                             {"type": "xy"}],
            [None,                             {"type": "xy"}],
            [None,                             {"type": "xy"}],
        ],
        column_widths=[0.65, 0.35],
        row_heights=[0.25, 0.25, 0.25, 0.25],
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
    if goal_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=goal_pos[:, 0], y=goal_pos[:, 1], z=goal_pos[:, 2],
            mode="lines",
            line=dict(color="seagreen", width=2, dash="dot"),
            opacity=0.25,
            name="sim OSC goal (full)",
            showlegend=True,
        ), row=1, col=1)

    joint_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
    axis_colors = ["crimson", "seagreen", "steelblue"]
    ts_err = ts_full[:T_min]

    # Row 1 — start-relative rotation per axis: real solid, sim dashed, goal dotted.
    if have_rot:
        for i, ax in enumerate("xyz"):
            c = axis_colors[i]
            fig.add_trace(go.Scatter(
                x=ts_full, y=rot_rep[:, i],
                mode="lines", line=dict(color=c, width=2),
                name=f"rot_{ax} real", legendgroup=f"rot{ax}",
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=ts_ref_full, y=rot_ref[:, i],
                mode="lines", line=dict(color=c, width=1.5, dash="dash"),
                name=f"rot_{ax} sim", legendgroup=f"rot{ax}",
            ), row=1, col=2)
            if rot_goal is not None:
                fig.add_trace(go.Scatter(
                    x=ts_ref_full[:len(rot_goal)], y=rot_goal[:, i],
                    mode="lines", line=dict(color=c, width=1, dash="dot"),
                    name=f"rot_{ax} goal", legendgroup=f"rot{ax}",
                    showlegend=False,
                ), row=1, col=2)

    # Row 2 — rotation error angles: the sim2real gap, plus each side's lag
    # behind the sim OSC's internal goal (controller-law comparison).
    if rot_err_gap is not None:
        fig.add_trace(go.Scatter(
            x=ts_err[:len(rot_err_gap)], y=rot_err_gap,
            mode="lines", line=dict(color="crimson", width=2),
            name="rot err real↔sim",
        ), row=2, col=2)
        if rot_err_real_goal is not None:
            fig.add_trace(go.Scatter(
                x=ts_full[:len(rot_err_real_goal)], y=rot_err_real_goal,
                mode="lines", line=dict(color="royalblue", width=1.5, dash="dash"),
                name="rot err real↔goal",
            ), row=2, col=2)
        if rot_err_sim_goal is not None:
            fig.add_trace(go.Scatter(
                x=ts_ref_full[:len(rot_err_sim_goal)], y=rot_err_sim_goal,
                mode="lines", line=dict(color="darkorange", width=1.5, dash="dot"),
                name="rot err sim↔goal",
            ), row=2, col=2)

    # Row 3 — per-axis position error + L2 norm.
    for i, (ax, c) in enumerate(zip("xyz", axis_colors)):
        fig.add_trace(go.Scatter(
            x=ts_err, y=pos_err[:, i],
            mode="lines", line=dict(color=c, width=2),
            name=f"err_{ax}",
        ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=ts_err, y=np.zeros(T_min),
        mode="lines", line=dict(color="black", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=ts_err, y=pos_err_norm,
        mode="lines", line=dict(color="darkorchid", width=2),
        name="L2 err (m)",
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=[ts_err[0], ts_err[-1]],
        y=[mean_err, mean_err],
        mode="lines", line=dict(color="darkorchid", width=1, dash="dash"),
        name=f"mean {mean_err*1000:.1f} mm",
        showlegend=False, hoverinfo="skip",
    ), row=3, col=2)

    # Row 4 — per-joint qpos overlay (or EE velocity norm fallback).
    if ref_qpos is not None and rep_qpos is not None:
        T_q = min(len(ref_qpos), T_rep)
        ts_q = ts_full[:T_q]
        for j in range(7):
            c = joint_colors[j]
            fig.add_trace(go.Scatter(
                x=ts_q, y=ref_qpos[:T_q, j],
                mode="lines", line=dict(color=c, width=1.5, dash="dash"),
                name=f"ref q{j+1}", legendgroup=f"q{j+1}",
            ), row=4, col=2)
            fig.add_trace(go.Scatter(
                x=ts_q, y=rep_qpos[:T_q, j],
                mode="lines", line=dict(color=c, width=2),
                name=f"rep q{j+1}", legendgroup=f"q{j+1}",
            ), row=4, col=2)
    else:
        vel_norm = np.linalg.norm(recorded["eef_lin_vel"], axis=1)
        fig.add_trace(go.Scatter(
            x=ts_full, y=vel_norm,
            mode="lines", line=dict(color="steelblue", width=2),
            name="|EE vel| (m/s)",
        ), row=4, col=2)

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

    # --- Reference EE orientation triad (animated, x/y/z axes) ---
    ref_pose0 = ref_poses[:1] if ref_poses is not None else np.zeros((0, 7), dtype=np.float32)
    for axis_trace in _pose_axes_traces(
        ref_pose0, "reference ee", _FORECAST_AXIS_LENGTH,
        colors=_REF_AXIS_COLORS, width=4, opacity=0.85,
        dash="dash", legendgroup="reference ee",
    ):
        _add_anim(axis_trace, 1, 1)

    # --- Replayed EE orientation triad (animated, x/y/z axes) ---
    rep_pose0 = rep_poses[:1] if rep_poses is not None else np.zeros((0, 7), dtype=np.float32)
    for axis_trace in _pose_axes_traces(
        rep_pose0, "replayed ee", _FORECAST_AXIS_LENGTH,
        colors=_REP_AXIS_COLORS, width=5, opacity=0.95,
        legendgroup="replayed ee",
    ):
        _add_anim(axis_trace, 1, 1)

    def _yrange(arr, pad=0.05):
        lo, hi = float(arr.min()), float(arr.max())
        span = max(hi - lo, 0.01)
        return lo - span * pad, hi + span * pad

    if have_rot:
        rot_stack = [rot_rep.ravel(), rot_ref.ravel()]
        if rot_goal is not None:
            rot_stack.append(rot_goal.ravel())
        yr_rot = _yrange(np.concatenate(rot_stack))
        rerr_all = np.concatenate([a for a in (rot_err_gap, rot_err_real_goal, rot_err_sim_goal)
                                   if a is not None])
        yr_rerr = (0.0, float(rerr_all.max()) * 1.1 + 1e-4)
    else:
        yr_rot, yr_rerr = (-1.0, 1.0), (0.0, 1.0)
    yr_pos = _yrange(np.concatenate([pos_err.ravel(), pos_err_norm]))
    yr_q   = _yrange(rep_qpos) if rep_qpos is not None else (-1.0, 1.0)

    # Reflex/recovery markers: deviations at these ticks are fault-caused, not
    # controller-caused. Plain traces — add_vline chokes on scene subplots.
    for r, (lo, hi) in zip((1, 2, 3, 4), (yr_rot, yr_rerr, yr_pos, yr_q)):
        for s in fault_steps:
            fig.add_trace(go.Scatter(
                x=[float(s) / fps] * 2, y=[lo, hi], mode="lines",
                line=dict(color="red", width=1, dash="dot"), opacity=0.6,
                showlegend=False, hoverinfo="skip",
            ), row=r, col=2)

    def _cursor(t_val, y_lo, y_hi):
        return go.Scatter(
            x=[t_val, t_val], y=[y_lo, y_hi],
            mode="lines",
            line=dict(color="black", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        )

    _add_anim(_cursor(0.0, *yr_rot),  1, 2)
    _add_anim(_cursor(0.0, *yr_rerr), 2, 2)
    _add_anim(_cursor(0.0, *yr_pos),  3, 2)
    _add_anim(_cursor(0.0, *yr_q),    4, 2)

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

        # Reference EE orientation triad at this frame (single pose -> 3 axis traces)
        ref_pose_fi = ref_poses[fi:fi + 1] if ref_poses is not None else np.zeros((0, 7), dtype=np.float32)
        fd.extend(_pose_axes_traces(
            ref_pose_fi, "reference ee", _FORECAST_AXIS_LENGTH,
            colors=_REF_AXIS_COLORS, width=4, opacity=0.85, dash="dash",
        ))

        # Replayed EE orientation triad at this frame
        rep_pose_fi = rep_poses[fi:fi + 1] if rep_poses is not None else np.zeros((0, 7), dtype=np.float32)
        fd.extend(_pose_axes_traces(
            rep_pose_fi, "replayed ee", _FORECAST_AXIS_LENGTH,
            colors=_REP_AXIS_COLORS, width=5, opacity=0.95,
        ))

        fd.append(_cursor(float(t_val), *yr_rot))
        fd.append(_cursor(float(t_val), *yr_rerr))
        fd.append(_cursor(float(t_val), *yr_pos))
        fd.append(_cursor(float(t_val), *yr_q))

        frames.append(go.Frame(data=fd, traces=anim_idxs, name=str(fi)))

    fig.frames = frames

    # -----------------------------------------------------------------------
    # 3D scene bounds
    # -----------------------------------------------------------------------
    all_xyz = np.concatenate(
        [ref_pos, rep_pos]
        + ([goal_pos] if goal_pos is not None else [])
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
            text=f"{title}<br><sup>max err {max_err*1000:.1f} mm | mean err {mean_err*1000:.1f} mm"
                 f"{rot_subtitle}{fault_note}</sup>",
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

    fig.update_yaxes(title_text="rotation (rad)",  row=1, col=2, range=list(yr_rot))
    fig.update_yaxes(title_text="rot error (rad)", row=2, col=2, range=list(yr_rerr))
    fig.update_yaxes(title_text="pos error (m)",   row=3, col=2, range=list(yr_pos))
    fig.update_yaxes(title_text="q (rad)",          row=4, col=2)
    for r in (1, 2, 3, 4):
        fig.update_xaxes(title_text="time (s)", row=r, col=2, range=[0.0, t_end])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"visualization saved to {path}")