"""System-ID trajectory replay: drive the robot along a reference trajectory and record response.

Usage
-----
python sysid/sysid.py <traj.hdf5> [--output sysid/data_replayed.hdf5] [--fps 20] [--kp 0] [--kd 0]

The input HDF5 must have the structure produced by the sim:
    f[group][episode][field] → (T, D) dataset

Fields consumed from the input:
    eef_goal_pos   (T, 3) – EE goal position sent to the controller (not recorded)
    eef_goal_quat  (T, 4) – EE goal quaternion sent to the controller (not recorded)
    eef_pos        (T, 3) – reference EE position (used for error visualisation)
    qpos           (T, 7) – joint angles used to initialise the home pose

Fields recorded in the output HDF5 (same structure):
    action         (T, 7) – [goal_pos(3), goal_quat(4)] commanded at each step
    eef_ang_vel    (T, 3) – actual EE angular velocity from robot state
    eef_lin_vel    (T, 3) – actual EE linear velocity from robot state
    eef_pos        (T, 3) – actual EE position from robot state
    eef_quat       (T, 4) – actual EE quaternion from robot state
    qpos           (T, 7) – actual joint angles
    qvel           (T, 7) – actual joint velocities
    t_sim          (T, 1) – wall-clock time since episode start
    tau_cmd        (T, 7) – zeros (not accessible via current RPC interface)

A comparison HTML visualization is also written alongside the HDF5.
"""

import argparse
import logging
import os
import select
import sys
import termios
import tty
import time
from pathlib import Path

import importlib.util

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "residual_wrapper"))

from env_wrapper import build_action, start_controller  # noqa: E402
from lerobot_robot_bimanual_franka import SingleArmFranka

# Direct file import so the script works from any CWD.
_fk_path = _HERE.parent / "lerobot_teleoperator_gello" / "lerobot_teleoperator_gello" / "franka_fk.py"
_spec = importlib.util.spec_from_file_location("franka_fk", _fk_path)
_fk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fk_mod)
franka_fk_chain = _fk_mod.franka_fk_chain

# ---------------------------------------------------------------------------
# Robot-to-world frame transform
# From SingleArmFrankaConfig defaults (world expressed in robot frame):
#   world_in_robot_translation_m   = (0.669, 0.003, 0.120)
#   world_in_robot_quat_wxyz       = (-0.376557, 0.0, 0.0, 0.926393)
#
# franka_fk / O_T_EE positions are in robot base frame.
# Reference sim trajectories (eef_pos) are in world frame.
# We transform robot-frame data to world frame before plotting.
# ---------------------------------------------------------------------------

# def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
#     q = q / np.linalg.norm(q)
#     w, x, y, z = q
#     return np.array([
#         [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
#         [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
#         [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
#     ], dtype=np.float64)


# _T_W_IN_R = np.array([0.669, 0.003, 0.120])
# _R_W_IN_R = _quat_wxyz_to_mat(np.array([-0.376557, 0.0, 0.0, 0.926393]))
# _R_R_IN_W = _R_W_IN_R.T
# _T_R_IN_W = -_R_R_IN_W @ _T_W_IN_R


# def _to_world(pts: np.ndarray) -> np.ndarray:
#     """Transform (N, 3) points from robot base frame to world frame."""
#     return pts @ _R_R_IN_W.T + _T_R_IN_W

logger = logging.getLogger(__name__)

# Action kp/kd in [-1, 1]; send_action maps them via kp_gain = 10**kp.
# Default 0.0 → kp_gain = 1.0 (minimum, safest for an open-loop sysid replay).
_DEFAULT_KP = 0.0
_DEFAULT_KD = 0.0


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def parse_traj(filename: str) -> dict[str, np.ndarray]:
    """Load the first episode from a sysid HDF5 file.

    Expected structure: f[group_key][episode_key][field] → (T, D) dataset.
    All datasets are copied into numpy arrays so the file can be closed.
    """
    traj: dict[str, np.ndarray] = {}
    with h5py.File(filename, "r") as f:
        group_key = list(f.keys())[0]
        group = f[group_key]
        episode_key = list(group.keys())[2]
        print(episode_key)
        episode = group[episode_key]
        for field in episode:
            traj[field] = episode[field][:]
    return traj


# ---------------------------------------------------------------------------
# Keyboard helpers (identical pattern to run_residual.py)
# ---------------------------------------------------------------------------

def _stdin_key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _read_key() -> str:
    time.sleep(0.03)
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):
        return "right"
    return ""


# ---------------------------------------------------------------------------
# Episode replay loop
# ---------------------------------------------------------------------------

def _run_episode(
    controller: SingleArmFranka,
    traj: dict[str, np.ndarray],
    fps: float = 20.0,
    kp: float = _DEFAULT_KP,
    kd: float = _DEFAULT_KD,
    gripper_norm: float = 1.0,
) -> dict[str, np.ndarray]:
    """Replay *traj* on the robot and record the kinematic response.

    At each step the robot kinematic state is read via a direct call to
    ``controller.robot_manager.current_kinematic_state_batch`` (which returns
    q, dq, jacobian, ee_pos, ee_quat, ee_vel_6d).  That snapshot is stored in
    ``controller._cached_kin_state`` so the subsequent ``send_action`` call
    consumes it without an extra RPyC round-trip.

    Press right-arrow to end early, Ctrl-C to abort.

    Returns a dict of stacked numpy arrays (one row per step).
    """
    n_steps = len(traj["eef_pos"])
    goal_pos_all = traj["eef_goal_pos"]    # (T, 3) — controller target, not recorded
    goal_quat_all = traj["eef_goal_quat"]  # (T, 4) — controller target, not recorded
    action_all = traj["action"]  # (T, 7)

    buf: dict[str, list] = {k: [] for k in (
        "action", "eef_ang_vel",
        "eef_lin_vel", "eef_pos", "eef_quat", "qpos", "qvel", "t_sim", "tau_cmd",
    )}

    dt = 1.0 / fps
    t_start = time.perf_counter()
    old_term = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        for step in range(n_steps):
            t_step = time.perf_counter()

            if _stdin_key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "right":
                    print("\r\nearly stop requested", flush=True)
                    break

            # --- read kinematic state ----------------------------------------
            # Store in _cached_kin_state so send_action re-uses it, avoiding
            # a redundant RPyC round-trip.
            kin = controller.robot_manager.current_kinematic_state_batch(["r"])
            controller._cached_kin_state = kin
            q, dq, _jac, ee_pos, ee_quat, ee_vel = kin["r"]
            # ee_vel layout: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]

            # --- build and send action ----------------------------------------
            goal_pos = goal_pos_all[step].astype(np.float64)
            goal_quat = goal_quat_all[step].astype(np.float64)
            # chunk_step: [x, y, z, qx, qy, qz, qw, gripper] (8 elements)
            chunk_step = np.concatenate([goal_pos, goal_quat, [gripper_norm]])
            action = build_action(chunk_step, kp=kp, kd=kd)
            dpos: np.ndarray = action_all[step][0:3].astype(np.float64) * 0.05
            drot: np.ndarray = action_all[step][3:6].astype(np.float64) * 0.5
            controller.cache_delta(dpos, drot)
            controller.send_action(action, True)

            # --- record -------------------------------------------------------
            t_now = time.perf_counter() - t_start
            buf["action"].append(np.concatenate([goal_pos, goal_quat]).astype(np.float32))
            buf["eef_ang_vel"].append(np.asarray(ee_vel[3:], dtype=np.float32))
            buf["eef_lin_vel"].append(np.asarray(ee_vel[:3], dtype=np.float32))
            buf["eef_pos"].append(np.asarray(ee_pos, dtype=np.float32))
            buf["eef_quat"].append(np.asarray(ee_quat, dtype=np.float32))
            buf["qpos"].append(np.asarray(q, dtype=np.float32))
            buf["qvel"].append(np.asarray(dq, dtype=np.float32))
            buf["t_sim"].append(np.array([t_now], dtype=np.float32))
            buf["tau_cmd"].append(np.zeros(7, dtype=np.float32))

            elapsed = time.perf_counter() - t_step
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    return {k: np.stack(v) for k, v in buf.items() if v}


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_sysid_hdf5(recorded: dict[str, np.ndarray], path: str) -> None:
    """Write the recorded episode to an HDF5 file with the sim-compatible layout."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        ep = f.create_group("data").create_group("episode_0")
        for field, arr in recorded.items():
            ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
    logger.info("saved %d steps to %s", next(iter(recorded.values())).shape[0], path)


# ---------------------------------------------------------------------------
# Comparison visualization
# ---------------------------------------------------------------------------

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
                     reference EE path (static, dashed orange — full trajectory)
                     replayed EE path (growing, solid blue — current step)
    Right (35 %) : three stacked time-series panels (static full data + moving cursor):
                     row 1  per-axis position error (x/y/z)
                     row 2  L2 position error norm
                     row 3  per-joint qpos (ref dashed, replayed solid; 7 joints each)
    """
    ref_pos = ref["eef_pos"]                           # (T_ref, 3) — world frame
    rep_pos = recorded["eef_pos"]                      # (T_rep, 3)
    T_rep = len(rep_pos)
    ts_full = np.arange(T_rep, dtype=np.float32) / fps

    # Position error
    T_min = min(len(ref_pos), T_rep)
    pos_err      = rep_pos[:T_min] - ref_pos[:T_min]  # (T_min, 3)
    pos_err_norm = np.linalg.norm(pos_err, axis=1)          # (T_min,)
    mean_err = float(pos_err_norm.mean())
    max_err  = float(pos_err_norm.max())

    ref_qpos = ref.get("qpos")        # (T_ref, 7) or None
    rep_qpos = recorded.get("qpos")   # (T_rep, 7)

    # Animation indices
    indices  = list(range(0, T_rep, max(1, frame_stride)))
    T        = len(indices)
    ts_anim  = np.array(indices, dtype=np.float32) / fps
    frame_ms = int(round(1000.0 / max(fps, 1.0)))

    # Down-sampled replayed data for animated trails (already world frame)
    rep_pos_s = rep_pos[indices]                       # (T, 3)
    rep_q_s   = [recorded["qpos"][i] for i in indices] if rep_qpos is not None else None

    # Pre-compute arm skeletons in world frame
    if rep_q_s is not None:
        skeletons = []
        for q in rep_q_s:
            chain = franka_fk_chain(q)                          # (8,4,4) robot frame
            pts_r = np.vstack([np.zeros((1, 3)), chain[:, :3, 3]])  # (9,3) robot frame
            skeletons.append(pts_r)                  # (9,3) world frame
    else:
        skeletons = None

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
    # Reference EE trail — full, static
    fig.add_trace(go.Scatter3d(
        x=ref_pos[:, 0], y=ref_pos[:, 1], z=ref_pos[:, 2],
        mode="lines",
        line=dict(color="darkorange", width=3, dash="dash"),
        name="reference EE",
    ), row=1, col=1)

    # Time-series: per-axis position error
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

    # Time-series: L2 error
    fig.add_trace(go.Scatter(
        x=ts_err, y=pos_err_norm,
        mode="lines", line=dict(color="darkorchid", width=2),
        name=f"L2 err (m)",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[ts_err[0], ts_err[-1]],
        y=[mean_err, mean_err],
        mode="lines", line=dict(color="darkorchid", width=1, dash="dash"),
        name=f"mean {mean_err*1000:.1f} mm",
        showlegend=False, hoverinfo="skip",
    ), row=2, col=2)

    # Time-series: per-joint qpos comparison
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

    skel0 = skeletons[0] if skeletons else np.zeros((2, 3))
    _add_anim(go.Scatter3d(
        x=skel0[:, 0], y=skel0[:, 1], z=skel0[:, 2],
        mode="lines+markers",
        line=dict(color="dimgray", width=6),
        marker=dict(size=5, color="dimgray"),
        name="skeleton",
    ), 1, 1)

    # Replayed EE trail — grows during animation
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

    yr_err  = _yrange(pos_err)
    yr_l2   = (0.0, float(pos_err_norm.max()) * 1.1 + 1e-4)
    yr_q    = _yrange(rep_qpos) if rep_qpos is not None else (-1.0, 1.0)

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

        # skeleton
        skel = skeletons[fi] if skeletons else np.zeros((2, 3))
        fd.append(go.Scatter3d(
            x=skel[:, 0], y=skel[:, 1], z=skel[:, 2],
            mode="lines+markers",
            line=dict(color="dimgray", width=6),
            marker=dict(size=5, color="dimgray"),
        ))

        # replayed EE trail 0..fi
        rp = rep_pos_s[:fi + 1]
        fd.append(go.Scatter3d(
            x=rp[:, 0], y=rp[:, 1], z=rp[:, 2],
            mode="lines+markers",
            line=dict(color="royalblue", width=4),
            marker=dict(size=3, color="royalblue"),
        ))

        # cursors
        fd.append(_cursor(float(t_val), *yr_err))
        fd.append(_cursor(float(t_val), *yr_l2))
        fd.append(_cursor(float(t_val), *yr_q))

        frames.append(go.Frame(data=fd, traces=anim_idxs, name=str(fi)))

    fig.frames = frames

    # -----------------------------------------------------------------------
    # 3D scene bounds (fixed so the camera doesn't jump)
    # -----------------------------------------------------------------------
    all_xyz = np.concatenate([ref_pos, rep_pos] + (skeletons or []), axis=0)
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
    logger.info("visualization saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a reference trajectory on the Franka and record the kinematic response."
    )
    parser.add_argument("traj_file", help="Input HDF5 trajectory file to replay")
    parser.add_argument(
        "--output", default=str(_HERE / "data_replayed.hdf5"),
        help="Output HDF5 file for the recorded response",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Control rate in Hz")
    parser.add_argument(
        "--kp", type=float, default=_DEFAULT_KP,
        help="EE PD kp in [-1, 1]; maps to gain 10**kp (default 0 → gain 1.0)",
    )
    parser.add_argument(
        "--kd", type=float, default=_DEFAULT_KD,
        help="EE PD kd in [-1, 1] (default 0)",
    )
    parser.add_argument("--gripper-norm", type=float, default=1.0,
                        help="Gripper openness [0, 1] held constant during replay")
    parser.add_argument("--home-max-time-s", type=float, default=5.0,
                        help="Maximum seconds allowed for the homing move")
    parser.add_argument("--home-tol-rad", type=float, default=0.05,
                        help="Joint-angle convergence tolerance (rad) for homing")
    parser.add_argument("--home-tol-m", type=float, default=0.025,
                        help="EE position convergence tolerance (m) for homing")
    parser.add_argument("--viz-out", default=None,
                        help="Path for the comparison HTML; defaults to <output>.html")
    parser.add_argument("--viz-stride", type=int, default=1,
                        help="Animate every Nth step in the visualization (use 2-4 for long episodes)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG)

    # Load reference trajectory
    logger.info("loading trajectory from %s", args.traj_file)
    traj = parse_traj(args.traj_file)
    n_steps = len(traj["eef_pos"])
    logger.info("trajectory: %d steps, keys: %s", n_steps, sorted(traj.keys()))

    # Connect robot
    logger.info("connecting to robot...")
    controller = start_controller()
    logger.info("robot connected")

    # Home to first trajectory position (uses qpos from trajectory if available)
    if "qpos" in traj:
        home_q = traj["qpos"][0].astype(np.float64)
        logger.info("homing to first trajectory qpos: %s", np.round(home_q, 3))
    else:
        logger.warning("trajectory has no 'qpos'; skipping pose-specific homing")
        home_q = None

    if home_q is not None:
        converged = controller.home(
            home_q_left=None,
            home_q_right=home_q,
            gripper_norm=args.gripper_norm,
            max_time_s=args.home_max_time_s,
            tol_rad=args.home_tol_rad,
            tol_pos_m=args.home_tol_m,
        )
        if not converged:
            logger.warning("homing did not converge; proceeding anyway")

    # Replay and record
    logger.info(
        "replaying %d steps at %.1f Hz (kp=%.2f → gain=%.2f) — press right-arrow to stop early",
        n_steps, args.fps, args.kp, 10.0 ** args.kp,
    )
    try:
        recorded = _run_episode(
            controller=controller,
            traj=traj,
            fps=args.fps,
            kp=args.kp,
            kd=args.kd,
            gripper_norm=args.gripper_norm,
        )
    finally:
        controller.disconnect()

    if not recorded:
        logger.warning("no steps recorded; exiting")
        return

    # Save HDF5
    save_sysid_hdf5(recorded, args.output)

    # Visualization
    viz_out = args.viz_out or str(Path(args.output).with_suffix(".html"))
    save_comparison_html(traj, recorded, viz_out, fps=args.fps, frame_stride=args.viz_stride)


if __name__ == "__main__":
    main()
