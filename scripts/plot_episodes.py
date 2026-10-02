#!/usr/bin/env python3
"""Animated Plotly HTML for a real episode -- a reach rollout or a dataset
trajectory -- optionally overlaid with its sim replay.

Same shape as sysid/_viz.py's comparison plot -- one 3D scene animated against
stacked time-series panels, driven by a Play button and a step slider -- with
the reach task's own references in place of the sim trajectory: the commanded
curve, the safety floor, and the keep-out sphere.

Everything drawn is WORLD frame, because that is the frame the floor and the
keep-out sphere are defined in. The episode files (EPISODE_HDF5.md) are base
frame; `base_to_world` maps every array out on load, and is never inverted.

  python scripts/plot_episodes.py ~/franka_data/real_reach/<timestamp>
  python scripts/plot_episodes.py <run_dir>/episodes.hdf5 --episode 2

real_reach_rollout.py calls save_run_html itself, so a live run writes its
HTML next to episodes.hdf5; this is for re-rendering an existing run. A
sim_replay.hdf5 in the run directory (from replay_goals_in_sim.py) is overlaid
episode by episode and the per-episode error summary goes to errors.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
# The error-by-gain table lives sim-side so both replay paths print the same one.
sys.path.insert(0, str(_ROOT / "multi-fast"))

import franka_config as fc  # noqa: E402
from utils.sysid import episode_hdf5  # noqa: E402
from utils.sysid.gains import (  # noqa: E402
	GAIN_AXIS_LEFT, GAIN_AXIS_RIGHT, format_gain_table, gain_panel_traces, gain_terciles,
)
from lerobot_robot_bimanual_franka.franka_fk import franka_fk_chain  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import (  # noqa: E402
	base_to_world, base_to_world_quat, keep_out_sphere, safety_z_floor_world,
	sim_ee_to_robot_ee,
)

# Axis colours for the EE triad, matching sysid/_viz.py's replayed triad so the
# two visualizations read the same way.
_AXIS_LENGTH = 0.05
_TRIAD_COLORS = (
	"rgba(220, 40, 40, 0.92)",
	"rgba(40, 170, 80, 0.92)",
	"rgba(50, 90, 235, 0.92)",
)
_SIM_TRIAD_COLORS = (
	"rgba(235, 95, 35, 0.75)",
	"rgba(95, 180, 70, 0.75)",
	"rgba(55, 120, 220, 0.75)",
)
# Shared with multi-fast/scripts/utils/compare_real_sim.py so the figures in this
# workspace read the same way.
_C_SIM = "#1baf7a"
_JOINT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
_XYZ_COLORS = ("crimson", "seagreen", "steelblue")


# ---------------------------------------------------------------------- io

def load_run(path: str | Path) -> tuple[Path, list[dict]]:
	"""Accept a run directory or its episodes.hdf5; return (run_dir, episodes).

	An episode here is {"name", "attrs", "arrays", "curve"} straight from the
	file, attrs with the legacy frame defaults filled in.
	"""
	p = Path(path).expanduser()
	if p.is_dir():
		p = p / episode_hdf5.REAL_FILE
	if not p.is_file():
		raise FileNotFoundError(f"{p} not found -- pass a run directory or its {episode_hdf5.REAL_FILE}")
	episodes = [{"name": n, "arrays": a, "attrs": episode_hdf5.with_defaults(t), "curve": c}
				for n, a, t, c in episode_hdf5.read_episodes(p)]
	return p.parent, episodes


def load_sim_run(path: str | Path) -> dict[str, dict]:
	"""Every sim episode in a sim_replay.hdf5 (or the run directory holding
	one), by episode name; refuses a file whose frame is not the sim's, because
	a silently-wrong frame renders as a plausible-looking divergence."""
	p = Path(path).expanduser()
	if p.is_dir():
		p = p / episode_hdf5.SIM_FILE
	if not p.is_file():
		return {}
	out = {}
	for n, a, t, c in episode_hdf5.read_episodes(p):
		t = episode_hdf5.with_defaults(t)
		if t.get("frame") != "base_sim" or t.get("ee_convention") != "robosuite_grip_site":
			raise ValueError(f"{p}:{n} frame/ee_convention is {t.get('frame')!r}/"
							 f"{t.get('ee_convention')!r}, not a sim replay")
		out[n] = {"name": n, "arrays": a, "attrs": t, "curve": c}
	return out


def _episode_arrays(ep: dict, arm: str) -> dict:
	"""One real episode's arrays, mapped from the file's base frame to WORLD."""
	arr, at, curve = ep["arrays"], ep["attrs"], ep.get("curve")
	n = int(at.get("num_samples", len(arr["eef_pos"])))

	def col(key, shape):
		v = arr.get(key)
		return (np.zeros((0,) + shape[1:]) if v is None or len(v) == 0
				else np.asarray(v, dtype=np.float64).reshape(shape))

	if curve is not None:
		goal = base_to_world(arm, np.asarray(curve["goal"], dtype=np.float64).reshape(3))
		wp = base_to_world(arm, np.asarray(curve["waypoints"], dtype=np.float64).reshape(-1, 3))
		vel = np.asarray(curve["velocity_scales"], dtype=np.float64).reshape(-1)
	else:
		goal, wp, vel = None, np.zeros((0, 3)), np.zeros(0)
	trace = col("eef_pos", (-1, 3))
	cmd = col("eef_goal_pos", (-1, 3))
	quat = col("eef_quat", (-1, 4))
	out = {
		"goal": goal,
		"waypoints": wp,
		"velocity_scales": vel,
		"trace": base_to_world(arm, trace) if len(trace) else trace,
		"commanded": base_to_world(arm, cmd) if len(cmd) else cmd,
		"ee_quat": (np.asarray([base_to_world_quat(arm, q) for q in quat], dtype=np.float64)
					if len(quat) else np.zeros((0, 4))),
		"qpos": col("qpos", (-1, 7)),
		"cursor_trace": col("cursor", (-1,)),
		"stale_anchor_m": col("anchor_gap_m", (-1,)),
		# The normalised kp/kd action each step sent; row t drove step t, like
		# `commanded`. Empty for records that never carried one. kp/kd are the
		# physical gains the arm resolved from it, when recorded.
		"gain_action": col("gain_action", (-1, 2)),
		"kp": col("kp", (-1, 6)), "kd": col("kd", (-1, 6)),
		"timing_shifted": False,
	}
	if n and at.get("obs_timing") != "post_period":
		# Row t is the state BEFORE action t (excitation runs, dataset conversions,
		# and old records read right after the send): the response to goal t-1,
		# one step behind sim's row t. Row 0 is therefore the start pose; keep it,
		# then shift the measurements left by one so index t means the same thing
		# on both sides. The last goal then has no measured response and is dropped.
		if len(quat):
			out["start"] = {"pos": out["trace"][0], "quat": out["ee_quat"][0],
							"qpos": out["qpos"][0]}
		for k in ("trace", "ee_quat", "qpos", "cursor_trace", "stale_anchor_m"):
			out[k] = out[k][1:]
		out["commanded"] = out["commanded"][:-1]
		for k in ("gain_action", "kp", "kd"):
			out[k] = out[k][:-1]
		out["timing_shifted"] = True
	pos0 = at.get("ee_pos0")
	if pos0 is None and curve is not None:
		pos0 = curve["waypoints"][0]          # a reach curve starts at the reset EE
	if pos0 is not None and n and "start" not in out:
		# t=0: the pose both sides start from, before any command. Drawn so the
		# identical start is visible; the first command applies from here.
		quat0 = at.get("ee_quat0")
		if quat0 is None:
			quat0 = arr["eef_goal_quat"][0]   # zero rotation delta: the latched reset orientation
		out["start"] = {
			"pos": base_to_world(arm, np.asarray(pos0, dtype=np.float64)),
			"quat": base_to_world_quat(arm, np.asarray(quat0, dtype=np.float64)),
			"qpos": np.asarray(at["init_qpos"], dtype=np.float64)}
	return out


def _sim_arrays(sim: dict, arm: str) -> dict:
	"""Sim base-frame grip-site arrays -> WORLD, in the robot's O_T_EE convention.

	Two maps, both in the sanctioned direction: the grip-site -> O_T_EE offset
	the sim replay measured and recorded (a -90 deg tool-z rotation and 7 mm --
	23% of the success threshold), then base->world for the arm that actually
	ran. `robot_base_in_world` is never inverted.
	"""
	arr, at = sim["arrays"], sim["attrs"]
	if at.get("site_in_otee_rotvec") is None:
		raise ValueError("sim episode has no site_in_otee attrs -- re-run replay_goals_in_sim.py")
	rv, sp = np.asarray(at["site_in_otee_rotvec"]), np.asarray(at["site_in_otee_pos"])
	pos = np.asarray(arr["eef_pos"], dtype=np.float64).reshape(-1, 3)
	quat = np.asarray(arr["eef_quat"], dtype=np.float64).reshape(-1, 4)
	gpos = np.asarray(arr["eef_goal_pos"], dtype=np.float64).reshape(-1, 3)
	gquat = np.asarray(arr["eef_goal_quat"], dtype=np.float64).reshape(-1, 4)
	tcp, ee = (zip(*(sim_ee_to_robot_ee(p, q, rv, sp) for p, q in zip(pos, quat)))
			   if len(pos) else ((), ()))
	tcp = np.asarray(tcp, dtype=np.float64).reshape(-1, 3)
	gtcp = np.asarray([sim_ee_to_robot_ee(p, q, rv, sp)[0] for p, q in zip(gpos, gquat)],
					  dtype=np.float64).reshape(-1, 3)
	start = None
	if at.get("ee_pos0") is not None and at.get("ee_quat0") is not None:
		p0, q0 = sim_ee_to_robot_ee(np.asarray(at["ee_pos0"]), np.asarray(at["ee_quat0"]), rv, sp)
		start = {"pos": base_to_world(arm, p0), "quat": base_to_world_quat(arm, q0),
				 "qpos": np.asarray(at["init_qpos"], dtype=np.float64)}
	cursor = arr.get("cursor")
	return {
		"start": start,
		"gain_action": (np.asarray(arr["gain_action"], dtype=np.float64).reshape(-1, 2)
						if "gain_action" in arr else np.zeros((0, 2))),
		"kp": np.asarray(arr["kp"], dtype=np.float64).reshape(-1, 6) if "kp" in arr else np.zeros((0, 6)),
		"kd": np.asarray(arr["kd"], dtype=np.float64).reshape(-1, 6) if "kd" in arr else np.zeros((0, 6)),
		"trace": base_to_world(arm, tcp) if len(tcp) else np.zeros((0, 3)),
		"trace_base": tcp,
		"commanded": base_to_world(arm, gtcp) if len(gtcp) else np.zeros((0, 3)),
		"ee_quat": (np.asarray([base_to_world_quat(arm, q) for q in ee], dtype=np.float64)
					if len(ee) else np.zeros((0, 4))),
		"qpos": np.asarray(arr["qpos"], dtype=np.float64).reshape(-1, 7),
		"cursor_trace": (np.asarray(cursor, dtype=np.float64).reshape(-1)
						 if cursor is not None else np.zeros(0)),
	}


def _stats(v) -> dict:
	v = np.asarray(v, dtype=np.float64).ravel()
	if v.size == 0:
		return {"mean": None, "max": None, "rms": None, "final": None}
	return {"mean": float(v.mean()), "max": float(np.abs(v).max()),
			"rms": float(np.sqrt((v ** 2).mean())), "final": float(v[-1])}


def compute_reach_errors(d: dict, s: dict, ep: dict, sim: dict) -> dict:
	"""Sim-vs-real summary for one episode.

	Three families, because they answer different questions and only the third
	is drift-invariant:
	  absolute      -- how far apart the two arms are; includes any start offset
	  start_aligned -- the t=0 offset removed, so a homing residual is not read
	                   as a tracking difference
	  delta         -- step to step, invariant to accumulated drift; this is
	                   where a friction/stiction signature shows up

	Deliberately NOT sysid/_viz.py's `compute_trajectory_errors`: that one calls
	`resolve_world_frame_offset` unconditionally and would subtract a second
	offset from an already-base-frame sim record, silently shifting sim by the
	very convention residual reported below. It also binds its base pose to one
	profile at import.
	"""
	n = min(len(d["trace"]), len(s["trace"]))
	real, simt = d["trace"][:n], s["trace"][:n]
	err = simt - real
	sa, ra = sim["attrs"], ep["attrs"]
	out = {
		"n_steps_real": int(len(d["trace"])), "n_steps_sim": int(len(s["trace"])),
		"n_steps_compared": int(n),
		"goal_transport_max_m": sa.get("goal_transport_max_m"),
		"start_qpos_max_err_rad": sa.get("start_qpos_max_err_rad"),
		"position_error_m": _stats(np.linalg.norm(err, axis=1)) if n else _stats([]),
		"position_error_axis_m": {ax: _stats(err[:, i]) for i, ax in enumerate("xyz")} if n else {},
	}
	if n:
		out["start_aligned"] = {
			"position_error_m": _stats(np.linalg.norm(err - err[0], axis=1))}
		out["delta_error"] = {"position_delta_error_m": _stats(
			np.linalg.norm(np.diff(simt, axis=0) - np.diff(real, axis=0), axis=1))}
		if len(d["ee_quat"]) >= n and len(s["ee_quat"]) >= n:
			rr = Rotation.from_quat(d["ee_quat"][:n])
			ss = Rotation.from_quat(s["ee_quat"][:n])
			out["rotation_error_rad"] = _stats((ss * rr.inv()).magnitude())
		if len(d["gain_action"]) >= n and np.any(d["gain_action"][:n] != d["gain_action"][0]):
			# Error by gain tercile: the number that says whether the two arms
			# still agree once the gain action moves (flat = yes).
			joint_err = (np.sqrt(np.mean((s["qpos"][:n] - d["qpos"][:n]) ** 2, axis=1))
						 if len(d["qpos"]) >= n and len(s["qpos"]) >= n else None)
			out["by_gain"] = gain_terciles(d["gain_action"][:n], np.linalg.norm(err, axis=1), joint_err)
			out["sim_impedance_mode"] = sa.get("impedance_mode")
		cr = d["cursor_trace"][:n]
		cs = s["cursor_trace"][:n]
		if len(cr) == n and len(cs) == n:
			out["cursor_lag"] = _stats(cs - cr)
	if ep.get("curve") is None:
		out.pop("cursor_lag", None)
		return out
	out["task"] = {
		"real_success": bool(ra.get("success", False)),
		"sim_success": bool(sa.get("success", False)),
		"real_cursor": int(ra.get("cursor", 0)),
		"sim_cursor": int(sa.get("cursor", 0)),
		"curve_len": int(ra.get("curve_len", 0)),
	}
	return out


# ----------------------------------------------------------------- traces

def _sphere(centre, radius, color, name, opacity=0.14, n=20):
	u = np.linspace(0.0, 2.0 * np.pi, n)
	v = np.linspace(0.0, np.pi, n)
	return go.Surface(
		x=centre[0] + radius * np.outer(np.cos(u), np.sin(v)),
		y=centre[1] + radius * np.outer(np.sin(u), np.sin(v)),
		z=centre[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
		colorscale=[[0.0, color], [1.0, color]], showscale=False,
		opacity=opacity, name=name, showlegend=True, hoverinfo="skip",
	)


def _plane(x_range, y_range, z, color, name, opacity=0.2):
	return go.Surface(
		x=np.asarray(x_range, dtype=np.float64),
		y=np.asarray(y_range, dtype=np.float64),
		z=np.full((2, 2), float(z)),
		colorscale=[[0.0, color], [1.0, color]], showscale=False,
		opacity=opacity, name=name, showlegend=True, hoverinfo="skip",
	)


def _triad_traces(pose, colors=_TRIAD_COLORS, width=5, name="EE frame"):
	"""Three local-axis segments for a single [pos(3), quat_xyzw(4)] pose.

	Empty input yields empty traces, so a run recorded without `ee_quat` still
	animates -- the triad is simply not drawn.
	"""
	traces = []
	for axis, color in enumerate(colors):
		xs, ys, zs = [], [], []
		if len(pose):
			origin = np.asarray(pose[0][:3], dtype=np.float64)
			tip = origin + Rotation.from_quat(pose[0][3:7]).as_matrix()[:, axis] * _AXIS_LENGTH
			xs, ys, zs = [origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]]
		traces.append(go.Scatter3d(
			x=xs, y=ys, z=zs, mode="lines",
			line=dict(color=color, width=width),
			name=name if axis == 0 else f"{name} axis {axis}",
			legendgroup=name, showlegend=axis == 0, hoverinfo="skip",
		))
	return traces


def _reference_traces(d, keep, succ_thr):
	"""The static scene: curve, goal, success sphere, safety floor, keep-out."""
	wp, goal, vel = d["waypoints"], d["goal"], d["velocity_scales"]
	marker = dict(size=4, color="mediumvioletred")
	if len(vel) == len(wp):
		# Curve nodes carry a velocity scale; colouring by it shows where the
		# base policy is meant to be slowing into the goal.
		marker = dict(size=4, color=vel, colorscale="Viridis", cmin=0.0, cmax=1.0,
					  colorbar=dict(title="vel scale", len=0.25, y=0.86, thickness=10))
	traces = [
		go.Scatter3d(
			x=wp[:, 0], y=wp[:, 1], z=wp[:, 2], mode="lines+markers",
			line=dict(color="mediumvioletred", width=3), marker=marker,
			name="commanded curve",
			hovertemplate="wp %{pointNumber}<br>%{x:.3f} %{y:.3f} %{z:.3f}<extra></extra>",
		),
		go.Scatter3d(
			x=[goal[0]], y=[goal[1]], z=[goal[2]], mode="markers",
			marker=dict(size=6, color="mediumvioletred", symbol="x"),
			name=f"goal ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})",
			hovertemplate="goal<br>%{x:.3f} %{y:.3f} %{z:.3f}<extra></extra>",
		),
		_sphere(goal, succ_thr, "mediumvioletred", f"success radius {succ_thr*1000:.0f} mm"),
	]
	if keep is not None:
		traces.append(_sphere(keep[0], keep[1], "firebrick",
							  f"keep-out r={keep[1]:.3f} m", opacity=0.10))
	return traces


def _skeletons(arm, qpos):
	"""(T, 9, 3) world-frame skeleton points: base origin then the 8 FK frames."""
	out = []
	for q in qpos:
		chain = franka_fk_chain(q)
		out.append(base_to_world(arm, np.vstack([np.zeros((1, 3)), chain[:, :3, 3]])))
	return out


def _yrange(arr, pad=0.05):
	arr = np.asarray(arr, dtype=np.float64)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return (-1.0, 1.0)
	lo, hi = float(arr.min()), float(arr.max())
	span = max(hi - lo, 1e-3)
	return lo - span * pad, hi + span * pad


# ------------------------------------------------------------------ main

def build_reach_figure(ep: dict, arm: str | None = None, fps: float | None = None,
					   frame_stride: int = 1, title: str | None = None,
					   sim: dict | None = None) -> go.Figure:
	"""One episode as an animated figure, optionally overlaid with a sim replay.

	Layout
	------
	Left (66 %)  : 3D scene -- commanded curve coloured by velocity scale, goal
	               and its success sphere, the safety floor plane, the keep-out
	               sphere, and animated over them the arm skeleton, the measured
	               EE trail, the commanded OSC goal trail, the EE orientation
	               triad, and the waypoint the cursor is on. With `sim`, the same
	               skeleton/trail/triad again for the sim arm.
	Right (34 %) : stacked panels with a moving time cursor --
	               row 1  distance to the tracked waypoint and to the goal, with
	                      the advance/success thresholds; cursor index on the
	                      right axis
	               row 2  world-z clearance above the safety floor, measured and
	                      commanded -- below zero is where ActionSafetyScreen
	                      fires and silently rescales the goal
	               row 3  per-axis and L2 gap between the commanded goal and the
	                      pose actually reached
	               row 4  (gain records only) the normalised kp/kd action each
	                      step sent (-1..1, left axis) and the kp/kd the
	                      controller actually ran (log right axis); sim dashed
	               row 5  (sim only) sim-vs-real position error per axis + L2,
	                      with the geodesic rotation error in degrees on the
	                      right axis
	               last   per-joint q, real solid and sim dashed on a shared hue

	Time alignment is index-for-index, truncated to the shorter side. Nothing is
	warped: sim step i consumed real's dispatched goal i, so the pairing is
	definitional. A step-count difference means one side finished earlier, which
	is a finding, not something to resample away.

	A dry-run record has no steps; it renders as the static scene alone, which
	is what to look at before letting the arm move.
	"""
	A = ep["attrs"]
	arm = A.get("arm") or arm or "left"
	fps = float(fps or A.get("fps") or fc.control_fps())
	d = _episode_arrays(ep, arm)
	s = _sim_arrays(sim, arm) if sim is not None else None

	task = fc.section("reach")["task"]
	succ_thr = float(task["success_threshold_m"])
	adv_thr = float(task["advance_threshold_m"])
	floor = safety_z_floor_world(arm)
	keep = keep_out_sphere(arm)

	wp, goal = d["waypoints"], d["goal"]
	trace, cmd, quat, qpos = d["trace"], d["commanded"], d["ee_quat"], d["qpos"]
	cursor = d["cursor_trace"]
	gain = d["gain_action"] if np.any(d["gain_action"] != (d["gain_action"][0] if len(d["gain_action"]) else 0)) else np.zeros((0, 2))
	gain_kp = d["kp"] if len(d["kp"]) == len(gain) and len(gain) else None
	gain_kd = d["kd"] if len(d["kd"]) == len(gain) and len(gain) else None
	# Frame 0 is the shared start pose, before any command; step t is frame t+1.
	# Both sides get it or neither, so index alignment is never disturbed.
	drew_start = (d.get("start") is not None and len(trace)
			  and (s is None or s.get("start") is not None))
	if drew_start:
		st = d["start"]
		trace = np.vstack([st["pos"][None], trace])
		quat = np.vstack([st["quat"][None], quat]) if len(quat) else quat
		qpos = np.vstack([st["qpos"][None], qpos]) if len(qpos) else qpos
		cmd = np.vstack([st["pos"][None], cmd]) if len(cmd) else cmd   # no goal yet: hold
		cursor = np.concatenate([[0.0], cursor]) if len(cursor) else cursor
		gain = np.vstack([gain[:1], gain]) if len(gain) else gain      # no command yet: hold
		gain_kp = np.vstack([gain_kp[:1], gain_kp]) if gain_kp is not None else None
		gain_kd = np.vstack([gain_kd[:1], gain_kd]) if gain_kd is not None else None
		if s is not None:
			ss = s["start"]
			s = dict(s)
			s["trace"] = np.vstack([ss["pos"][None], s["trace"]])
			s["ee_quat"] = np.vstack([ss["quat"][None], s["ee_quat"]]) if len(s["ee_quat"]) else s["ee_quat"]
			s["qpos"] = np.vstack([ss["qpos"][None], s["qpos"]]) if len(s["qpos"]) else s["qpos"]
			s["commanded"] = np.vstack([ss["pos"][None], s["commanded"]]) if len(s["commanded"]) else s["commanded"]
			s["cursor_trace"] = (np.concatenate([[0.0], s["cursor_trace"]])
								 if len(s["cursor_trace"]) else s["cursor_trace"])
			for k in ("gain_action", "kp", "kd"):     # same hold as the real gain row
				s[k] = np.vstack([s[k][:1], s[k]]) if len(s[k]) else s[k]
	T_ = len(trace)
	has_curve = goal is not None
	curve_len = int(A.get("curve_len") or len(wp)) if has_curve else 0

	if len(cursor) != T_:
		cursor = np.zeros(T_)
	skels = _skeletons(arm, qpos) if len(qpos) == T_ and T_ else []
	poses = (np.hstack([trace, quat]) if len(quat) == T_ and T_
			 else np.zeros((0, 7), dtype=np.float64))

	# Sim arrays, built BEFORE the scene bounds -- the scene uses hard ranges
	# with autorange off, so anything left out of `pts` is silently clipped.
	sim_trace = s["trace"] if s else np.zeros((0, 3))
	sim_skels = _skeletons(arm, s["qpos"]) if s is not None and len(s["qpos"]) else []
	sim_poses = (np.hstack([sim_trace, s["ee_quat"]])
				 if s is not None and len(s["ee_quat"]) == len(sim_trace) and len(sim_trace)
				 else np.zeros((0, 7), dtype=np.float64))

	pts = (([wp, goal[None, :]] if has_curve else [])
		   + [a for a in (trace, cmd, sim_trace) if len(a)] + skels + sim_skels)
	if keep is not None:
		pts.append(np.array([keep[0] - keep[1], keep[0] + keep[1]]))
	all_xyz = np.concatenate(pts, axis=0)
	mn, mx = all_xyz.min(0), all_xyz.max(0)
	mn[2] = min(mn[2], floor)
	pad = float(max((mx - mn).max() * 0.05, 0.01))
	mn, mx = mn - pad, mx + pad
	extents = mx - mn
	ext_max = float(extents.max()) or 1.0

	scene_only = T_ == 0
	# Panel rows, top to bottom: distances+cursor (curve only), floor clearance,
	# commanded-vs-reached, gain action (gain records only), sim-vs-real (sim
	# only), per-joint q.
	has_gain = len(gain) == T_ and T_ > 0
	order = (([("dist", True)] if has_curve else []) + [("floor", False), ("track", False)]
			 + ([("gain", True)] if has_gain else [])
			 + ([("simdiff", True)] if s is not None else []) + [("q", False)])
	R = {name: i + 1 for i, (name, _) in enumerate(order)}
	SECONDARY = {i + 1 for i, (_, sec) in enumerate(order) if sec}
	ROWS = len(order)
	rows = tuple(range(1, ROWS + 1))
	if scene_only:
		fig = go.Figure()
	else:
		fig = make_subplots(
			rows=ROWS, cols=2,
			specs=([[{"type": "scene", "rowspan": ROWS},
					 {"type": "xy", "secondary_y": 1 in SECONDARY}]]
				   + [[None, {"type": "xy", "secondary_y": r in SECONDARY}]
					  for r in range(2, ROWS + 1)]),
			column_widths=[0.66, 0.34],
			row_heights=[1.0 / ROWS] * ROWS,
			horizontal_spacing=0.06, vertical_spacing=0.05,
		)

	def add(trace_obj, row=1, col=1, **kw):
		if scene_only:
			fig.add_trace(trace_obj)
		else:
			fig.add_trace(trace_obj, row=row, col=col, **kw)

	# ---- static scene ------------------------------------------------------
	if has_curve:
		for t in _reference_traces(d, keep, succ_thr):
			add(t)
	add(_plane((mn[0], mx[0]), (mn[1], mx[1]), floor, "slategray",
			   f"safety floor z={floor:.3f}"))
	if T_:
		add(go.Scatter3d(x=trace[:, 0], y=trace[:, 1], z=trace[:, 2], mode="lines",
						 line=dict(color="royalblue", width=2), opacity=0.2,
						 name="measured EE (full)"))
	if len(cmd):
		add(go.Scatter3d(x=cmd[:, 0], y=cmd[:, 1], z=cmd[:, 2], mode="lines",
						 line=dict(color="darkorange", width=2, dash="dot"), opacity=0.2,
						 name="commanded goal (full)"))
	if len(sim_trace):
		add(go.Scatter3d(x=sim_trace[:, 0], y=sim_trace[:, 1], z=sim_trace[:, 2],
						 mode="lines", line=dict(color=_C_SIM, width=2), opacity=0.2,
						 name="sim EE (full)"))

	ts = np.arange(T_, dtype=np.float64) / fps
	err = np.zeros((0, 3))
	err_norm = np.zeros(0)
	sim_err = np.zeros((0, 3))
	sim_err_norm = np.zeros(0)
	rot_err_deg = np.zeros(0)

	if not scene_only:
		# ---- distances + cursor (curve only) ------------------------------
		d_wp = d_goal = np.zeros(0)
		if has_curve:
			wp_idx = np.clip(cursor.astype(int), 0, max(curve_len - 1, 0))
			d_wp = np.linalg.norm(trace - wp[wp_idx], axis=1)
			d_goal = np.linalg.norm(trace - goal, axis=1)
			add(go.Scatter(x=ts, y=d_wp, mode="lines", line=dict(color="mediumvioletred", width=2),
						   name="|EE - waypoint|"), row=R["dist"], col=2, secondary_y=False)
			add(go.Scatter(x=ts, y=d_goal, mode="lines", line=dict(color="royalblue", width=2),
						   name="|EE - goal|"), row=R["dist"], col=2, secondary_y=False)
			if len(sim_trace):
				n = min(T_, len(sim_trace))
				add(go.Scatter(x=ts[:n], y=np.linalg.norm(sim_trace[:n] - goal, axis=1),
							   mode="lines", line=dict(color=_C_SIM, width=2, dash="dash"),
							   name="|sim EE - goal|"), row=R["dist"], col=2, secondary_y=False)
			for thr, label, color in ((adv_thr, "advance", "mediumvioletred"),
									  (succ_thr, "success", "royalblue")):
				add(go.Scatter(x=[ts[0], ts[-1]], y=[thr, thr], mode="lines",
							   line=dict(color=color, width=1, dash="dash"),
							   name=f"{label} {thr*1000:.0f} mm", hoverinfo="skip"),
					row=R["dist"], col=2, secondary_y=False)
			add(go.Scatter(x=ts, y=cursor, mode="lines", line=dict(color="dimgray", width=1.5),
						   name=f"cursor (of {curve_len})"), row=R["dist"], col=2, secondary_y=True)
			if s is not None and len(s["cursor_trace"]):
				n = min(T_, len(s["cursor_trace"]))
				add(go.Scatter(x=ts[:n], y=s["cursor_trace"][:n], mode="lines",
							   line=dict(color=_C_SIM, width=1.5, dash="dash"),
							   name="sim cursor"), row=R["dist"], col=2, secondary_y=True)

		# ---- row 2: clearance above the safety floor ----------------------
		add(go.Scatter(x=ts, y=trace[:, 2] - floor, mode="lines",
					   line=dict(color="royalblue", width=2), name="EE above floor"),
			row=R["floor"], col=2)
		if len(cmd):
			add(go.Scatter(x=ts[:len(cmd)], y=cmd[:, 2] - floor, mode="lines",
						   line=dict(color="darkorange", width=1.5, dash="dash"),
						   name="goal above floor"), row=R["floor"], col=2)
		if len(sim_trace):
			# Sim has no table. A sim dip below the real floor is exactly where
			# real's screen fired and sim's did not.
			add(go.Scatter(x=ts[:len(sim_trace)], y=sim_trace[:, 2] - floor, mode="lines",
						   line=dict(color=_C_SIM, width=1.5, dash="dash"),
						   name="sim EE above floor"), row=R["floor"], col=2)
		add(go.Scatter(x=[ts[0], ts[-1]], y=[0.0, 0.0], mode="lines",
					   line=dict(color="red", width=1, dash="dot"),
					   showlegend=False, hoverinfo="skip"), row=R["floor"], col=2)

		# ---- row 3: commanded goal vs pose reached ------------------------
		if len(cmd) == T_:
			err = trace - cmd
			err_norm = np.linalg.norm(err, axis=1)
			for i, (ax, c) in enumerate(zip("xyz", _XYZ_COLORS)):
				add(go.Scatter(x=ts, y=err[:, i], mode="lines",
							   line=dict(color=c, width=2), name=f"err_{ax}"), row=R["track"], col=2)
			add(go.Scatter(x=ts, y=err_norm, mode="lines",
						   line=dict(color="darkorchid", width=2), name="L2 err"),
				row=R["track"], col=2)
			add(go.Scatter(x=[ts[0], ts[-1]], y=[0.0, 0.0], mode="lines",
						   line=dict(color="black", width=1, dash="dot"),
						   showlegend=False, hoverinfo="skip"), row=R["track"], col=2)
		if s is not None and len(s["commanded"]) and len(sim_trace):
			n = min(len(s["commanded"]), len(sim_trace))
			add(go.Scatter(x=ts[:n], y=np.linalg.norm(sim_trace[:n] - s["commanded"][:n], axis=1),
						   mode="lines", line=dict(color=_C_SIM, width=1.5, dash="dash"),
						   name="sim L2 err"), row=R["track"], col=2)

		# ---- gain row (gain records only): the action, and the gain it became --
		if has_gain:
			remap = {"base_kp": float(A.get("osc_base_kp", 150.0)),
					 "base_ratio": float(A.get("osc_default_damping_ratio", 1.0)),
					 "exp_base": float(A.get("gain_exp_base", 10.0))}
			for tr, sec in gain_panel_traces(ts, gain, gain_kp, gain_kd, **remap):
				add(tr, row=R["gain"], col=2, secondary_y=sec)
			if s is not None and len(s["kp"]):
				# The sim's own resolved gains, dashed: equal to real's by
				# construction once the remap check passed, so a gap here is a bug.
				n_s = len(s["kp"])
				sim_ga = s["gain_action"] if len(s["gain_action"]) == n_s else gain[:n_s]
				for tr, sec in gain_panel_traces(ts[:n_s], sim_ga, s["kp"], s["kd"], label="sim",
												 dash="dash", show_action=False):
					add(tr, row=R["gain"], col=2, secondary_y=sec)

		# ---- sim row (sim only): sim vs real -------------------------------
		if s is not None:
			n = min(T_, len(sim_trace))
			if n:
				sim_err = sim_trace[:n] - trace[:n]
				sim_err_norm = np.linalg.norm(sim_err, axis=1)
				for i, (ax, c) in enumerate(zip("xyz", _XYZ_COLORS)):
					add(go.Scatter(x=ts[:n], y=sim_err[:, i], mode="lines",
								   line=dict(color=c, width=2), name=f"sim-real {ax}"),
						row=R["simdiff"], col=2, secondary_y=False)
				add(go.Scatter(x=ts[:n], y=sim_err_norm, mode="lines",
							   line=dict(color="darkorchid", width=3), name="sim-real L2"),
					row=R["simdiff"], col=2, secondary_y=False)
				add(go.Scatter(x=[ts[0], ts[n - 1]], y=[0.0, 0.0], mode="lines",
							   line=dict(color="black", width=1, dash="dot"),
							   showlegend=False, hoverinfo="skip"),
					row=R["simdiff"], col=2, secondary_y=False)
				if len(quat) >= n and len(s["ee_quat"]) >= n:
					rot_err_deg = np.degrees(
						(Rotation.from_quat(s["ee_quat"][:n])
						 * Rotation.from_quat(quat[:n]).inv()).magnitude())
					add(go.Scatter(x=ts[:n], y=rot_err_deg, mode="lines",
								   line=dict(color="goldenrod", width=2), name="rot err"),
						row=R["simdiff"], col=2, secondary_y=True)
				if n < max(T_, len(sim_trace)):
					# Truncation is visible rather than implied.
					add(go.Scatter(x=[ts[n - 1]] * 2,
								   y=[float(sim_err.min()), float(sim_err.max())],
								   mode="lines", line=dict(color="gray", width=1, dash="dash"),
								   name=f"compared to step {n}", hoverinfo="skip"),
						row=R["simdiff"], col=2, secondary_y=False)

		# ---- last row: per-joint q ----------------------------------------
		for j in range(qpos.shape[1] if len(qpos) else 0):
			add(go.Scatter(x=ts[:len(qpos)], y=qpos[:, j], mode="lines",
						   line=dict(color=_JOINT_COLORS[j], width=1.5),
						   name=f"q{j+1}", legendgroup=f"q{j+1}"), row=ROWS, col=2)
		if s is not None and len(s["qpos"]):
			n = min(T_, len(s["qpos"]))
			for j in range(s["qpos"].shape[1]):
				add(go.Scatter(x=ts[:n], y=s["qpos"][:n, j], mode="lines",
							   line=dict(color=_JOINT_COLORS[j], width=1.5, dash="dash"),
							   name=f"sim q{j+1}", legendgroup=f"q{j+1}",
							   showlegend=False), row=ROWS, col=2)

	# ---- animated traces ---------------------------------------------------
	indices = list(range(0, T_, max(1, frame_stride)))
	anim_idxs: list[int] = []

	def add_anim(trace_obj, row=1, col=1, **kw):
		add(trace_obj, row=row, col=col, **kw)
		anim_idxs.append(len(fig.data) - 1)

	def skeleton_trace(i, arr, color, width, size, name):
		p = arr[min(i, len(arr) - 1)] if len(arr) else np.zeros((0, 3))
		return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers",
							line=dict(color=color, width=width),
							marker=dict(size=size, color=color), name=name)

	def trail_trace(arr, i, color, dash, name, width=4):
		p = arr[:min(i, len(arr) - 1) + 1] if len(arr) else np.zeros((0, 3))
		return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
							line=dict(color=color, width=width, dash=dash), name=name)

	def waypoint_marker(i):
		if not has_curve or not curve_len:
			return go.Scatter3d(x=[], y=[], z=[], mode="markers", name="tracked waypoint")
		w = wp[min(int(cursor[i]), curve_len - 1)]
		return go.Scatter3d(x=[w[0]], y=[w[1]], z=[w[2]], mode="markers",
							marker=dict(size=7, color="gold", symbol="diamond"),
							name="tracked waypoint")

	def cursor_line(t_val, lo, hi):
		return go.Scatter(x=[t_val, t_val], y=[lo, hi], mode="lines",
						  line=dict(color="black", width=1, dash="dot"),
						  showlegend=False, hoverinfo="skip")

	def anim_data(i):
		"""Every animated trace for frame `i`, in one place.

		Registration and each go.Frame both come from this list, so the two can
		no longer drift -- a frame mapping onto the wrong trace is the failure
		mode this function exists to remove.
		"""
		out = [skeleton_trace(i, skels, "dimgray", 6, 4, "arm"),
			   trail_trace(trace, i, "royalblue", "solid", "measured EE"),
			   trail_trace(cmd, i, "darkorange", "dot", "commanded goal")]
		out.extend(_triad_traces(poses[i:i + 1]))
		out.append(waypoint_marker(i))
		if s is not None:
			out.append(skeleton_trace(i, sim_skels, "#9aa0a6", 4, 3, "sim arm"))
			out.append(trail_trace(sim_trace, i, _C_SIM, "solid", "sim EE"))
			j = min(i, len(sim_poses) - 1) if len(sim_poses) else 0
			out.extend(_triad_traces(sim_poses[j:j + 1], colors=_SIM_TRIAD_COLORS,
									 width=4, name="sim EE frame"))
		return out

	if indices:
		for t in anim_data(indices[0]):
			add_anim(t)

		ranges = {
			"dist": _yrange(np.concatenate([d_wp, d_goal, [0.0]])),
			"floor": _yrange(np.concatenate([trace[:, 2] - floor,
											 (cmd[:, 2] - floor) if len(cmd) else [],
											 (sim_trace[:, 2] - floor) if len(sim_trace) else [],
											 [0.0]])),
			"track": _yrange(np.concatenate([err.ravel(), err_norm, [0.0]])),
			"gain": _yrange(np.concatenate([gain.ravel(), [0.0]])) if has_gain else (-1.0, 1.0),
			"simdiff": _yrange(np.concatenate([sim_err.ravel(), sim_err_norm, [0.0]])),
			"q": _yrange(qpos) if len(qpos) else (-1.0, 1.0),
		}
		yr = [ranges[name] for name, _ in order]

		for r in rows:
			kw = dict(secondary_y=False) if r in SECONDARY else {}
			add_anim(cursor_line(0.0, *yr[r - 1]), row=r, col=2, **kw)

		fig.frames = [
			go.Frame(data=anim_data(i) + [cursor_line(float(ts[i]), *yr[r - 1]) for r in rows],
					 traces=anim_idxs, name=str(fi))
			for fi, i in enumerate(indices)
		]

	# ---- layout ------------------------------------------------------------
	what = "real reach" if has_curve else "trajectory"
	head = title or f"{what} — {ep['name']} — arm {arm}"
	if s is not None:
		head += " — sim vs real"
	if A.get("dry_run"):
		sub = "dry run — curve sampled, no action sent"
	elif T_ and not has_curve:
		src = A.get("source", "arm")
		sub = (f"{A.get('steps', T_)} steps | real = "
			   f"{'the dataset recording (older controller)' if src == 'dataset' else 'arm re-run'} | "
			   f"min floor clearance {(trace[:, 2] - floor).min()*1000:+.0f} mm")
	elif T_:
		sub = (f"{'SUCCESS' if A.get('success') else 'timeout'} in {A.get('steps', T_)} steps | "
			   f"cursor {int(A.get('cursor', cursor[-1]))}/{curve_len} | "
			   f"final |EE-goal| {np.linalg.norm(trace[-1] - goal)*1000:.0f} mm | "
			   f"min floor clearance {(trace[:, 2] - floor).min()*1000:+.0f} mm")
		if len(cmd) == T_:
			sub += f" | tracking err mean {err_norm.mean()*1000:.1f} max {err_norm.max()*1000:.1f} mm"
	else:
		sub = "no steps recorded"
	if s is not None and len(sim_err_norm):
		n = len(sim_err_norm)
		how = ("same policy re-run closed loop in sim" if sim["attrs"].get("mode") == "policy_rerun"
			   else "sim replays real's dispatched goals")
		sub += (f"<br>{how}; sim vs real over {n} steps: mean {sim_err_norm.mean()*1000:.1f} "
				f"max {sim_err_norm.max()*1000:.1f} mm")
		if len(rot_err_deg):
			sub += f" | rot mean {rot_err_deg.mean():.2f} max {rot_err_deg.max():.2f} deg"
		if has_curve:
			sub += (f" | sim {'SUCCESS' if sim['attrs'].get('success') else 'timeout'} "
					f"cursor {sim['attrs'].get('cursor', 0)}/{curve_len}")
		transport = sim["attrs"].get("goal_transport_max_m")
		if transport is not None and transport > 1e-9:
			# Sim modified the command it was handed; every error above is
			# meaningless until that is fixed.
			sub = (f"<span style='color:red'>REPLAY NOT FAITHFUL — goal transport "
				   f"{transport:.2e} m</span><br>" + sub)
	if d.get("timing_shifted"):
		sub += ("<br>real trace was read right after each send (one step stale); "
				"shifted one step to align with sim")
	fig.update_layout(
		title=dict(text=f"{head}<br><sup>{sub}</sup>", x=0.5, xanchor="center"),
		showlegend=True,
		legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.7)", font=dict(size=9)),
		margin=dict(l=0, r=10, t=80, b=60),
		scene=dict(
			xaxis=dict(range=[mn[0], mx[0]], autorange=False, title="x (m)"),
			yaxis=dict(range=[mn[1], mx[1]], autorange=False, title="y (m)"),
			zaxis=dict(range=[mn[2], mx[2]], autorange=False, title="z (m)"),
			aspectmode="manual",
			aspectratio=dict(x=float(extents[0]/ext_max), y=float(extents[1]/ext_max),
							 z=float(extents[2]/ext_max)),
		),
	)
	if indices:
		frame_ms = int(round(1000.0 / max(fps, 1.0)))
		fig.update_layout(
			updatemenus=[dict(
				type="buttons", showactive=False,
				x=0.0, y=0.0, xanchor="left", yanchor="top",
				buttons=[
					dict(label="Play", method="animate", args=[None, dict(
						frame=dict(duration=frame_ms, redraw=True),
						fromcurrent=True, transition=dict(duration=0))]),
					dict(label="Pause", method="animate", args=[[None], dict(
						frame=dict(duration=0, redraw=False), mode="immediate")]),
				],
			)],
			sliders=[dict(
				active=0, currentvalue=dict(prefix="step: "), pad=dict(t=40),
				steps=[dict(method="animate", args=[[str(fi)], dict(
					mode="immediate", frame=dict(duration=0, redraw=True),
					transition=dict(duration=0))], label=str(i))
					for fi, i in enumerate(indices)],
			)],
		)
	if not scene_only:
		if has_curve:
			fig.update_yaxes(title_text="distance (m)", row=R["dist"], col=2, secondary_y=False)
			fig.update_yaxes(title_text="cursor", row=R["dist"], col=2, secondary_y=True,
							 range=[0, max(curve_len, 1)], showgrid=False)
		fig.update_yaxes(title_text="above floor (m)", row=R["floor"], col=2)
		fig.update_yaxes(title_text="goal - reached (m)", row=R["track"], col=2)
		if has_gain:
			fig.update_yaxes(title_text=GAIN_AXIS_LEFT, range=[-1.05, 1.05], row=R["gain"], col=2,
							 secondary_y=False)
			fig.update_yaxes(title_text=GAIN_AXIS_RIGHT, type="log", row=R["gain"], col=2,
							 secondary_y=True, showgrid=False)
		if s is not None:
			fig.update_yaxes(title_text="sim - real (m)", row=R["simdiff"], col=2, secondary_y=False)
			fig.update_yaxes(title_text="rot err (deg)", row=R["simdiff"], col=2, secondary_y=True,
							 showgrid=False)
		fig.update_yaxes(title_text="q (rad)", row=ROWS, col=2)
		for r in rows:
			fig.update_xaxes(title_text="time (s)", row=r, col=2,
							 range=[0.0, float(ts[-1]) + 0.1])
	return fig


def save_reach_html(ep: dict, path: str | Path, arm: str | None = None,
					fps: float | None = None, frame_stride: int = 1,
					title: str | None = None, sim: dict | None = None) -> None:
	"""Render one episode to a self-contained HTML."""
	fig = build_reach_figure(ep, arm=arm, fps=fps, frame_stride=frame_stride,
							 title=title, sim=sim)
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.write_html(str(path), include_plotlyjs="cdn")



def save_run_html(run_dir: str | Path, episodes: list[dict], arm: str | None = None,
				  frame_stride: int = 1, sim_dir: str | Path | None = None,
				  errors_path: str | Path | None = None) -> list[Path]:
	"""One HTML per episode in a run directory. Returns the paths written.

	An episode that has a matching one in `sim_dir`'s sim_replay.hdf5 (default:
	the run directory) is rendered as a comparison into `compare_<name>.html`,
	so a re-render with sim never clobbers the hardware-only figure.
	"""
	out, summaries = [], []
	sims = load_sim_run(sim_dir or run_dir)
	for ep in episodes:
		sim = sims.get(ep["name"])
		p = Path(run_dir) / (f"compare_{ep['name']}.html" if sim else f"episode_{ep['name']}.html")
		save_reach_html(ep, p, arm=arm, frame_stride=frame_stride, sim=sim)
		out.append(p)
		if sim is not None:
			a = ep["attrs"].get("arm") or arm or "left"
			summaries.append({"episode": ep["name"],
							  **compute_reach_errors(_episode_arrays(ep, a),
													 _sim_arrays(sim, a), ep, sim)})
			if "by_gain" in summaries[-1]:
				print(f"{ep['name']} error by gain tercile (sim {summaries[-1]['sim_impedance_mode']}):")
				print(format_gain_table(summaries[-1]["by_gain"]))
	if summaries and errors_path:
		Path(errors_path).write_text(json.dumps(
			{"schema": "real_reach_errors/1", "alignment": "index",
			 "episodes": summaries}, indent=1))
		print(f"errors -> {errors_path}")
	return out


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("run", help="run directory under ~/franka_data/real_reach, or its episodes.hdf5")
	ap.add_argument("--episode", type=int, default=None, help="render only this episode index")
	ap.add_argument("--out", default=None, help="output HTML path (single episode only)")
	ap.add_argument("--arm", default=None, choices=("left", "right"),
					help="arm the run drove; only needed for episodes without an arm attr")
	ap.add_argument("--stride", type=int, default=1, help="animate every Nth step")
	ap.add_argument("--sim", default=None,
					help="a sim_replay.hdf5, or the directory holding one "
						 "(default: the run directory itself)")
	ap.add_argument("--no-errors", action="store_true",
					help="skip writing errors.json alongside a comparison")
	args = ap.parse_args()

	run_dir, episodes = load_run(args.run)
	if args.episode is not None:
		episodes = [e for e in episodes if int(e["attrs"].get("episode", -1)) == args.episode]
		if not episodes:
			print(f"no episode {args.episode} in {run_dir / episode_hdf5.REAL_FILE}")
			return 1
	if args.out and len(episodes) != 1:
		print("--out takes a single episode; pass --episode too")
		return 1

	if args.out:
		ep = episodes[0]
		sim = load_sim_run(args.sim or run_dir).get(ep["name"])
		save_reach_html(ep, args.out, arm=args.arm, frame_stride=args.stride, sim=sim)
		if sim is not None and not args.no_errors:
			a = ep["attrs"].get("arm") or args.arm or "left"
			summary = compute_reach_errors(_episode_arrays(ep, a), _sim_arrays(sim, a), ep, sim)
			print(json.dumps(summary, indent=1))
			if "by_gain" in summary:
				print(format_gain_table(summary["by_gain"]))
		print(args.out)
		return 0
	for p in save_run_html(run_dir, episodes, arm=args.arm, frame_stride=args.stride,
						   sim_dir=args.sim,
						   errors_path=None if args.no_errors else run_dir / "errors.json"):
		print(p)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
