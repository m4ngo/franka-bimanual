"""Workspace bounds and episode sampling for the real reach task.

The sampling maths is the sim's, imported unchanged from
multi-fast/utils/envs/reach_sampling.py. This module supplies the two things
the sim cannot: bounds that describe THIS rig, and the frame conversion between
the arm's base and the world the safety layer reasons in.

Frames. Sampling happens in the arm's BASE frame, because the sim's workspace
box is meaningful relative to the arm and the right arm here is yawed ~136
degrees. `robot_base_in_world` maps base -> world and is never inverted, per
the repo convention. Positions are in the robot's own EE convention -- what
`send_action` commands and `get_observation` reports. Mapping to the sim's
grip-site convention is `fc.sim_ee_convention`, applied by the caller when
comparing against sim, not here.

Quaternions from the sampler are xyzw (robosuite); config/*.yaml is wxyz.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import franka_config as fc

from .ee_kinematics import eef_poses_from_qpos

_MULTI_FAST = Path(__file__).resolve().parents[2] / "multi-fast"
_SAMPLING = _MULTI_FAST / "utils" / "envs" / "reach_sampling.py"


def _load_sampling():
	"""Import reach_sampling.py by path.

	Not a package import: `utils.envs.__init__` pulls in robosuite, which is not
	installed on the workstation (it drives hardware; sim runs on the cluster).
	reach_sampling.py is robosuite-free precisely so this works.
	"""
	if not _SAMPLING.is_file():
		raise FileNotFoundError(
			f"{_SAMPLING} not found -- the multi-fast submodule is missing or "
			f"predates the reach_sampling split"
		)
	spec = importlib.util.spec_from_file_location("reach_sampling", _SAMPLING)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


S = _load_sampling()


def safety_z_floor_world(arm: str) -> float:
	"""Lowest world z the commanded EE may occupy without ActionSafetyScreen
	firing, for ANY orientation.

	The screen compares the collision sphere's lowest point against
	`worktable.height_m + distance_min_m`. The sphere centre is a TOOL-frame
	offset that rotates with the gripper, so worst case it hangs
	`|center_tool| + radius` below the EE. Bounding that worst case makes the
	floor orientation-independent -- deliberately conservative, because a
	sampler that merely stays legal would still let the screen rescale a goal
	near a tilted pose, and a rescaled goal means the executed trace is no
	longer the commanded curve.
	"""
	return (fc.worktable_height_m()
			+ float(fc.control("worktable_brake.distance_min_m"))
			+ _ee_sphere_reach(arm))


def _ee_sphere_reach(arm: str) -> float:
	"""How far the collision sphere reaches from the EE origin, any orientation."""
	sphere = fc.ee_sphere(arm)
	return float(np.linalg.norm(sphere.center_tool_m)) + float(sphere.radius_m)


def _home_key(arm: str) -> str:
	"""home_poses key for an arm, via the bimanual profile's authoritative map."""
	arms = fc.section("rig")["profiles"]["bimanual_franka"]["arms"]
	for key, name in arms.items():
		if name == arm:
			return key
	raise KeyError(f"arm {arm!r} is not in rig.yaml's bimanual_franka arm map {arms}")


def keep_out_sphere(arm: str) -> tuple[np.ndarray, float] | None:
	"""World-frame (centre, radius) the curve may not enter, or None if disabled.

	ActionSafetyScreen implements no bimanual arm-repel, so when one arm runs
	beside a parked one the sampler is the only separation between them. The
	derived radius sums both arms' collision-sphere reach, which is the distance
	at which the two spheres touch for any pair of orientations; margin_m is on
	top of that.

	The default centre is the other arm's HOME EE, which is an assumption about
	where that arm is standing, not a measurement -- see config/reach.yaml.
	"""
	cfg = fc.section("reach")["workspace"].get("keep_out") or {}
	other = cfg.get("arm")
	if other is None:
		return None
	if other == "other":
		names = list(fc.section("rig")["profiles"]["bimanual_franka"]["arms"].values())
		peers = [n for n in names if n != arm]
		if len(peers) != 1:
			raise ValueError(
				f"reach.workspace.keep_out.arm is 'other' but the bimanual profile "
				f"lists {names} — cannot resolve the peer of {arm!r}"
			)
		other = peers[0]
	if other == arm:
		raise ValueError(
			f"reach.workspace.keep_out.arm is {other!r}, which is the arm being "
			f"sampled -- it cannot be its own keep-out"
		)

	centre = cfg.get("center_world_m")
	if centre is None:
		q = np.asarray(fc.home_q(key=_home_key(other)), dtype=np.float64)[None, :]
		centre = base_to_world(other, eef_poses_from_qpos(q)[0][0])
	centre = np.asarray(centre, dtype=np.float64)

	radius = cfg.get("radius_m")
	if radius is None:
		radius = _ee_sphere_reach(arm) + _ee_sphere_reach(other)
	return centre, float(radius) + float(cfg.get("margin_m", 0.0))


def _base_pose(arm: str):
	pose = fc.robot_base_in_world(arm)
	# A base-frame z floor only maps to a world-frame one when base z IS world
	# z. True for a yawed mount, false for a tilted one -- refuse rather than
	# silently sampling under the table.
	if not np.isclose(pose.rotation[2, 2], 1.0, atol=1e-6):
		raise ValueError(
			f"arm {arm!r} base is tilted (R[2,2]={pose.rotation[2, 2]:.6f}); "
			f"the base-frame z floor in config/reach.yaml assumes a yaw-only mount"
		)
	return pose


def workspace_bounds_base(arm: str) -> np.ndarray:
	"""(2, 3) xyz min/max in base frame, with the safety floor applied.

	The configured floor and the derived one are combined with MAX: the derived
	value is a safety bound, so a config entry may only shrink the volume.
	"""
	cfg = fc.section("reach")["workspace"]
	bounds = np.asarray(cfg["bounds_base_m"], dtype=np.float64).copy()
	base_z = float(_base_pose(arm).translation[2])

	derived = safety_z_floor_world(arm) - base_z + float(cfg["floor_margin_m"])
	override = cfg.get("z_floor_base_m", None)
	floor = derived if override is None else max(derived, float(override))

	bounds[0, 2] = max(bounds[0, 2], floor)
	if bounds[0, 2] >= bounds[1, 2]:
		raise ValueError(
			f"arm {arm!r}: safety floor {bounds[0, 2]:.3f} is at or above the "
			f"box ceiling {bounds[1, 2]:.3f} (base frame) -- nothing to sample"
		)
	return bounds


def curve_is_safe(points: np.ndarray, bounds: np.ndarray, radius: float,
                  arm: str | None = None,
                  keep_out: tuple[np.ndarray, float] | None = None) -> bool:
	"""Every point inside the base-frame box and the fast-reach sphere, and
	outside the other arm's keep-out."""
	pts = np.asarray(points, dtype=np.float64)
	if not (np.all(pts >= bounds[0]) and np.all(pts <= bounds[1])
			and np.all(np.linalg.norm(pts, axis=1) <= radius)):
		return False
	if keep_out is None:
		return True
	if arm is None:
		raise ValueError("curve_is_safe needs `arm` to map a keep-out into world frame")
	centre, keep_r = keep_out
	# Compared in world: the keep-out is a world object, and base_to_world is
	# the sanctioned direction (robot_base_in_world is never inverted).
	return bool(np.all(np.linalg.norm(base_to_world(arm, pts) - centre, axis=1) >= keep_r))


def base_to_world(arm: str, points: np.ndarray) -> np.ndarray:
	"""(..., 3) base-frame -> world, via p_world = R @ p_base + t."""
	pose = _base_pose(arm)
	pts = np.asarray(points, dtype=np.float64)
	return pts @ np.asarray(pose.rotation).T + np.asarray(pose.translation)


def base_to_world_quat(arm: str, quats_xyzw: np.ndarray) -> np.ndarray:
	"""(..., 4) base-frame xyzw orientation -> world, via R_world = R_base_in_world @ R."""
	pose = _base_pose(arm)
	rot = Rotation.from_matrix(np.asarray(pose.rotation))
	return (rot * Rotation.from_quat(np.asarray(quats_xyzw, dtype=np.float64))).as_quat()


def sim_ee_to_robot_ee(pos: np.ndarray, quat_xyzw: np.ndarray,
					   site_rotvec: np.ndarray, site_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""robosuite grip-site pose -> the FR3's O_T_EE. Base frame, xyzw.

	`site_rotvec` / `site_pos` are the site expressed IN O_T_EE (tool frame), as
	the sim replay measured them at qpos0 and wrote into its record -- so this is
	the exact inverse of the map that produced the sim goals, never a constant
	remembered separately. On the shipped Panda it is a -90 deg tool-z rotation
	and a 6.9 mm tool-z shift; `fc.sim_ee_convention` is NOT it (that describes
	the hand body, which sim-trained policies observe but the OSC does not drive).
	"""
	c_rot = Rotation.from_rotvec(np.asarray(site_rotvec, dtype=np.float64))
	r_otee = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)) * c_rot.inv()
	tcp = np.asarray(pos, dtype=np.float64) - r_otee.apply(np.asarray(site_pos, dtype=np.float64))
	return tcp, r_otee.as_quat()


def sample_episode(arm: str, start_base: np.ndarray, rng: np.random.Generator) -> dict:
	"""One episode's task: goal + dense waypoint curve, in BASE frame.

	`start_base` is the arm's current EE position in base frame. Mirrors
	Reach._sample_episode_curves, both branches and the velocity profile, so a
	real episode is drawn the way a sim one is -- with one addition: the whole
	curve is checked against this rig's workspace and redrawn if it leaves it.

	The samplers default `base_pos` to the SIM's world-frame Franka mount, which
	is where their reachability check measures from -- in base frame that origin
	is (0, 0, 0), so it must be passed explicitly or every candidate is rejected
	against a point 1.1 m away.
	"""
	cfg = fc.section("reach")
	curve, orient, ws = cfg["curve"], cfg["orientation"], cfg["workspace"]
	n_segments = int(curve["n_segments"])
	node_v = [S._resolve_velocity(v) for v in curve["node_velocities"]]
	peak_v = S._resolve_velocity(curve["segment_peak_velocity"])
	if len(node_v) != n_segments + 1:
		raise ValueError(
			f"reach.curve.node_velocities has {len(node_v)} entries; n_segments="
			f"{n_segments} needs one per node, {n_segments + 1}"
		)

	bounds = workspace_bounds_base(arm)
	start = np.asarray(start_base, dtype=np.float64)
	origin = np.zeros(3)                       # the base IS the origin in base frame
	radius = float(ws["reachable_radius_m"])
	keep_out = keep_out_sphere(arm)

	# The curve begins AT the start, so an out-of-box start makes every redraw
	# fail. Say so here rather than after burning max_sample_attempts on a
	# condition no resample can fix -- and refuse rather than clamping: a start
	# below the floor means the arm is somewhere homing should have fixed, which
	# is an operator problem, not a sampling one.
	if not curve_is_safe(start[None, :], bounds, radius):
		raise ValueError(
			f"arm {arm!r}: EE start {start} is outside the sampling workspace "
			f"(box {bounds[0]}..{bounds[1]}, radius {radius}). Home the arm "
			f"first; if this is the homed pose, it sits too close to a bound."
		)
	# Separate message: if the OTHER arm is what the start conflicts with,
	# homing this one is the wrong advice.
	if keep_out is not None:
		gap = float(np.linalg.norm(base_to_world(arm, start) - keep_out[0]))
		if gap < keep_out[1]:
			raise ValueError(
				f"arm {arm!r}: EE start is {gap:.3f} m from the keep-out centre, "
				f"inside its {keep_out[1]:.3f} m radius. Park the other arm clear "
				f"or re-measure reach.workspace.keep_out."
			)
	n_waypoints = int(curve["n_waypoints"])
	attempts = int(curve["max_sample_attempts"])
	control_offset = float(curve["control_offset"])

	# Bounding the endpoints does NOT bound the curve: sample_dense_curve offsets
	# its Bezier control point by control_offset, so waypoints bulge outside the
	# box between start and goal -- measured up to 3.7 cm below the floor. In sim
	# that is harmless (no table, and reject_infeasible catches the rest); here it
	# would put the commanded path under the worktable. So the whole curve is
	# checked, and a bulging one is redrawn rather than clipped -- clipping would
	# kink the curve the base policy is tracking.
	for _ in range(attempts):
		if n_segments > 1:
			# Dense count scales with the segment count so the per-segment
			# spacing the cursor was tuned for stays put.
			waypoints, goal, velocities = S.sample_multi_segment_curve(
				start, bounds,
				n_segments=n_segments,
				n_points_total=min(n_waypoints * n_segments, S.MAX_WAYPOINTS),
				control_offset=control_offset,
				reachable_radius=radius, base_pos=origin, max_attempts=attempts,
				node_velocities=node_v, segment_peak_velocity=peak_v, rng=rng,
			)
		else:
			goal = S.sample_goal(
				start, bounds,
				min_dist=float(curve["min_goal_dist_m"]),
				max_attempts=attempts,
				reachable_radius=radius, base_pos=origin, rng=rng,
			)
			waypoints = S.sample_dense_curve(
				start, goal, n_points=n_waypoints, control_offset=control_offset, rng=rng,
			)
			velocities = np.asarray(
				S._build_dense_velocities(node_v, n_waypoints, peak=peak_v), dtype=np.float32
			)[:n_waypoints]
			if velocities.shape[0] < n_waypoints:
				velocities = np.concatenate([velocities, np.full(
					n_waypoints - velocities.shape[0], node_v[-1], dtype=np.float32)])
		if curve_is_safe(waypoints, bounds, radius, arm=arm, keep_out=keep_out):
			break
	else:
		raise RuntimeError(
			f"arm {arm!r}: no curve from {start} stayed inside the workspace in "
			f"{attempts} attempts. The start is probably too near a bound -- the "
			f"Bezier needs ~control_offset of room on every side."
		)

	quats = None
	if float(orient["delta_max_deg"]) > 0.0:
		quats = S.sample_orientation_curve(
			np.array([0.0, 0.0, 0.0, 1.0]),      # xyzw identity; caller re-anchors
			len(waypoints),
			np.deg2rad(float(orient["delta_max_deg"])),
			n_segments=n_segments, rng=rng,
		)

	return {
		"goal": goal,
		"waypoints": waypoints,
		"waypoint_quats": quats,
		"velocity_scales": velocities,
		"bounds_base": bounds,
	}
