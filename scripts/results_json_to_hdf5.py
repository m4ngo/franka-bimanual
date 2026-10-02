#!/usr/bin/env python3
"""Translate a legacy rollout run (results.json + sim_episode_*.json) into
episodes.hdf5 + sim_replay.hdf5, the layout everything now reads.

  python scripts/results_json_to_hdf5.py ~/franka_data/real_reach/20260913_094502
  python scripts/results_json_to_hdf5.py ~/franka_data/real_traj/*/* --remove-json

The JSON files are left in place unless --remove-json is given, and it only
removes them after the written files validate. See EPISODE_HDF5.md.

results.json stored the measured trace in WORLD frame; the episode files are
base frame, so this is the one place `robot_base_in_world` is inverted -- the
exact inverse of the map the old recorder applied, for legacy data only.
Nothing that reads episode files does this.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "multi-fast"))

import franka_config as fc  # noqa: E402
from lerobot_robot_bimanual_franka.osc_torque_controller import resolve_gains  # noqa: E402
from lerobot_robot_bimanual_franka.reach_record import (  # noqa: E402
	POLICY_ACTION_LEROBOT, POLICY_ACTION_REACH, episode_name,
)
from utils.sysid import episode_hdf5  # noqa: E402


def _world_to_base(arm: str, pts: np.ndarray) -> np.ndarray:
	pose = fc.robot_base_in_world(arm)
	R = np.asarray(pose.rotation)
	return (np.asarray(pts, dtype=np.float64) - np.asarray(pose.translation)) @ R


def _world_to_base_quat(arm: str, quats: np.ndarray) -> np.ndarray:
	pose = fc.robot_base_in_world(arm)
	return (Rotation.from_matrix(np.asarray(pose.rotation)).inv()
			* Rotation.from_quat(np.asarray(quats, dtype=np.float64))).as_quat()


def _arr(v, shape):
	return (np.zeros((0,) + shape[1:]) if not v
			else np.asarray(v, dtype=np.float64).reshape(shape))


def _opt(v):
	"""A list -> float array, None (or empty) -> None, so the attr is skipped."""
	return None if v is None or len(v) == 0 else np.asarray(v, dtype=np.float64)


def real_episode(rec: dict, arm_default: str | None) -> tuple | None:
	"""One results.json record -> (name, arrays, attrs, curve), or None when
	the record predates the `replay` block and has no dispatched goals."""
	rp = rec.get("replay")
	if rp is None:
		return None
	arm = rec.get("arm") or rp.get("arm") or arm_default or "left"
	fps = float(rec.get("fps") or rp.get("fps") or fc.control_fps())
	name = episode_name(rec.get("episode", 0))
	curve = None
	if rp.get("waypoints") is not None:
		curve = {"goal": np.asarray(rp["goal"], dtype=np.float64),
				 "waypoints": np.asarray(rp["waypoints"], dtype=np.float64).reshape(-1, 3),
				 "velocity_scales": np.asarray(rp["velocity_scales"], dtype=np.float64)}
	remap = rp.get("gain_remap") or {}
	attrs = {
		"episode": int(rec.get("episode", 0)), "arm": arm, "seed": rec.get("seed"),
		"source": rec.get("real_source", "arm"), "fps": fps,
		"frame": "base", "quat_order": "xyzw", "ee_convention": "O_T_EE",
		"obs_timing": rec.get("obs_timing"),
		"control_mode": rp.get("control_mode"),
		"action_format": "absolute_pose_quat", "action_space": "EE_POS",
		"init_qpos": np.asarray(rp["qpos0"], dtype=np.float64),
		"ee_quat0": _opt(rp.get("ee_quat0")), "ee_pos0": _opt(rp.get("ee_pos0")),
		"sim_ee_convention_rotvec": _opt((rp.get("sim_ee_convention") or {}).get("rotvec_rad")),
		"sim_ee_convention_pos": _opt((rp.get("sim_ee_convention") or {}).get("pos_tool_m")),
		"osc_base_kp": remap.get("osc_base_kp"),
		"osc_default_damping_ratio": remap.get("osc_default_damping_ratio"),
		"gain_exp_base": remap.get("gain_exp_base"),
		"kp_limits": remap.get("kp_limits"), "damping_ratio_limits": remap.get("damping_ratio_limits"),
		"tuning_gain_scales": remap.get("tuning_gain_scales"),
		"translated_from": "results.json",
	}
	if curve is not None:
		attrs["curve_len"] = int(rec.get("curve_len", len(curve["waypoints"])))
	if rec.get("dry_run"):
		empty = {k: np.zeros((0,) + s) for k, s in (
			("action", (7,)), ("eef_goal_pos", (3,)), ("eef_goal_quat", (4,)),
			("eef_pos", (3,)), ("eef_quat", (4,)), ("qpos", (7,)), ("qvel", (7,)))}
		return name, empty, {**attrs, "num_samples": 0, "steps": 0, "dry_run": True}, curve

	trace_w = _arr(rec.get("trace"), (-1, 3))
	quat_w = _arr(rec.get("ee_quat"), (-1, 4))
	n = len(trace_w)
	goal_pos = _arr(rp.get("osc_goal_pos"), (-1, 3))[:n]
	goal_quat = _arr(rp.get("osc_goal_quat"), (-1, 4))[:n]
	qpos = _arr(rec.get("qpos"), (-1, 7))[:n]
	arrays = {
		"action": np.concatenate([goal_pos, goal_quat], axis=1),
		"eef_goal_pos": goal_pos, "eef_goal_quat": goal_quat,
		"eef_pos": _world_to_base(arm, trace_w),
		"eef_quat": _world_to_base_quat(arm, quat_w) if len(quat_w) else np.zeros((0, 4)),
		"qpos": qpos,
		"qvel": np.gradient(qpos, 1.0 / fps, axis=0) if n > 1 else np.zeros((n, 7)),
		"anchor_gap_m": _arr(rec.get("stale_anchor_m"), (-1,))[:n],
	}
	if rec.get("cursor_trace"):
		arrays["cursor"] = np.asarray(rec["cursor_trace"], dtype=np.int32)[:n]
	acts = rp.get("actions")
	if acts:
		pa = np.asarray(acts, dtype=np.float64).reshape(n, -1)
		arrays["policy_action"] = pa
		attrs["policy_action_format"] = POLICY_ACTION_LEROBOT if pa.shape[1] == 10 else POLICY_ACTION_REACH
	if rp.get("gain_action"):
		ga = np.asarray(rp["gain_action"], dtype=np.float64).reshape(n, 2)
		arrays["gain_action"] = ga
		attrs["gain_varies"] = bool(np.any(ga != ga[0]))
		trims = remap.get("tuning_gain_scales")
		if trims:
			# The physical gains the arm resolved, through the trims it ran with.
			kp6 = np.empty((n, 6)); kd6 = np.empty((n, 6))
			for t, (a_kp, a_kd) in enumerate(ga):
				kp6[t], kd6[t] = resolve_gains(a_kp, a_kd, trims["kp_ori_scale"], trims["kd_ori_scale"],
											   kp_pos_scale=trims["kp_pos_scale"], kd_pos_scale=trims["kd_pos_scale"])
			arrays["kp"], arrays["kd"] = kp6, kd6
	attrs.update({"num_samples": n, "steps": int(rec.get("steps", n)),
				  "qvel_source": "central_difference"})
	if "success" in rec:
		attrs["success"] = bool(rec["success"])
	if "cursor" in rec:
		attrs["cursor"] = int(rec["cursor"])
	return name, arrays, attrs, curve


def sim_episode(sim: dict, curve: dict | None) -> tuple:
	"""One sim_episode_NNN.json -> (name, arrays, attrs, curve)."""
	st = sim["steps"]
	n = len(st["eef_pos"])
	goal_pos = _arr(st["goal_pos"], (-1, 3))
	goal_quat = _arr(st["goal_quat"], (-1, 4))
	arrays = {
		"action": np.concatenate([goal_pos, goal_quat], axis=1),
		"eef_goal_pos": goal_pos, "eef_goal_quat": goal_quat,
		"eef_pos": _arr(st["eef_pos"], (-1, 3)), "eef_quat": _arr(st["eef_quat"], (-1, 4)),
		"qpos": _arr(st["qpos"], (-1, 7)), "qvel": _arr(st["qvel"], (-1, 7)),
		"tau_cmd": _arr(st["tau_cmd"], (-1, 7)),
	}
	if st.get("cursor"):
		arrays["cursor"] = np.asarray(st["cursor"], dtype=np.int32)
	for key, w in (("gain_action", 2), ("kp", 6), ("kd", 6)):
		if st.get(key):
			arrays[key] = _arr(st[key], (-1, w))
	conv = sim["sim"]["site_in_otee"]
	ctrl = sim["sim"].get("controller", {})
	start = sim.get("start", {})
	attrs = {
		"episode": int(sim.get("episode", 0)), "arm": sim.get("arm"), "source": "sim",
		"real_file": sim.get("source"), "mode": sim.get("mode"),
		"num_samples": n, "fps": float(sim["fps"]),
		"frame": "base_sim", "quat_order": "xyzw", "ee_convention": "robosuite_grip_site",
		"action_format": "absolute_pose_quat", "action_space": "EE_POS",
		"init_qpos": np.asarray(start["qpos"], dtype=np.float64),
		"start_qpos_max_err_rad": start.get("qpos_max_err_rad"),
		"ee_pos0": _opt(start.get("eef_pos")), "ee_quat0": _opt(start.get("eef_quat")),
		"site_in_otee_rotvec": np.asarray(conv["rotvec_rad"], dtype=np.float64),
		"site_in_otee_pos": np.asarray(conv["pos_m"], dtype=np.float64),
		"goal_transport_max_m": st.get("goal_transport_max_m"),
		"impedance_mode": ctrl.get("impedance_mode"), "controller": ctrl,
		"plant": sim["sim"].get("plant"),
		"sim_base_pos_world": _opt(sim["sim"].get("base_pos_world")),
		"control_freq": sim["sim"].get("control_freq"), "n_substeps": sim["sim"].get("n_substeps"),
		"generated_utc": sim.get("generated_utc"),
		"translated_from": "sim_episode.json",
	}
	out = sim.get("outcome", {})
	if "success" in out:
		attrs.update({"success": bool(out["success"]), "cursor": int(out.get("cursor", 0)),
					  "curve_len": int(out.get("curve_len", 0))})
	return episode_name(attrs["episode"]), arrays, attrs, curve


def translate(run_dir: Path, arm: str | None, remove_json: bool) -> int:
	results = run_dir / "results.json"
	if not results.is_file():
		print(f"{run_dir}: no results.json, skipped")
		return 0
	records = json.loads(results.read_text())
	meta_path = run_dir / "meta.json"
	meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None

	episodes, curves = [], {}
	for rec in records:
		ep = real_episode(rec, arm)
		if ep is None:
			print(f"  episode {rec.get('episode')}: predates the replay block (no dispatched goals); skipped")
			continue
		episodes.append(ep)
		curves[ep[0]] = ep[3]
	written = []
	if episodes:
		root = {"meta": meta} if meta else {}
		if episodes[0][2].get("arm"):
			root["arm"] = episodes[0][2]["arm"]
		path = episode_hdf5.write_episodes(run_dir / episode_hdf5.REAL_FILE, episodes, root,
										   producer="scripts/results_json_to_hdf5.py")
		written.append(Path(path))

	sims = []
	for sp in sorted(run_dir.glob("sim_episode_*.json")):
		sim = json.loads(sp.read_text())
		if sim.get("schema") != "sim_reach_replay/1":
			print(f"  {sp.name}: unexpected schema {sim.get('schema')!r}; skipped")
			continue
		ep = sim_episode(sim, curves.get(episode_name(sim.get("episode", 0))))
		sims.append(ep)
	if sims:
		path = episode_hdf5.write_episodes(run_dir / episode_hdf5.SIM_FILE, sims,
										   producer="scripts/results_json_to_hdf5.py")
		written.append(Path(path))

	bad = 0
	for path in written:
		problems = episode_hdf5.validate(path, legacy_ok=False)
		print(f"  {path.name}: {len(episode_hdf5.episode_names(path))} episode(s), "
			  f"{'OK' if not problems else f'{len(problems)} problem(s)'}")
		for pr in problems:
			print(f"    PROBLEM: {pr}")
		bad += bool(problems)
	if remove_json and written and not bad:
		for p in [results, meta_path, *run_dir.glob("sim_episode_*.json")]:
			if p.is_file():
				p.unlink()
		print("  removed the JSON originals")
	return bad


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("runs", nargs="+", help="run directories holding a results.json")
	ap.add_argument("--arm", default=None, choices=("left", "right"),
					help="for records without an arm field")
	ap.add_argument("--remove-json", action="store_true",
					help="delete results.json / meta.json / sim_episode_*.json once the "
						 "written files validate")
	args = ap.parse_args()
	bad = 0
	for run in args.runs:
		run_dir = Path(run).expanduser()
		print(run_dir)
		bad += translate(run_dir, args.arm, args.remove_json)
	return 1 if bad else 0


if __name__ == "__main__":
	raise SystemExit(main())
