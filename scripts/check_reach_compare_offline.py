#!/usr/bin/env python3
"""Exercise the sim/real reach comparison with no robot and no robosuite.

Synthesises a sim record from a real run by perturbing it in known ways, then
asserts the viz and the error summary report exactly the perturbation. Catches
the three things that fail silently: a frame inversion, a sim trace clipped out
of the scene, and animation frames mapped onto the wrong traces.

  python scripts/check_real_reach_offline.py -n 3 --out /tmp/fake_run
  python scripts/check_reach_compare_offline.py /tmp/fake_run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from lerobot_robot_bimanual_franka import gain_schedule as gs  # noqa: E402
from lerobot_robot_bimanual_franka.osc_torque_controller import resolve_gains  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import base_to_world  # noqa: E402
from plot_episodes import (  # noqa: E402
	_episode_arrays, _sim_arrays, build_reach_figure, compute_reach_errors, load_run,
)


def synth_sim(ep: dict, arm: str, offset=(0.0, 0.0, 0.0), lag: int = 0) -> dict:
	"""A sim episode that IS the real episode, in sim's frame and convention.

	Built by inverting every map the comparison applies, so a zero perturbation
	must round-trip to zero error. `offset` is added in BASE frame; `lag`
	truncates sim, standing in for an episode that ended earlier.
	"""
	arr, at = ep["arrays"], ep["attrs"]
	# The site offset a sim replay would measure on the shipped Panda. Any rigid
	# tool-frame offset must round-trip; these are the real numbers so the test
	# also reads as documentation.
	site_rot = Rotation.from_rotvec([0.0, 0.0, -1.571289])
	site_pos = np.array([0.0, 0.0, -0.0069])

	pos_b = np.asarray(arr["eef_pos"], dtype=np.float64) + np.asarray(offset)
	quat_b = np.asarray(arr["eef_quat"], dtype=np.float64)
	goal_b = np.asarray(arr["eef_goal_pos"], dtype=np.float64)
	gq = np.asarray(arr["eef_goal_quat"], dtype=np.float64)

	def to_site(p, q):
		"""O_T_EE -> robosuite grip site, the map the sim replay applies to goals."""
		r = Rotation.from_quat(q)
		return np.asarray(p) + r.apply(site_pos), (r * site_rot).as_quat()

	site = [to_site(p, q) for p, q in zip(pos_b, quat_b)]
	gsite = [to_site(p, q) for p, q in zip(goal_b, gq)]
	n = len(site) - lag
	spos = np.asarray([p for p, _ in site[:n]]).reshape(n, 3)
	squat = np.asarray([q for _, q in site[:n]]).reshape(n, 4)
	gpos = np.asarray([p for p, _ in gsite[:n]]).reshape(n, 3)
	gquat = np.asarray([q for _, q in gsite[:n]]).reshape(n, 4)
	arrays = {
		"action": np.concatenate([gpos, gquat], axis=1),
		"eef_pos": spos, "eef_quat": squat, "eef_goal_pos": gpos, "eef_goal_quat": gquat,
		"qpos": np.asarray(arr["qpos"], dtype=np.float64)[:n],
		"qvel": np.asarray(arr["qvel"], dtype=np.float64)[:n],
	}
	if "cursor" in arr:
		arrays["cursor"] = np.asarray(arr["cursor"])[:n]
	for key in ("gain_action", "kp", "kd"):
		if key in arr:
			arrays[key] = np.asarray(arr[key], dtype=np.float64)[:n]
	attrs = {
		"num_samples": n, "fps": at["fps"], "frame": "base_sim", "quat_order": "xyzw",
		"ee_convention": "robosuite_grip_site", "action_format": "absolute_pose_quat",
		"action_space": "EE_POS", "init_qpos": np.asarray(at["init_qpos"], dtype=np.float64),
		"site_in_otee_rotvec": site_rot.as_rotvec(), "site_in_otee_pos": site_pos,
		"goal_transport_max_m": 0.0, "start_qpos_max_err_rad": 0.0,
		"impedance_mode": "variable" if "gain_action" in arr else "fixed",
	}
	if "cursor" in arr:
		attrs.update({"success": bool(at.get("success", False)),
					  "cursor": int(np.asarray(arr["cursor"])[n - 1]),
					  "curve_len": int(at["curve_len"])})
	return {"name": ep["name"], "arrays": arrays, "attrs": attrs, "curve": ep.get("curve")}


def check(ep: dict, arm: str) -> list[str]:
	fails = []

	def req(cond, msg):
		print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
		if not cond:
			fails.append(msg)

	d = _episode_arrays(ep, arm)

	# (a) round trip: every map the comparison applies must invert exactly.
	sim = synth_sim(ep, arm)
	e = compute_reach_errors(d, _sim_arrays(sim, arm), ep, sim)
	req(e["position_error_m"]["max"] < 1e-9,
		f"zero-perturbation round trip == 0 (got {e['position_error_m']['max']:.2e} m)")
	req(e["rotation_error_rad"]["max"] < 1e-9,
		f"orientation round trip == 0 (got {e['rotation_error_rad']['max']:.2e} rad)")

	# (b) frame direction: a pure +x BASE offset must read as +x in base, and the
	# world-frame trail must sit 10 mm away along the arm's YAWED x.
	sim = synth_sim(ep, arm, offset=(0.010, 0.0, 0.0))
	s = _sim_arrays(sim, arm)
	e = compute_reach_errors(d, s, ep, sim)
	req(abs(e["position_error_axis_m"]["x"]["mean"] - 0.010) < 1e-9,
		f"10 mm base +x reads as +x in base (got {e['position_error_axis_m']['x']['mean']:.6f})")
	req(abs(e["position_error_m"]["max"] - 0.010) < 1e-9,
		"offset magnitude preserved through base->world")
	yawed = base_to_world(arm, np.array([0.010, 0.0, 0.0])) - base_to_world(arm, np.zeros(3))
	moved = s["trace"][0] - d["trace"][0]
	req(np.allclose(moved, yawed, atol=1e-9),
		f"world trail moves along the arm's yawed x, not world x (got {np.round(moved, 5)})")

	# (c) truncation
	sim = synth_sim(ep, arm, lag=5)
	e = compute_reach_errors(d, _sim_arrays(sim, arm), ep, sim)
	req(e["n_steps_compared"] == min(e["n_steps_real"], e["n_steps_sim"]),
		f"truncated to the shorter side ({e['n_steps_compared']} steps)")

	# (d)/(e) figure integrity, both with and without sim
	sim = synth_sim(ep, arm)
	for label, kw in (("hardware-only", {}), ("with-sim", {"sim": sim})):
		f = build_reach_figure(ep, arm=arm, **kw)
		req(all(len(fr.traces) == len(fr.data) for fr in f.frames),
			f"{label}: every frame supplies one trace per index")
		req(all(tuple(fr.traces) == tuple(f.frames[0].traces) for fr in f.frames),
			f"{label}: frame trace indices are stable")
		req(max(f.frames[0].traces) < len(f.data),
			f"{label}: frame indices are in range")
		if kw:
			r = [f.layout.scene.xaxis.range, f.layout.scene.yaxis.range,
				 f.layout.scene.zaxis.range]
			p = _sim_arrays(sim, arm)["trace"]
			req(all(r[i][0] <= p[:, i].min() and p[:, i].max() <= r[i][1] for i in range(3)),
				f"{label}: sim trace lies inside the hard scene bounds")

	# (f) transport check surfaces
	sim = synth_sim(ep, arm)
	sim["attrs"]["goal_transport_max_m"] = 1e-3
	fig = build_reach_figure(ep, arm=arm, sim=sim)
	req("REPLAY NOT FAITHFUL" in fig.layout.title.text,
		"an unfaithful goal transport is called out in the title")

	# (g) a record whose gain action moved: the error-by-gain table appears, the
	# round trip is still exactly zero, and the figure grows a gain panel.
	req("by_gain" not in compute_reach_errors(d, _sim_arrays(synth_sim(ep, arm), arm), ep,
											  synth_sim(ep, arm)),
		"no gain record -> no by_gain")
	gep = {"name": ep["name"], "arrays": dict(ep["arrays"]), "attrs": dict(ep["attrs"]),
		   "curve": ep.get("curve")}
	n_steps = len(gep["arrays"]["eef_goal_pos"])
	t_s = np.arange(n_steps) / float(gep["attrs"]["fps"])
	gep["arrays"]["gain_action"] = ga = gs.quadrature_schedule(t_s, 0.3, 0.3, 0.25)
	remap = gs.remap_constants()
	trims = remap["tuning_gain_scales"]
	resolved = [resolve_gains(a[0], a[1], trims["kp_ori_scale"], trims["kd_ori_scale"],
							  kp_pos_scale=trims["kp_pos_scale"], kd_pos_scale=trims["kd_pos_scale"]) for a in ga]
	gep["arrays"]["kp"] = np.asarray([r[0] for r in resolved])
	gep["arrays"]["kd"] = np.asarray([r[1] for r in resolved])
	gep["attrs"].update({k: v for k, v in remap.items() if k != "tuning_gain_scales"})
	gd = _episode_arrays(gep, arm)
	req(gd["gain_action"].shape == (len(gd["commanded"]), 2),
		"gain_action is read per step, aligned with the commands")
	gsim = synth_sim(gep, arm)
	e = compute_reach_errors(gd, _sim_arrays(gsim, arm), gep, gsim)
	req("by_gain" in e and e["sim_impedance_mode"] == "variable",
		"a moving gain action yields by_gain and reports the sim's impedance mode")
	if "by_gain" in e:
		bg = e["by_gain"]
		req(all(bg[ch][t]["n"] > 0 for ch in ("a_kp", "a_kd") for t in ("low", "mid", "high")),
			"three non-empty terciles per gain channel")
		req(all(bg[ch][t]["rmse_pos_mm"] < 1e-6 for ch in ("a_kp", "a_kd") for t in ("low", "mid", "high")),
			"zero-perturbation round trip is 0 in every tercile")
		req(bg["a_kp"]["low"]["range"][1] <= bg["a_kp"]["mid"]["range"][0] + 1e-12
			<= bg["a_kp"]["high"]["range"][0] + 1e-12, "terciles are ordered")
	f = build_reach_figure(gep, arm=arm, sim=gsim)
	titles = [a.text for a in f.layout.annotations] if f.layout.annotations else []
	titles = {ax.title.text for ax in (f.layout[k] for k in f.layout if k.startswith("yaxis"))
			  if ax.title.text}
	req("gain action (-1..1)" in titles and "kp, kd (log)" in titles,
		"with-gain figure carries the gain panel: action on the left axis, kp/kd on the right")
	names = {tr.name for tr in f.data if tr.name}
	req({"a_kp", "a_kd", "kp", "kd"} <= names, "gain panel draws a_kp, a_kd, kp and kd")
	req("kp (sim)" in names and "kd (sim)" in names, "gain panel draws the sim's resolved kp/kd dashed")
	req(all(len(fr.traces) == len(fr.data) for fr in f.frames),
		"with-gain: every frame supplies one trace per index")
	return fails


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("run", help="a run directory written by --out or by the rollout")
	ap.add_argument("--episode", type=int, default=0)
	args = ap.parse_args()

	_, eps = load_run(args.run)
	ep = next(e for e in eps if int(e["attrs"].get("episode", -1)) == args.episode
			  and not e["attrs"].get("dry_run"))
	arm = ep["attrs"].get("arm", "left")
	print(f"{ep['name']}, arm {arm}, {ep['attrs']['num_samples']} steps")
	fails = check(ep, arm)
	print("\nPASS" if not fails else f"\nFAIL ({len(fails)})")
	return 0 if not fails else 1


if __name__ == "__main__":
	raise SystemExit(main())
