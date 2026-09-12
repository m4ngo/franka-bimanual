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
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import franka_config as fc  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import base_to_world  # noqa: E402
from real_reach_viz import (  # noqa: E402
	_episode_arrays, _sim_arrays, build_reach_figure, compute_reach_errors, load_run,
)


def _world_to_base(arm: str, pts: np.ndarray) -> np.ndarray:
	"""TEST ONLY. Inverts `base_to_world` to rebuild the input this harness needs.

	Shipped code never does this -- `robot_base_in_world` maps base->world and is
	not inverted by consumers. Here it only reconstructs, exactly (R is
	orthonormal), the base-frame trace that produced the recorded world one, so
	the round-trip assertion has something to be exact against.
	"""
	pose = fc.robot_base_in_world(arm)
	R = np.asarray(pose.rotation)
	return (np.asarray(pts, dtype=np.float64) - np.asarray(pose.translation)) @ R


def _world_to_base_quat(arm: str, quats: np.ndarray) -> np.ndarray:
	pose = fc.robot_base_in_world(arm)
	return (Rotation.from_matrix(np.asarray(pose.rotation)).inv()
			* Rotation.from_quat(np.asarray(quats, dtype=np.float64))).as_quat()


def synth_sim(ep: dict, arm: str, offset=(0.0, 0.0, 0.0), lag: int = 0) -> dict:
	"""A sim record that IS the real episode, in sim's frame and convention.

	Built by inverting every map the comparison applies, so a zero perturbation
	must round-trip to zero error. `offset` is added in BASE frame; `lag`
	truncates sim, standing in for an episode that ended earlier.
	"""
	rp = ep["replay"]
	# The site offset a sim replay would measure on the shipped Panda. Any rigid
	# tool-frame offset must round-trip; these are the real numbers so the test
	# also reads as documentation.
	site_rot = Rotation.from_rotvec([0.0, 0.0, -1.571289])
	site_pos = np.array([0.0, 0.0, -0.0069])

	pos_b = _world_to_base(arm, ep["trace"]) + np.asarray(offset)
	quat_b = _world_to_base_quat(arm, ep["ee_quat"])
	goal_b = np.asarray(rp["osc_goal_pos"], dtype=np.float64)
	gq = np.asarray(rp["osc_goal_quat"], dtype=np.float64)

	def to_site(p, q):
		"""O_T_EE -> robosuite grip site, the map the sim replay applies to goals."""
		r = Rotation.from_quat(q)
		return np.asarray(p) + r.apply(site_pos), (r * site_rot).as_quat()

	site = [to_site(p, q) for p, q in zip(pos_b, quat_b)]
	gsite = [to_site(p, q) for p, q in zip(goal_b, gq)]
	n = len(site) - lag
	return {
		"schema": "sim_reach_replay/1", "episode": ep["episode"],
		"frame": "base_sim", "quat_order": "xyzw", "fps": ep["fps"],
		"ee_convention": "robosuite_grip_site", "mode": "replay_absolute_goal",
		"steps": {
			"eef_pos": [p.tolist() for p, _ in site[:n]],
			"eef_quat": [q.tolist() for _, q in site[:n]],
			"goal_pos": [p.tolist() for p, _ in gsite[:n]],
			"goal_quat": [q.tolist() for _, q in gsite[:n]],
			"qpos": np.asarray(ep["qpos"], dtype=np.float64)[:n].tolist(),
			"cursor": ep["cursor_trace"][:n],
			"goal_transport_max_m": 0.0,
		},
		"start": {"qpos_max_err_rad": 0.0},
		"sim": {"site_in_otee": {"rotvec_rad": site_rot.as_rotvec().tolist(),
								 "pos_m": site_pos.tolist()}},
		"outcome": {"steps": n, "success": bool(ep["success"]),
					"cursor": ep["cursor_trace"][n - 1], "curve_len": ep["curve_len"]},
	}


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
	sim["steps"]["goal_transport_max_m"] = 1e-3
	fig = build_reach_figure(ep, arm=arm, sim=sim)
	req("REPLAY NOT FAITHFUL" in fig.layout.title.text,
		"an unfaithful goal transport is called out in the title")
	return fails


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("run", help="a run directory written by --out or by the rollout")
	ap.add_argument("--episode", type=int, default=0)
	args = ap.parse_args()

	_, eps = load_run(args.run)
	ep = next(e for e in eps if int(e.get("episode", -1)) == args.episode
			  and not e.get("dry_run"))
	arm = ep.get("arm", "left")
	print(f"episode {ep['episode']}, arm {arm}, {ep['steps']} steps")
	fails = check(ep, arm)
	print("\nPASS" if not fails else f"\nFAIL ({len(fails)})")
	return 0 if not fails else 1


if __name__ == "__main__":
	raise SystemExit(main())
