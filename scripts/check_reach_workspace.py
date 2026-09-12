#!/usr/bin/env python3
"""Offline validation for the real reach workspace. No hardware.

Samples many episodes and asserts every waypoint is (a) clear of the worktable
by the safety margin and (b) inside the arm's fast-reach radius. Run this before
the arm ever moves, and again after touching config/reach.yaml.

  python scripts/check_reach_workspace.py --arm right -n 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import (  # noqa: E402
	_home_key,
	base_to_world,
	keep_out_sphere,
	safety_z_floor_world,
	sample_episode,
	workspace_bounds_base,
)


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--arm", default="right")
	ap.add_argument("-n", "--n-episodes", type=int, default=10000)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--start-jitter-m", type=float, default=0.01,
	                help="std-dev of reset-to-reset EE start variation")
	args = ap.parse_args()

	arm = args.arm
	bounds = workspace_bounds_base(arm)
	floor_world = safety_z_floor_world(arm)
	base = fc.robot_base_in_world(arm)
	radius = float(fc.section("reach")["workspace"]["reachable_radius_m"])
	verified = fc.robot_base_in_world_verified(arm)
	try:
		keep_out = keep_out_sphere(arm)
	except ValueError as exc:
		# No silent fallback: an unvalidated keep-out is the failure this check exists to catch.
		print(f"cannot check --arm {arm}: {exc}")
		return 1

	print(f"arm                : {arm}")
	print(f"base translation   : {np.asarray(base.translation)}")
	print(f"base verified      : {verified}")
	print(f"worktable (world z): {fc.worktable_height_m():.4f}")
	print(f"safety floor world : {floor_world:.4f}  (EE must stay above, any orientation)")
	print(f"bounds base xyz min: {bounds[0]}")
	print(f"bounds base xyz max: {bounds[1]}")
	print(f"usable z span      : {bounds[1, 2] - bounds[0, 2]:.4f} m")
	if keep_out is None:
		print("keep-out           : DISABLED — nothing separates this arm from the other")
	else:
		print(f"keep-out centre    : {np.round(keep_out[0], 4)} (world)")
		print(f"keep-out radius    : {keep_out[1]:.4f} m")
	if not verified:
		print("\n  WARNING: this arm's base pose is NOT verified in config/world.yaml.")
		print("  Every world-frame bound below inherits that uncertainty.\n")

	rng = np.random.default_rng(args.seed)
	worst_clear = np.inf
	worst_radius = 0.0
	n_fail_floor = n_fail_reach = n_fail_keep = n_no_curve = 0
	worst_keep = np.inf
	starts = []

	# The real start is wherever homing leaves the EE, not an arbitrary point in
	# the box — the box corners are 1.1 m from the base, well outside the
	# fast-reach sphere, so uniform starts test a distribution the arm never
	# visits. Jitter models reset-to-reset variation.
	home_base, _ = eef_poses_from_qpos(np.asarray(fc.home_q(key=_home_key(arm)))[None, :])
	home_base = home_base[0]
	print(f"home EE (base)     : {home_base}  |p|={np.linalg.norm(home_base):.4f}")
	print(f"home EE (world)    : {base_to_world(arm, home_base)}")

	n_start_oob = 0
	for i in range(args.n_episodes):
		start = home_base + rng.normal(0.0, args.start_jitter_m, size=3)
		try:
			ep = sample_episode(arm, start, rng)
		except ValueError:
			# Reset variation put the homed EE outside the box. Counted, not
			# fatal: it measures how much margin the home pose actually has.
			n_start_oob += 1
			continue
		except RuntimeError:
			# No curve survived max_sample_attempts. A breach of the acceptance
			# rate, not of a bound, so it is counted and reported separately.
			n_no_curve += 1
			continue
		pts = np.vstack([ep["waypoints"], ep["goal"][None, :]])

		world_z = base_to_world(arm, pts)[:, 2]
		clearance = float(world_z.min() - floor_world)
		worst_clear = min(worst_clear, clearance)
		if clearance < 0:
			n_fail_floor += 1
			if n_fail_floor == 1:
				print(f"\nFLOOR BREACH at episode {i}: min world z {world_z.min():.4f} "
					  f"< floor {floor_world:.4f}")

		if keep_out is not None:
			gap = float(np.linalg.norm(base_to_world(arm, pts) - keep_out[0], axis=1).min())
			worst_keep = min(worst_keep, gap - keep_out[1])
			if gap < keep_out[1]:
				n_fail_keep += 1
				if n_fail_keep == 1:
					print(f"\nKEEP-OUT BREACH at episode {i}: {gap:.4f} m "
						  f"< {keep_out[1]:.4f} m")

		dist = float(np.linalg.norm(pts, axis=1).max())   # base frame, base at origin
		worst_radius = max(worst_radius, dist)
		if dist > radius:
			n_fail_reach += 1
			if n_fail_reach == 1:
				print(f"\nREACH BREACH at episode {i}: max |p| {dist:.4f} > {radius:.4f}")
		starts.append(start)

	print(f"\nsampled {args.n_episodes} episodes")
	print(f"  worst floor clearance : {worst_clear:+.4f} m  ({n_fail_floor} breaches)")
	print(f"  worst reach distance  : {worst_radius:.4f} m / {radius:.4f}  ({n_fail_reach} breaches)")
	if keep_out is not None:
		print(f"  worst keep-out margin : {worst_keep:+.4f} m  ({n_fail_keep} breaches)")
	print(f"  curves not found      : {n_no_curve} ({n_no_curve / args.n_episodes:.2%}) — max_sample_attempts exhausted")
	print(f"  starts outside box    : {n_start_oob} ({n_start_oob / args.n_episodes:.2%}) — home-pose margin vs {args.start_jitter_m*100:.0f} cm jitter")

	ok = n_fail_floor == 0 and n_fail_reach == 0 and n_fail_keep == 0 and n_no_curve == 0
	print("\nPASS" if ok else "\nFAIL")
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
