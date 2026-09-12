#!/usr/bin/env python3
"""Run the reach task on the real right FR3 under the analytic base policy.

Phase 2: base policy only, EE_DELTA, orientation off. No residual, no reward.
Verify offline first -- scripts/check_real_reach_offline.py exercises the same
code path against a fake arm.

  python scripts/real_reach_rollout.py --episodes 5
  python scripts/real_reach_rollout.py --episodes 1 --dry-run   # no motion

Traces land in ~/franka_data/real_reach/<timestamp>/ (never in the repo), as
results.json plus one animated episode_NNN.html per episode. Everything in
results.json is WORLD frame; scripts/real_reach_viz.py re-renders it.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "multi-fast"))

import franka_config as fc  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot_robot_bimanual_franka import (  # noqa: E402
	ControlMode, SingleArmFrankaConfig, SingleArmRightConfig,
)
from lerobot_robot_bimanual_franka.reach_record import ReachEpisodeRecorder  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach import RealReach  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import (  # noqa: E402
	base_to_world, base_to_world_quat, keep_out_sphere, safety_z_floor_world,
	workspace_bounds_base,
)
from utils.base_policy_utils import ReachBaseWrapper  # noqa: E402


def build_base(cfg):
	bp = cfg["base_policy"]
	return ReachBaseWrapper(
		chunk_size=int(bp["chunk_size"]),
		prediction_horizon=int(bp["prediction_horizon"]),
		max_step=float(bp["max_step"]),
		lookahead_k=int(bp["lookahead_k"]),
		osc_output_max=float(bp["osc_output_max"]),
		osc_rot_output_max=float(bp["osc_rot_output_max"]),
		advance_threshold=float(cfg["task"]["advance_threshold_m"]),
		chunk_alpha=float(bp["chunk_alpha"]),
		velocity_lookahead_window=int(bp["velocity_lookahead_window"]),
		min_velocity_floor=float(bp["min_velocity_floor"]),
		include_orient=float(cfg["orientation"]["delta_max_deg"]) > 0.0,
		goal_mode=str(bp["goal_mode"]),
		control_freq=float(fc.control_fps()),
	)


def _git_sha(repo: Path) -> str | None:
	try:
		out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
							 capture_output=True, text=True, timeout=5)
		return out.stdout.strip() or None
	except Exception:
		return None


def run_metadata(arm: str, seed: int) -> dict:
	"""Everything needed to say what produced a run and what the arm was running.

	The sim comparison is only meaningful against a known controller, so the
	`torque.osc` law and the `tuning` rig trims are recorded rather than assumed
	-- `tuning.friction_kc` in particular has no sim counterpart.
	"""
	return {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"arm": arm, "seed": seed, "fps": float(fc.control_fps()),
		"command": " ".join(sys.argv),
		"git": {"franka_ws": _git_sha(_ROOT), "multi_fast": _git_sha(_ROOT / "multi-fast")},
		"osc": {k: fc.control(f"torque.osc.{k}") for k in
				("default_kp", "gain_exp_base", "uncouple_pos_ori",
				 "cross_coupling_compensation", "lambda_rcond")},
		"tuning": fc.control("tuning"),
		"limits": fc.control("torque.limits"),
		"delta": fc.control("torque.delta"),
		"bounds_base": workspace_bounds_base(arm).tolist(),
		"safety_z_floor_world": safety_z_floor_world(arm),
	}


def flush(run_dir: Path, results: list) -> None:
	"""Rewrite results.json after every episode.

	A hardware run is expensive and a fault mid-run used to lose all of it: the
	file was written once, after the loop. tmp+rename so a reader never sees a
	half-written file (same reason sysid.save_sysid_hdf5 does it).
	"""
	tmp = run_dir / "results.json.tmp"
	tmp.write_text(json.dumps(results, indent=1))
	tmp.replace(run_dir / "results.json")


def write_viz(run_dir: Path, results: list, stride: int) -> None:
	"""One animated HTML per episode, next to results.json.

	Imported late on purpose: results.json is already written by the time this
	runs, so a missing plotly costs the rendering and not the traces.
	"""
	try:
		from real_reach_viz import save_run_html
		for path in save_run_html(run_dir, results, frame_stride=stride):
			print(f"viz -> {path}")
	except Exception as e:
		print(f"viz skipped ({type(e).__name__}: {e}); render later with\n"
			  f"  python scripts/real_reach_viz.py {run_dir}")


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--episodes", type=int, default=5)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--dry-run", action="store_true",
					help="connect and sample, but never send an action")
	ap.add_argument("--out", default=str(Path.home() / "franka_data" / "real_reach"))
	ap.add_argument("--arm", default="left", choices=("left", "right"),
					help="physical arm to drive; the key prefix stays r_ either way")
	ap.add_argument("--no-viz", action="store_true",
					help="write results.json only, no episode HTML")
	ap.add_argument("--viz-stride", type=int, default=1,
					help="animate every Nth step in the HTML")
	args = ap.parse_args()

	cfg = fc.section("reach")
	if float(cfg["orientation"]["delta_max_deg"]) > 0.0:
		print("NOTE: orientation is enabled; phase 2 was scoped position-only.")

	arm = args.arm
	if not fc.robot_base_in_world_verified(arm):
		print(f"WARNING: robot_base_in_world({arm!r}) is unverified in config/world.yaml.\n"
			  f"         The worktable floor derives from it. Verify before running near the table.")

	# Both profiles expose their arm under the r_ prefix; single_arm_franka is
	# the left FR3, single_arm_right the physical right one.
	cls = SingleArmFrankaConfig if arm == "left" else SingleArmRightConfig
	print(f"arm {arm} via {cls.__name__}")
	robot = make_robot_from_config(cls(control_mode=ControlMode.EE_DELTA))
	robot.connect()
	try:
		env = RealReach(robot, arm=arm, arm_key="r", seed=args.seed)
		base = build_base(cfg)
		floor = safety_z_floor_world(arm)
		keep_out = keep_out_sphere(arm)
		if keep_out is None:
			print("WARNING: reach.workspace.keep_out is disabled and safety.py has no\n"
				  "         arm-repel — nothing keeps this arm off the other one.")
		else:
			print(f"keep-out: {np.round(keep_out[0], 3)} (world) r={keep_out[1]:.3f} m\n"
				  f"         CONFIRM the other arm is where this assumes before proceeding.")

		run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
		run_dir.mkdir(parents=True, exist_ok=True)
		# Written before the arm moves, so a session that faults on episode 0 still
		# says what it was running.
		(run_dir / "meta.json").write_text(json.dumps(run_metadata(arm, args.seed), indent=1))
		results = []

		recorder = ReachEpisodeRecorder(arm, args.seed)
		for ep in range(args.episodes):
			obs = env.reset()
			done, info = False, {}
			recorder.begin_reach(env, ep)
			print(f"\nepisode {ep}: goal(base) {np.round(env.goal, 3)}  "
				  f"world {np.round(base_to_world(arm, env.goal), 3)}")
			if args.dry_run:
				print("  dry run — curve sampled, no action sent")
				results.append(recorder.dry_run())
				flush(run_dir, results)
				continue

			while not done:
				batched = {k: (v[None, ...] if isinstance(v, np.ndarray) else v)
					   for k, v in obs.items() if v is not None}
				for a in np.asarray(base(batched))[0]:
					seen = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
					obs, _, done, info = env.step(a)
					pos_w = recorder.record_reach(seen, obs, info)
					if pos_w[2] < floor:
						print(f"  !! EE at world z {pos_w[2]:.4f}, under floor {floor:.4f}")
					if keep_out is not None:
						gap = float(np.linalg.norm(pos_w - keep_out[0]))
						if gap < keep_out[1]:
							print(f"  !! EE {gap:.4f} m from keep-out centre, "
								  f"inside {keep_out[1]:.4f} m")
					if done:
						break

			ok = info.get("success", False)
			print(f"  {'SUCCESS' if ok else 'timeout'} in {info['episode_steps']} steps, "
				  f"cursor {info['next_waypoint_idx']}/{env._curve_len}")
			results.append(recorder.finish_reach(info))
			flush(run_dir, results)

		done_eps = [r for r in results if not r.get("dry_run")]
		if done_eps:
			n_ok = sum(r["success"] for r in done_eps)
			print(f"\n{n_ok}/{len(done_eps)} succeeded")
		print(f"traces -> {run_dir}")
		if not args.no_viz:
			write_viz(run_dir, results, args.viz_stride)
	finally:
		robot.disconnect()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
