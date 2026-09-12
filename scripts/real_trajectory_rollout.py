#!/usr/bin/env python3
"""One LeRobot dataset episode as a real-side trajectory record, for a sim diff.

Emits the same `results.json` the reach rollout does -- base-frame commands
under `replay`, world-frame measurements at top level -- so
`multi-fast/scripts/reach/replay_real_reach.py` and `scripts/real_reach_viz.py`
consume it unchanged. There is just no curve.

The episode's EE_DELTA actions are converted OFFLINE to the absolute OSC goals
they produced, through the robot's own `OSCGoalBuilder` + `ActionSafetyScreen`
(`scripts/replay_dataset.py::to_ee_pose_actions`), anchored on the episode's
own recorded joint states. Those goals are what both sides then track.

Two things can stand in for "real":

  --source dataset   No arm. The trajectory IS the recording: EE from FK of the
                     recorded joints, goals from the offline conversion. Compare
                     sim against what the arm did when the dataset was made.
                     NOTE the recording predates the controller changes since
                     (cross_coupling_compensation was on); label it as such.
  --source arm       Re-run the goals on the arm now, EE_POS, and measure.
                     Compares sim against today's controller. Homes to the
                     episode's own start configuration first.

  python scripts/real_trajectory_rollout.py --repo-id HuskyMango/basket-7-24 --episode 3
  python scripts/real_trajectory_rollout.py --repo-id HuskyMango/basket-7-24 --episode 3 --source arm

Output: ~/franka_data/real_traj/<repo>/ep<NNN>_<source>/results.json (+ HTML).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import franka_config as fc  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot_robot_bimanual_franka import ControlMode, SingleArmFrankaConfig, SingleArmRightConfig  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.reach_record import EpisodeRecorder  # noqa: E402
from real_reach_rollout import flush, run_metadata, write_viz  # noqa: E402
from replay_dataset import NUM_JOINTS, episode_slice, to_ee_pose_actions  # noqa: E402

KEY = "r"      # both single-arm profiles expose their arm under r_


def load_episode(repo_id: str, episode: int, root: str | None) -> dict:
	ds = LeRobotDataset(repo_id, root=root)
	n_eps = int(ds.meta.total_episodes)
	if not 0 <= episode < n_eps:
		raise SystemExit(f"{repo_id} has {n_eps} episodes (0..{n_eps - 1}); no episode {episode}")
	a, b = episode_slice(ds.meta, episode)
	actions = np.asarray(ds.hf_dataset["action"][a:b], dtype=np.float64)
	states = np.asarray(ds.hf_dataset["observation.state"][a:b], dtype=np.float64)
	if actions.shape[1] != 10 or states.shape[1] != NUM_JOINTS + 1:
		raise SystemExit(f"unexpected feature shapes action{actions.shape} state{states.shape}")
	# kp/kd ride in the last two action columns. Sim replays under fixed
	# impedance, which is exact only if the recording never moved them.
	if np.any(actions[:, 8:] != 0.0):
		raise SystemExit("episode carries nonzero kp/kd gain actions; the sim replay's "
						 "fixed impedance cannot follow them")
	if int(ds.meta.fps) != int(fc.control_fps()):
		raise SystemExit(f"dataset fps {ds.meta.fps} != control fps {fc.control_fps()}")
	return {"actions": actions, "states": states, "fps": int(ds.meta.fps),
			"n": int(b - a)}


def record_from_dataset(rec: EpisodeRecorder, ep: dict, goals: np.ndarray,
						episode: int) -> dict:
	"""The recording as the real trajectory. No hardware."""
	q = ep["states"][:, :NUM_JOINTS]
	pos, quat = eef_poses_from_qpos(q)
	rec.begin(episode, q[0], quat[0], ee_pos0=pos[0])
	for t in range(ep["n"]):
		# The conversion anchored each goal on state t, so state t is also the
		# pose the policy acted on: the anchor gap is zero by construction.
		rec.record(pos[t], pos[t], quat[t], q[t], goals[t, :3], goals[t, 3:7],
				   pos[t], ep["actions"][t])
	return rec.finish(ep["n"])


def record_from_arm(rec: EpisodeRecorder, ep: dict, goals: np.ndarray,
					episode: int, robot, dry_run: bool) -> dict:
	"""Re-run the goal sequence on the arm, EE_POS, and measure."""
	q0 = ep["states"][0, :NUM_JOINTS]
	if not robot.home(home_q_left=None, home_q_right=q0,
					  gripper_norm=float(ep["states"][0, NUM_JOINTS])):
		print("WARNING: homing did not converge; replaying anyway")
	snap = robot.robot_manager.current_kinematic_state_batch([KEY])[KEY]
	rec.begin(episode, np.asarray(snap[0]), np.asarray(snap[4]), ee_pos0=np.asarray(snap[3]))
	if dry_run:
		print("  dry run — homed and converted, no action sent")
		return rec.dry_run()

	keys = list(robot.action_features)
	period = 1.0 / float(fc.control_fps())
	deadline = time.perf_counter()
	for t in range(ep["n"]):
		seen = np.asarray(snap[3], dtype=np.float64)
		robot.send_action({k: float(v) for k, v in zip(keys, goals[t])})
		gp, gq = robot._last_osc_goal[KEY]
		ap, _ = robot._last_osc_anchor[KEY]
		# Read at the END of the period so trace[t] is the response to goal t,
		# which is what sim's step t records. Read right after the send, the arm
		# has not moved yet and real sits one step behind sim throughout.
		deadline += period
		now = time.perf_counter()
		if deadline > now:
			time.sleep(deadline - now)
		else:
			deadline = now
		snap = robot.robot_manager.current_kinematic_state_batch([KEY])[KEY]
		rec.record(seen, snap[3], snap[4], snap[0], gp, gq, ap, goals[t])
	return rec.finish(ep["n"])


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--repo-id", required=True)
	ap.add_argument("--episode", type=int, required=True)
	ap.add_argument("--root", default=None, help="local dataset root (default: the HF cache)")
	ap.add_argument("--source", choices=("dataset", "arm"), default="dataset")
	ap.add_argument("--arm", default="left", choices=("left", "right"),
					help="physical arm; the key prefix stays r_ either way")
	ap.add_argument("--out", default=str(Path.home() / "franka_data" / "real_traj"))
	ap.add_argument("--dry-run", action="store_true", help="--source arm: home and convert only")
	ap.add_argument("--no-viz", action="store_true")
	ap.add_argument("--viz-stride", type=int, default=2)
	args = ap.parse_args()

	ep = load_episode(args.repo_id, args.episode, args.root)
	cls = SingleArmFrankaConfig if args.arm == "left" else SingleArmRightConfig
	cfg = cls(control_mode=ControlMode.EE_POS)
	goals = to_ee_pose_actions(ep["actions"], ep["states"], cfg)
	print(f"{args.repo_id} episode {args.episode}: {ep['n']} steps, "
		  f"|delta| p99 {np.percentile(np.abs(ep['actions'][:, :3]), 99):.4f} m, "
		  f"start q {np.round(ep['states'][0, :NUM_JOINTS], 3)}")

	run_dir = (Path(args.out) / args.repo_id.replace("/", "__")
			   / f"ep{args.episode:03d}_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
	run_dir.mkdir(parents=True, exist_ok=True)
	meta = run_metadata(args.arm, seed=-1)
	meta.update({"repo_id": args.repo_id, "episode": args.episode, "source": args.source})
	(run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

	rec = EpisodeRecorder(args.arm, source=args.source)
	if args.source == "dataset":
		results = [record_from_dataset(rec, ep, goals, args.episode)]
	else:
		from lerobot.robots.utils import make_robot_from_config
		robot = make_robot_from_config(cfg)
		robot.connect()
		try:
			results = [record_from_arm(rec, ep, goals, args.episode, robot, args.dry_run)]
		finally:
			robot.disconnect()
	flush(run_dir, results)
	print(f"traces -> {run_dir}")
	if not args.no_viz:
		write_viz(run_dir, results, args.viz_stride)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
