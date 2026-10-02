#!/usr/bin/env python3
"""LeRobot dataset episodes as real-side trajectory records, for a sim diff or a fit.

Emits the same `episodes.hdf5` the reach rollout does (base frame, see
EPISODE_HDF5.md), so `multi-fast/scripts/reach/replay_goals_in_sim.py` and
`scripts/plot_episodes.py` consume it unchanged. There is just no curve.
`--episode N` records one episode; `--all` records every episode of the
dataset into one file, as `ep000`, `ep001`, ... -- the input the plant fit
wants, once `sysid/merge_episodes.py` has joined it with an excitation run.

The episode's actions are converted OFFLINE to the absolute OSC goals they
produced, through the robot's own `OSCGoalBuilder` + `ActionSafetyScreen`
(`scripts/replay_dataset.py::episode_goals`). An EE_DELTA recording is anchored
on its own recorded joint states; an EE_POS recording is its rows, screened.
Which one it is gets decided by reach, because both write the same feature
names -- a pose read as a delta becomes "5 cm further and 0.5 rad more from
wherever you are" every step, which winds the wrist up against its limit.
Those goals are what both sides then track.

Two things can stand in for "real":

  --source dataset   No arm. The trajectory IS the recording: EE from FK of the
                     recorded joints, goals from the offline conversion. Compare
                     sim against what the arm did when the dataset was made.
                     NOTE the recording predates the controller changes since
                     (cross_coupling_compensation was on); label it as such.
  --source arm       Re-run the goals on the arm now, EE_POS, and measure.
                     Compares sim against today's controller. Homes to the
                     episode's own start configuration first; a home that does
                     not converge is retried once, then the run stops with the
                     episodes already done on disk.

  python scripts/real_trajectory_rollout.py --repo-id HuskyMango/basket-7-24 --episode 3
  python scripts/real_trajectory_rollout.py --repo-id HuskyMango/basket-7-24 --episode 3 --source arm
  python scripts/real_trajectory_rollout.py --repo-id HuskyMango/sysid-8-28 --episode 4 --source arm --gain-amp 0.3
  python scripts/real_trajectory_rollout.py --repo-id HuskyMango/basket-7-24 --all --source arm --no-viz

The kp/kd action columns ride along as `replay.gain_action`, and the sim replay
follows them under variable impedance. `--gain-amp` (arm source only) adds a
quadrature kp/kd oscillation on top of the recording's own gain columns, which
turns any dataset episode into a task-shaped gain-excitation run.

Output: ~/franka_data/real_traj/<repo>/<ep<NNN>|all>_<source>_<ts>/episodes.hdf5
(+ one HTML per episode unless --no-viz). Under --all the file is rewritten
after every episode, so a fault mid-run keeps the episodes already done.

The arm source records `obs_timing: post_period` (row t is the state after
action t); the fit's loader re-rows that as `pre_action` itself.
"""

import argparse
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
from lerobot_robot_bimanual_franka import gain_schedule as gs  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.reach_record import (  # noqa: E402
	POLICY_ACTION_LEROBOT, EpisodeRecorder,
)
from real_reach_rollout import flush, run_metadata, write_viz  # noqa: E402
from replay_dataset import NUM_JOINTS, episode_goals, episode_slice  # noqa: E402

KEY = "r"      # both single-arm profiles expose their arm under r_
GAIN = slice(8, 10)   # kp, kd: the last two action columns


def open_dataset(repo_id: str, root: str | None) -> LeRobotDataset:
	ds = LeRobotDataset(repo_id, root=root)
	if int(ds.meta.fps) != int(fc.control_fps()):
		raise SystemExit(f"dataset fps {ds.meta.fps} != control fps {fc.control_fps()}")
	return ds


def load_episode(ds: LeRobotDataset, episode: int) -> dict:
	n_eps = int(ds.meta.total_episodes)
	if not 0 <= episode < n_eps:
		raise SystemExit(f"{ds.repo_id} has {n_eps} episodes (0..{n_eps - 1}); no episode {episode}")
	a, b = episode_slice(ds.meta, episode)
	actions = np.asarray(ds.hf_dataset["action"][a:b], dtype=np.float64)
	states = np.asarray(ds.hf_dataset["observation.state"][a:b], dtype=np.float64)
	if actions.shape[1] != 10 or states.shape[1] != NUM_JOINTS + 1:
		raise SystemExit(f"unexpected feature shapes action{actions.shape} state{states.shape}")
	return {"actions": actions, "states": states, "fps": int(ds.meta.fps),
			"n": int(b - a)}


class HomingFailed(RuntimeError):
	pass


def record_from_dataset(rec: EpisodeRecorder, ep: dict, goals: np.ndarray,
						episode: int, space: str) -> tuple:
	"""The recording as the real trajectory. No hardware."""
	q = ep["states"][:, :NUM_JOINTS]
	pos, quat = eef_poses_from_qpos(q)
	# Row t is the recorded state t, i.e. BEFORE action t (the LeRobot / fit
	# convention), not the post-period read the arm source makes.
	rec.begin(episode, q[0], quat[0], ee_pos0=pos[0],
			  extra={"control_mode": space, "eef_source": "franka_fk",
					 "obs_timing": "pre_action"})
	for t in range(ep["n"]):
		# The conversion anchored each goal on state t, so state t is also the
		# pose the policy acted on: the anchor gap is zero by construction.
		rec.record(pos[t], pos[t], quat[t], q[t], goals[t, :3], goals[t, 3:7],
				   pos[t], ep["actions"][t], gain_action=ep["actions"][t, GAIN])
	return rec.finish(ep["n"])


def record_from_arm(rec: EpisodeRecorder, ep: dict, goals: np.ndarray,
					episode: int, robot, dry_run: bool) -> tuple:
	"""Re-run the goal sequence on the arm, EE_POS, and measure. `goals` rows
	carry the gain columns that will be sent, schedule already applied."""
	q0 = ep["states"][0, :NUM_JOINTS]
	grip0 = float(ep["states"][0, NUM_JOINTS])
	# home() gives up after its time budget, which a long move from where the
	# last episode ended can outrun; a second attempt from nearer is cheap. Not
	# converging twice means the arm is stuck, and goals from a wrong start are
	# not the episode.
	if not robot.home(home_q_left=None, home_q_right=q0, gripper_norm=grip0):
		print("  homing did not converge; retrying once")
		if not robot.home(home_q_left=None, home_q_right=q0, gripper_norm=grip0):
			raise HomingFailed(f"episode {episode}: homing to the start configuration "
							   f"did not converge twice")
	snap = robot.robot_manager.current_kinematic_state_batch([KEY])[KEY]
	rec.begin(episode, np.asarray(snap[0]), np.asarray(snap[4]), ee_pos0=np.asarray(snap[3]),
			  qvel0=np.asarray(snap[1]))
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
		rec.record(seen, snap[3], snap[4], snap[0], gp, gq, ap, goals[t],
				   gain_action=goals[t, GAIN], qvel=snap[1])
	return rec.finish(ep["n"])


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--repo-id", required=True)
	which = ap.add_mutually_exclusive_group(required=True)
	which.add_argument("--episode", type=int, help="one episode index")
	which.add_argument("--all", action="store_true",
					   help="every episode of the dataset, into one episodes.hdf5")
	ap.add_argument("--root", default=None, help="local dataset root (default: the HF cache)")
	ap.add_argument("--source", choices=("dataset", "arm"), default="dataset")
	ap.add_argument("--arm", default="left", choices=("left", "right"),
					help="physical arm; the key prefix stays r_ either way")
	ap.add_argument("--out", default=str(Path.home() / "franka_data" / "real_traj"))
	ap.add_argument("--dry-run", action="store_true", help="--source arm: home and convert only")
	ap.add_argument("--gain-amp", type=float, default=0.0,
					help="--source arm: amplitude of a quadrature kp/kd gain-action oscillation "
						 "added to the recording's own gain columns (0.3 -> kp 75..300)")
	ap.add_argument("--gain-freq", type=float, default=gs.DEFAULT_FREQ_HZ)
	ap.add_argument("--gain-ramp-s", type=float, default=gs.DEFAULT_RAMP_S)
	ap.add_argument("--no-viz", action="store_true")
	ap.add_argument("--viz-stride", type=int, default=2)
	args = ap.parse_args()

	if args.gain_amp != 0.0 and args.source != "arm":
		raise SystemExit("--gain-amp changes what is SENT; it needs --source arm")

	ds = open_dataset(args.repo_id, args.root)
	episodes = list(range(int(ds.meta.total_episodes))) if args.all else [args.episode]
	cls = SingleArmFrankaConfig if args.arm == "left" else SingleArmRightConfig
	cfg = cls(control_mode=ControlMode.EE_POS)
	schedule = None
	if args.gain_amp != 0.0:
		schedule = gs.describe(args.gain_amp, args.gain_amp, args.gain_freq, args.gain_ramp_s,
							   gs.DEFAULT_PHASE_KD_RAD)

	def prepare(episode: int) -> tuple[dict, np.ndarray, str]:
		"""The episode's frames, the absolute goals they command (gains applied)
		and the space the recording is in."""
		ep = load_episode(ds, episode)
		goals, space = episode_goals(ep["actions"], ep["states"], cfg)
		if space == "EE_DELTA":
			extent = f"|delta| p99 {np.percentile(np.abs(ep['actions'][:, :3]), 99):.4f} m"
		else:
			extent = f"goal span {np.round(np.ptp(goals[:, :3], axis=0), 3)} m"
		print(f"{args.repo_id} episode {episode}: {ep['n']} steps, {space}, {extent}, "
			  f"start q {np.round(ep['states'][0, :NUM_JOINTS], 3)}")
		if np.any(ep["actions"][:, GAIN] != ep["actions"][0, GAIN]):
			print(f"  recording moves its gains: a_kp {ep['actions'][:, 8].min():+.3f}..{ep['actions'][:, 8].max():+.3f}, "
				  f"a_kd {ep['actions'][:, 9].min():+.3f}..{ep['actions'][:, 9].max():+.3f}")
		if args.gain_amp != 0.0:
			t_s = np.arange(ep["n"]) / float(fc.control_fps())
			# The recording's own gain columns are the centre, per step.
			osc = gs.quadrature_schedule(t_s, args.gain_amp, args.gain_amp, args.gain_freq,
										 args.gain_ramp_s)
			goals[:, GAIN] = np.clip(goals[:, GAIN] + osc, -1.0, 1.0)
			print(f"  gain excitation: a_kp {goals[:, 8].min():+.2f}..{goals[:, 8].max():+.2f}, "
				  f"a_kd {goals[:, 9].min():+.2f}..{goals[:, 9].max():+.2f} at {args.gain_freq:g} Hz")
		return ep, goals, space

	which = "all" if args.all else f"ep{args.episode:03d}"
	run_dir = (Path(args.out) / args.repo_id.replace("/", "__")
			   / f"{which}_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
	run_dir.mkdir(parents=True, exist_ok=True)
	meta = run_metadata(args.arm, seed=-1)
	meta.update({"repo_id": args.repo_id, "episodes": episodes, "source": args.source,
				 "gain_schedule": schedule})
	producer = "scripts/real_trajectory_rollout.py"

	rec = EpisodeRecorder(args.arm, source=args.source, policy_action_format=POLICY_ACTION_LEROBOT)
	results = []
	stopped = None
	if args.source == "dataset":
		for episode in episodes:
			ep, goals, space = prepare(episode)
			results.append(record_from_dataset(rec, ep, goals, episode, space))
	else:
		from lerobot.robots.utils import make_robot_from_config
		robot = make_robot_from_config(cfg)
		robot.connect()
		try:
			# Written before the arm moves, and again after every episode, so a
			# fault keeps the episodes already done and what the arm was running.
			flush(run_dir, results, meta, producer)
			for episode in episodes:
				ep, goals, _ = prepare(episode)
				try:
					results.append(record_from_arm(rec, ep, goals, episode, robot, args.dry_run))
				except HomingFailed as e:
					stopped = str(e)
					break
				flush(run_dir, results, meta, producer)
		finally:
			robot.disconnect()
	print(f"{len(results)} episode(s) -> {flush(run_dir, results, meta, producer)}")
	if stopped:
		print(f"STOPPED: {stopped}; {len(episodes) - len(results)} episode(s) not recorded")
	if not args.no_viz:
		write_viz(run_dir, args.viz_stride)
	return 1 if stopped else 0


if __name__ == "__main__":
	raise SystemExit(main())
