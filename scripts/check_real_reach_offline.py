#!/usr/bin/env python3
"""Drive RealReach with the REAL analytic base policy against a fake arm.

No hardware. The fake arm models the EE as a first-order response to the OSC
goal -- the same approximation ReachBaseWrapper's chunk_alpha encodes -- which
is enough to exercise everything that does not need physics: unit conversion at
the send_action boundary, cursor advance, success detection, the padding rule,
and the robot shim ReachObservationWrapper reads.

What it cannot tell you: whether the arm tracks. That is what phase 2 on
hardware is for.

  python scripts/check_real_reach_offline.py -n 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "multi-fast"))

import franka_config as fc  # noqa: E402
from lerobot_robot_bimanual_franka.ee_goals import OSCGoalBuilder  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.safety import ActionSafetyScreen  # noqa: E402
from lerobot_robot_bimanual_franka.reach_record import ReachEpisodeRecorder  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach import RealReach  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import (  # noqa: E402
	_home_key, base_to_world, keep_out_sphere, safety_z_floor_world,
)
from real_reach_rollout import flush, run_metadata  # noqa: E402
from utils.base_policy_utils import ReachBaseWrapper  # noqa: E402
from utils.sysid import episode_hdf5  # noqa: E402


class FakeArm:
	"""Stands in for BimanualFranka. Applies the commanded delta as a
	first-order step toward the goal, and records every action so the harness
	can assert on what would actually have gone to the wire.

	The goal is composed through the REAL `OSCGoalBuilder` and the REAL
	`ActionSafetyScreen`, so the clip, the tuning fudges, the latched `goal_ori`
	and the worktable floor are all exercised here rather than only on hardware
	-- and `_last_osc_goal` means what it means on the real robot."""

	ALPHA = 0.3            # matches base_policy.chunk_alpha

	def __init__(self, arm="right", key="r"):
		self.arm, self.k = arm, key
		self.robot_manager = self
		self.sent = []
		self._goals = OSCGoalBuilder(
			translation_fudge=float(fc.control("tuning.ee_translation_fudge")),
			rotation_fudge=float(fc.control("tuning.ee_rotation_fudge")),
			use_noise=False, noise_pos_scale=0.0, noise_rot_scale=0.0,
		)
		self.safety = ActionSafetyScreen({key: fc.robot_base_in_world(arm)},
										 {key: fc.ee_sphere(arm)})
		self._last_osc_goal = {}
		self._last_osc_anchor = {}
		# Pre-screen, post-fudge: what the screen was ASKED for. The dispatched
		# goal can no longer show a floor breach, because the screen removed it.
		self._last_raw_goal = None
		self._home()

	def _home(self):
		q = np.asarray(fc.home_q(key=_home_key(self.arm)))[None, :]
		pos, quat = eef_poses_from_qpos(q)
		# Reported as the measured q. Zeros here would make the recorded qpos0
		# a pose no arm was ever in, and a sim replay would start from it.
		self.q = q[0].astype(np.float64)
		self.pos = pos[0].astype(np.float64)
		self.quat = quat[0].astype(np.float64)
		self.twist = np.zeros(6)

	def home(self, home_q_left=None, home_q_right=None, **kw):
		# Mirror the real signature: a bare home() must fail here too.
		if {"l": home_q_left, "r": home_q_right}[self.k] is None:
			raise TypeError(f"home() got no q for key {self.k!r}")
		self._home()

	@property
	def last_ee_wrench(self):
		# |F| equals the number of goals sent so far.
		return {self.k: np.array([0.0, 0.0, float(len(self.sent)), 0.0, 0.0, 0.0])}

	def current_kinematic_state_batch(self, arms):
		# Registered under the key prefix; echoing back any key would hide a
		# wrong-key lookup, which is exactly what it did.
		bad = [a for a in arms if a != self.k]
		if bad:
			raise KeyError(f"no driver for {bad}; registered: [{self.k!r}]")
		return {a: (self.q, np.zeros(7), None, self.pos, self.quat, self.twist)
				for a in arms}

	def send_action(self, cmd, ignore_action=False):
		self.sent.append(dict(cmd))
		dpos = np.array([cmd[f"{self.k}_{c}"] for c in "xyz"])
		dq = np.array([cmd[f"{self.k}_{c}"] for c in ("qx", "qy", "qz", "qw")])
		drot = Rotation.from_quat(dq).as_rotvec()
		# clip_delta lives inside from_delta; clipping again here would skip the
		# fudge and define the envelope in two places.
		self._last_osc_anchor = {self.k: (self.pos.copy(), self.quat.copy())}
		raw = self._goals.from_delta(self.k, dpos, drot, self.pos, self.quat)
		self._last_raw_goal = np.asarray(raw[0], dtype=np.float64).copy()
		goals = self.safety.shape_goal({self.k: raw})
		self._last_osc_goal = {a: (np.asarray(p, dtype=np.float64),
								   np.asarray(q, dtype=np.float64))
							   for a, (p, q) in goals.items()}
		goal_pos, goal_quat = self._last_osc_goal[self.k]
		prev = self.pos.copy()
		# First-order toward the SCREENED goal, so a goal the floor moved is what
		# the fake arm actually chases.
		self.pos = self.pos + self.ALPHA * (goal_pos - self.pos)
		self.quat = (Rotation.from_rotvec(self.ALPHA * (
			Rotation.from_quat(goal_quat) * Rotation.from_quat(self.quat).inv()
		).as_rotvec()) * Rotation.from_quat(self.quat)).as_quat()
		self.twist = np.concatenate([(self.pos - prev) * fc.control_fps(), np.zeros(3)])
		return cmd


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--episodes", type=int, default=20)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--arm", default="left", choices=("left", "right"),
					help="physical arm to sample for; the key prefix stays r_ either way")
	ap.add_argument("--out", default=None,
						help="write an episodes.hdf5 here, in the hardware layout, so the "
							 "sim replay can be exercised with no arm")
	args = ap.parse_args()

	cfg = fc.section("reach")
	bp = cfg["base_policy"]
	arm = FakeArm(arm=args.arm, key="r")
	env = RealReach(arm, arm=args.arm, arm_key="r", seed=args.seed, realtime=False)
	print(f"envelope assert passed: {bp['max_step'] * bp['osc_output_max']} m "
		  f"<= {fc.control('torque.delta.pos_max_m')} m")

	base = ReachBaseWrapper(
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
		include_orient=False,
		goal_mode=str(bp["goal_mode"]),
		control_freq=float(fc.control_fps()),
	)

	n_success = 0
	floor_clear = []
	keep_clear = []
	floor = safety_z_floor_world(args.arm)
	keep_out = keep_out_sphere(args.arm)
	steps_used, cursor_end, clipped = [], [], 0
	recorder = ReachEpisodeRecorder(args.arm, args.seed)
	results = []
	pmax = float(fc.control("torque.delta.pos_max_m"))

	for ep in range(args.episodes):
		obs = env.reset()
		recorder.begin_reach(env, ep)
		assert obs["waypoints"].shape == (128, 3), obs["waypoints"].shape
		done = False
		while not done:
			batched = {k: (v[None, ...] if isinstance(v, np.ndarray) else v)
					   for k, v in obs.items() if v is not None}
			chunk = np.asarray(base(batched))          # (1, chunk_size, 7)
			for a in chunk[0]:
				seen = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
				obs, _, done, info = env.step(a)
				recorder.record_reach(seen, obs, info)
				sent = arm.sent[-1]
				d = np.array([sent[f"r_{c}"] for c in "xyz"])
				if np.any(np.abs(d) > pmax + 1e-9):
					clipped += 1
				# The goal send_action composes is ee_pos + dpos. Phase 1 bounds
				# the CURVE; this checks what the base policy actually commands,
				# which can overshoot it. If any goal dips under the floor the
				# safety screen fires and silently rescales -- the one thing that
				# would make the executed trace stop matching the curve.
				goal_w = base_to_world(args.arm, arm._last_raw_goal)
				floor_clear.append(float(goal_w[2] - floor))
				# Same argument for the other arm: the curve is bounded, the
				# commanded goal is not, and nothing repels the two arms.
				if keep_out is not None:
					keep_clear.append(
						float(np.linalg.norm(goal_w - keep_out[0]) - keep_out[1]))
				assert np.isclose(sent["r_gripper"], cfg["task"]["gripper_norm"])
				if done:
					break
		results.append(recorder.finish_reach(info))
		n_success += bool(info["success"])
		steps_used.append(info["episode_steps"])
		cursor_end.append(info["next_waypoint_idx"])

	print(f"\narm                 : {args.arm}  (key prefix r_)")
	print(f"episodes            : {args.episodes}")
	print(f"success             : {n_success}/{args.episodes} ({n_success/args.episodes:.0%})")
	print(f"median steps         : {int(np.median(steps_used))}")
	print(f"cursor reached end   : {sum(c >= env._curve_len for c in cursor_end)}/{args.episodes}")
	print(f"deltas over envelope : {clipped}  (must be 0)")
	print(f"min goal clearance   : {min(floor_clear):+.4f} m above safety floor "
	      f"({sum(c < 0 for c in floor_clear)} would trip the screen)")
	# Every step carries the wrench read with its pose, and the layout validator
	# accepts the episode with it.
	force_ok = all("ee_force" in a and a["ee_force"].shape == (at["num_samples"], 3)
				   and np.all(np.diff(a["ee_force"][:, 2]) == 1)
				   and not episode_hdf5.validate_episode(n, a, at, c)
				   for n, a, at, c in results)
	print(f"EE force recorded    : {'every step, validator clean' if force_ok else 'MISSING or invalid'}")
	keep_ok = True
	if keep_out is not None:
		n_keep_bad = sum(c < 0 for c in keep_clear)
		keep_ok = n_keep_bad == 0
		print(f"min keep-out margin  : {min(keep_clear):+.4f} m outside the "
		      f"{keep_out[1]:.3f} m sphere ({n_keep_bad} breaches)")

	if args.out:
		out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
		meta = {**run_metadata(args.arm, args.seed), "fake_arm": True}
		path = flush(out, results, meta, "scripts/check_real_reach_offline.py")
		print(f"\nwrote {len(results)} episodes -> {path}")

	ok = (clipped == 0 and n_success == args.episodes and min(floor_clear) >= 0
	      and keep_ok and force_ok)
	print("\nPASS" if ok else "\nFAIL")
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
