"""The reach task on a real FR3.

Stands in for multi-fast's `Reach(SingleArmEnv)` so `ReachObservationWrapper`
and `ReachBaseWrapper` run against hardware unchanged. Everything the sim env
does with mujoco -- scene, reward shaping, the state-snapshot reference rollout
-- is gone; what remains is the task itself: sample a curve, advance a cursor
along it, decide success.

No reward. RL does not run on hardware, so `traj_deviation_lambda` and the
per-episode DTW reference (which needed sim state snapshots) are dropped rather
than reimplemented. `step` returns 0.0 and the success flag.

Units at the boundary, which is where this file earns its keep:
  * ReachBaseWrapper emits NORMALISED OSC units -- translation in multiples of
    osc_output_max (0.05 m), rotation as an axis-angle in multiples of
    osc_rot_output_max (0.5 rad).
  * BimanualFranka.send_action EE_DELTA takes METRES and a delta QUATERNION.
Converting between them is `_action_to_delta`. Getting it wrong is silent: a
normalised 1.0 passed straight through reads as 1 metre, gets clipped to
torque.delta.pos_max_m, and looks like a tracking problem.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

import franka_config as fc

from .osc_torque_controller import resolve_gains
from .real_reach_geometry import S, _home_key, base_to_world, sample_episode

MAX_WAYPOINTS = S.MAX_WAYPOINTS


class _ControllerView:
	"""The `.controller.kp/.kd` ReachObservationWrapper reads to build
	`controller_state`. Holds the gains that were actually commanded, so the
	obs never disagrees with what the arm is running."""

	def __init__(self, kp: np.ndarray, kd: np.ndarray):
		self.kp = np.asarray(kp, dtype=np.float64)
		self.kd = np.asarray(kd, dtype=np.float64)


class _RecentEEVelView:
	def __init__(self):
		self.current = np.zeros(6, dtype=np.float32)


class _RobotView:
	"""Presents RealReach in the shape ReachObservationWrapper expects.

	The wrapper was written against robosuite and reaches through
	`env.robots[0]` for gains and EE velocity. Rather than fork it -- which
	would put the gain remap and obs assembly in two copies that drift -- this
	forwards those names to real measured state. Pure forwarding by design: it
	computes nothing, so it can never report a number the arm is not running.
	"""

	def __init__(self):
		self.controller = _ControllerView(np.full(6, 150.0), np.full(6, 24.494897))
		self.recent_ee_vel = _RecentEEVelView()


class RealReach:
	"""One arm, one curve, EE_DELTA. Gym-ish: reset() -> obs, step() -> 4-tuple."""

	def __init__(self, robot, arm: str = "right", arm_key: str = "r", seed: int | None = None,
			  realtime: bool = True):
		self.robot = robot
		# step() returns the state AFTER the action has had a control period to act,
		# as Reach.step does in sim. On hardware that means waiting the period out
		# before reading; a fake arm responds instantly and skips the wait.
		self._period = 1.0 / float(fc.control_fps()) if realtime else 0.0
		self._t_sent = None
		self.arm = arm
		self.k = arm_key
		self.rng = np.random.default_rng(seed)

		cfg = fc.section("reach")
		self.n_waypoints = int(cfg["curve"]["n_waypoints"])
		self.success_threshold = float(cfg["task"]["success_threshold_m"])
		self.advance_threshold = float(cfg["task"]["advance_threshold_m"])
		self.max_episode_steps = int(cfg["task"]["max_episode_steps"])
		self.gripper_norm = float(cfg["task"]["gripper_norm"])
		self.include_orient = float(cfg["orientation"]["delta_max_deg"]) > 0.0

		self.osc_output_max = float(cfg["base_policy"]["osc_output_max"])
		self.osc_rot_output_max = float(cfg["base_policy"]["osc_rot_output_max"])
		self._assert_delta_envelope(float(cfg["base_policy"]["max_step"]))

		self.robots = [_RobotView()]
		self._goal = None
		self._waypoints = None
		self._waypoint_quats = None
		self._velocity_scales = None
		self._next_waypoint_idx = 0
		self._curve_len = 0
		self.timestep = 0
		self._last_q = None
		self._last_pos = None
		# The post-home measured q. A sim replay starts from this, and it is NOT
		# qpos[0] of the recorded trace -- that one is already a step old.
		self._reset_q = None
		self._reset_quat = None

	# ---------------------------------------------------------------- guards

	def _assert_delta_envelope(self, max_step: float) -> None:
		"""A base action of `max_step` must fit inside torque.delta, or
		`clip_delta` truncates it and the arm silently under-tracks the curve
		with nothing in the log to say so."""
		pos = max_step * self.osc_output_max
		rot = max_step * self.osc_rot_output_max
		pos_max = float(fc.control("torque.delta.pos_max_m"))
		rot_max = float(fc.control("torque.delta.rot_max_rad"))
		if pos > pos_max + 1e-12 or rot > rot_max + 1e-12:
			raise ValueError(
				f"base policy can command {pos:.4f} m / {rot:.4f} rad per step, "
				f"outside torque.delta ({pos_max} m / {rot_max} rad). Lower "
				f"max_step or osc_output_max -- do NOT raise torque.delta, which "
				f"is the sim's own envelope."
			)

	# ------------------------------------------------------------- ee access

	def _ee(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""(pos, quat_xyzw, twist) in the arm's BASE frame, from one read."""
		# The driver registry is keyed by the exposed key prefix, not the physical
		# arm name -- self.arm is for config lookups only.
		snap = self.robot.robot_manager.current_kinematic_state_batch([self.k])[self.k]
		q, _, _, pos, quat_xyzw, twist = snap
		# Kept for `step`'s info: q is not in the obs (the sim's obs has no joint
		# key) and pos is the anchor the next commanded goal is built on.
		self._last_q = np.asarray(q, dtype=np.float64)
		self._last_pos = np.asarray(pos, dtype=np.float64)
		return (self._last_pos.copy(),
				np.asarray(quat_xyzw, dtype=np.float64),
				np.asarray(twist, dtype=np.float64))

	# ------------------------------------------------------------------ gym

	def reset(self) -> dict:
		# home() takes one q per key prefix; the q itself is the PHYSICAL arm's.
		q = {"l": None, "r": None}
		q[self.k] = fc.home_q(key=_home_key(self.arm))
		self.robot.home(home_q_left=q["l"], home_q_right=q["r"])
		self._t_sent = None
		pos, quat, _ = self._ee()
		self._reset_q = self._last_q.copy()
		self._reset_quat = quat.copy()

		ep = sample_episode(self.arm, pos, self.rng)
		self._goal = np.asarray(ep["goal"], dtype=np.float32)
		self._waypoints = np.asarray(ep["waypoints"], dtype=np.float32)
		self._velocity_scales = np.asarray(ep["velocity_scales"], dtype=np.float32)
		self._waypoint_quats = ep["waypoint_quats"]
		if self._waypoint_quats is not None:
			# Sampled about identity; re-anchor on the pose the arm actually
			# homed to, so the curve is a delta from here rather than an
			# absolute orientation the arm would snap to on step one.
			r0 = Rotation.from_quat(quat)
			self._waypoint_quats = np.stack([
				(Rotation.from_quat(q) * r0).as_quat() for q in self._waypoint_quats
			]).astype(np.float32)

		self._curve_len = len(self._waypoints)
		self._next_waypoint_idx = 0
		self.timestep = 0
		self.robots[0].recent_ee_vel.current = np.zeros(6, dtype=np.float32)
		return self._observation(pos, quat)

	def step(self, action) -> tuple[dict, float, bool, dict]:
		dpos, dquat = self._action_to_delta(np.asarray(action, dtype=np.float64))

		cmd = {f"{self.k}_{ax}": float(v) for ax, v in zip("xyz", dpos)}
		cmd.update({f"{self.k}_{ax}": float(v)
					for ax, v in zip(("qx", "qy", "qz", "qw"), dquat)})
		# The base policy emits 0 in the gripper slot meaning "no manipulation",
		# but send_action reads {arm}_gripper as an ABSOLUTE normalised position
		# -- passing that 0 through would drive the gripper shut. Hold it open.
		cmd[f"{self.k}_gripper"] = self.gripper_norm
		cmd["kp"] = 0.0        # normalised: 0 -> default_kp via the exp remap
		cmd["kd"] = 0.0
		self.robot.send_action(cmd)
		# Read at the END of the period, not right after the send: read immediately,
		# the arm has not responded yet and every observation is one step stale --
		# which put real one step behind sim in the first trajectory diff.
		if self._period > 0.0:
			now = time.perf_counter()
			deadline = (self._t_sent or now) + self._period
			if deadline > now:
				time.sleep(deadline - now)
			self._t_sent = max(deadline, now)
		self.timestep += 1
		pos, quat, twist = self._ee()
		self._advance_cursor(pos)
		self.robots[0].recent_ee_vel.current = twist.astype(np.float32)

		success = self._check_success(pos)
		done = bool(success or self.timestep >= self.max_episode_steps)
		osc_pos, osc_quat = self._dispatched_goal()
		anchor_pos, _ = self._dispatched_anchor()
		info = {"success": success, "episode_steps": self.timestep,
				"next_waypoint_idx": int(self._next_waypoint_idx),
				# BASE frame. The goal that was DISPATCHED -- clip_delta, the tuning
				# fudges and the worktable screen already applied.
				"osc_goal_pos": osc_pos, "osc_goal_quat": osc_quat,
				# The pose send_action anchored on. Its gap from the pose this side last
				# observed IS the stale-anchor effect; reported so it is measurable
				# rather than folded into the goal.
				"osc_anchor_pos": anchor_pos,
				"action": np.asarray(action, dtype=np.float64).copy(),
				"qpos": self._last_q.copy()}
		return self._observation(pos, quat), 0.0, done, info

	# ------------------------------------------------------------ internals

	def _action_to_delta(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Normalised OSC 7-vector -> (metres, delta quat xyzw)."""
		dpos = a[0:3] * self.osc_output_max
		rotvec = a[3:6] * self.osc_rot_output_max
		return dpos, Rotation.from_rotvec(rotvec).as_quat()

	def _dispatched_goal(self) -> tuple[np.ndarray | None, np.ndarray | None]:
		"""The OSC goal send_action actually dispatched, base frame.

		`commanded_goal` is this side's estimate; it differs by the stale-anchor gap
		(send_action re-reads the pose), by `clip_delta` and the `tuning` fudges, and
		by the worktable screen. A sim replay has to reproduce the dispatched pose,
		not the estimate. None when the robot does not expose it (the offline fakes).
		"""
		goal = getattr(self.robot, "_last_osc_goal", {}).get(self.k)
		if goal is None:
			return None, None
		pos, quat = goal
		return (np.asarray(pos, dtype=np.float64).copy(),
				np.asarray(quat, dtype=np.float64).copy())

	def _dispatched_anchor(self) -> tuple[np.ndarray | None, np.ndarray | None]:
		"""The measured pose send_action composed its goal on, base frame."""
		snap = getattr(self.robot, "_last_osc_anchor", {}).get(self.k)
		if snap is None:
			return None, None
		pos, quat = snap
		return (np.asarray(pos, dtype=np.float64).copy(),
				np.asarray(quat, dtype=np.float64).copy())

	def _advance_cursor(self, eef_pos: np.ndarray) -> None:
		"""Reach._post_action's rule, verbatim: advance while the EE is within
		advance_threshold of the current waypoint OR is already nearer the next
		one (which catches corner-cutting at kinks, where the EE flies past a
		waypoint without ever entering its threshold). The final waypoint has no
		successor, so it is proximity-only."""
		while self._next_waypoint_idx < self._curve_len - 1:
			i = self._next_waypoint_idx
			cur_d = float(np.linalg.norm(eef_pos - self._waypoints[i]))
			next_d = float(np.linalg.norm(eef_pos - self._waypoints[i + 1]))
			if cur_d < self.advance_threshold or next_d < cur_d:
				self._next_waypoint_idx += 1
			else:
				break
		if (self._next_waypoint_idx == self._curve_len - 1
				and float(np.linalg.norm(
					eef_pos - self._waypoints[self._next_waypoint_idx])) < self.advance_threshold):
			self._next_waypoint_idx += 1

	def _check_success(self, eef_pos: np.ndarray) -> bool:
		if self._goal is None:
			return False
		at_goal = float(np.linalg.norm(eef_pos - self._goal)) < self.success_threshold
		return bool(at_goal and self._next_waypoint_idx >= self._curve_len)

	def _observation(self, pos: np.ndarray, quat: np.ndarray) -> dict:
		return {
			"robot0_eef_pos": pos.astype(np.float32),
			"robot0_eef_quat": quat.astype(np.float32),
			"robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
			"waypoints": self.waypoints,
			"next_waypoint_idx": np.array([self._next_waypoint_idx], dtype=np.float32),
			"velocity_scales": self.velocity_scales,
			"goal": self.goal,
		}

	# ---- accessors ReachObservationWrapper / ReachBaseWrapper read ----

	@property
	def reset_q(self):
		"""The post-home measured joint configuration this episode started from."""
		return None if self._reset_q is None else self._reset_q.copy()

	@property
	def reset_quat(self):
		"""O_T_EE orientation at reset, base frame xyzw. With `qpos0` it lets a sim
		replay measure how its own EE frame sits relative to the real one."""
		return None if self._reset_quat is None else self._reset_quat.copy()

	@property
	def goal(self):
		return None if self._goal is None else self._goal.copy()

	@property
	def waypoints(self):
		"""Padded to MAX_WAYPOINTS with the GOAL, not with zeros or the last
		point: ReachBaseWrapper indexes waypoints[idx + lookahead_k] without
		bounds-checking, and filling with the goal makes that clamp correctly."""
		if self._waypoints is None:
			return np.zeros((MAX_WAYPOINTS, 3), dtype=np.float32)
		out = np.repeat(self._goal[None, :], MAX_WAYPOINTS, axis=0)
		out[:self._curve_len] = self._waypoints
		return out

	@property
	def waypoint_quats(self):
		if self._waypoint_quats is None:
			return None
		out = np.repeat(self._waypoint_quats[-1][None, :], MAX_WAYPOINTS, axis=0)
		out[:self._curve_len] = self._waypoint_quats
		return out

	@property
	def velocity_scales(self):
		if self._velocity_scales is None:
			return np.zeros(MAX_WAYPOINTS, dtype=np.float32)
		out = np.zeros(MAX_WAYPOINTS, dtype=np.float32)
		out[:len(self._velocity_scales)] = self._velocity_scales
		return out

	@property
	def next_waypoint_idx(self):
		return int(self._next_waypoint_idx)

	def waypoints_world(self):
		"""The curve in world frame -- for logging and the safety cross-check."""
		return base_to_world(self.arm, self._waypoints)
