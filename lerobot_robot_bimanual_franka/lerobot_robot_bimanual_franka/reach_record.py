"""One episode's on-disk record, for a reach curve or a dataset trajectory.

Shared by `scripts/real_reach_rollout.py`, `scripts/real_trajectory_rollout.py`
and `scripts/check_real_reach_offline.py`, so every producer emits the same
episode and the sim replay and the figure consume one layout: the episode HDF5
described in EPISODE_HDF5.md (multi-fast/utils/sysid/episode_hdf5.py).

Everything is recorded in the arm's BASE frame, O_T_EE convention, xyzw
quaternions -- the frame the goals were produced in. The figure maps base to
world when it draws; nothing here inverts `robot_base_in_world`.

`eef_goal_*` (and `action`, the same thing in the fit's column order) is the
DISPATCHED goal, after `clip_delta`, the `tuning` fudges, the latched `goal_ori`
and the worktable screen -- not an estimate of it. Under a zero rotation delta
`from_delta` never re-latches, so `eef_goal_quat` is CONSTANT across a
position-only episode. That is the real controller's behaviour and a replay has
to reproduce it, which is why the quaternion is recorded per step rather than
assumed.

The reach curve is optional. Without it the record is a plain trajectory and
the task-specific fields (cursor, success) are absent rather than faked.

`gain_action` (T,2) is the normalised kp/kd action each step sent, when the
producer passes it, and `kp`/`kd` (T,6) the physical gains `resolve_gains`
made of it; the rig's action -> gain map (base kp, exp base, limits, `tuning`
trims) is stamped as attrs so the sim replay can check its own map against it
before switching to variable impedance.
"""

from __future__ import annotations

import numpy as np

import franka_config as fc

from .gain_schedule import remap_constants
from .osc_torque_controller import resolve_gains
from .real_reach_geometry import base_to_world

POLICY_ACTION_REACH = "osc_delta_norm7"     # [dpos_norm(3), drot_norm(3), gripper]
POLICY_ACTION_LEROBOT = "lerobot_row10"     # [x y z qx qy qz qw gripper kp kd]


def episode_name(episode: int) -> str:
	return f"ep{int(episode):03d}"


class EpisodeRecorder:
	"""Accumulates one episode, then emits (name, arrays, attrs, curve)."""

	def __init__(self, arm: str, seed: int | None = None, source: str = "arm",
				 policy_action_format: str = POLICY_ACTION_REACH):
		self.arm = arm
		self.seed = None if seed is None else int(seed)
		self.source = source        # "arm" (measured) or "dataset" (as recorded)
		self.policy_action_format = policy_action_format
		self.fps = float(fc.control_fps())

	def begin(self, episode: int, qpos0: np.ndarray, ee_quat0: np.ndarray,
			  curve: dict | None = None, extra: dict | None = None,
			  ee_pos0: np.ndarray | None = None, qvel0: np.ndarray | None = None) -> None:
		"""`curve`, when present, is {goal (3), waypoints (N,3), velocity_scales
		(N)} in base frame. `extra` lands in the episode attrs verbatim. `qvel0`
		is the measured joint velocity at the start, when the producer has one."""
		self.episode = int(episode)
		self.eef_pos, self.eef_quat, self.qpos, self.qvel = [], [], [], []
		self.goal_pos, self.goal_quat, self.policy_action = [], [], []
		self.cursor, self.anchor_gap, self.gains, self.kp, self.kd = [], [], [], [], []
		self.ee_force, self.ee_torque = [], []
		conv_rotvec, conv_pos = fc.sim_ee_convention()
		self.curve = None if curve is None else {
			"goal": np.asarray(curve["goal"], dtype=np.float64),
			"waypoints": np.asarray(curve["waypoints"], dtype=np.float64),
			"velocity_scales": np.asarray(curve["velocity_scales"], dtype=np.float64)}
		remap = remap_constants()
		self._trims = remap["tuning_gain_scales"]
		self.attrs = {
			"episode": self.episode, "arm": self.arm, "seed": self.seed,
			"source": self.source, "fps": self.fps,
			"frame": "base", "quat_order": "xyzw", "ee_convention": "O_T_EE",
			# eef_pos[t] is the state after action t acted for one control period.
			"obs_timing": "post_period",
			"control_mode": "EE_DELTA" if curve is not None else "EE_POS",
			"action_format": "absolute_pose_quat", "action_space": "EE_POS",
			"policy_action_format": self.policy_action_format,
			"init_qpos": np.asarray(qpos0, dtype=np.float64),
			# O_T_EE at qpos0. A sim replay measures its EE frame against this at
			# the same joints, which is how it learns the rigid offset between
			# robosuite's grip site and the FR3's O_T_EE without a hardcoded constant.
			"ee_quat0": np.asarray(ee_quat0, dtype=np.float64),
			"ee_pos0": None if ee_pos0 is None else np.asarray(ee_pos0, dtype=np.float64),
			# With init_qpos / ee_pos0 / ee_quat0, the start row a reader needs to
			# re-row a post_period record as pre_action.
			"qvel0": None if qvel0 is None else np.asarray(qvel0, dtype=np.float64),
			# Describes the hand BODY for sim-trained policies; the OSC replay
			# measures its own frame and does not use this.
			"sim_ee_convention_rotvec": np.asarray(conv_rotvec, dtype=np.float64),
			"sim_ee_convention_pos": np.asarray(conv_pos, dtype=np.float64),
			# This rig's normalised gain action -> (kp, kd) map.
			"osc_base_kp": remap["osc_base_kp"],
			"osc_default_damping_ratio": remap["osc_default_damping_ratio"],
			"gain_exp_base": remap["gain_exp_base"],
			"kp_limits": remap["kp_limits"],
			"damping_ratio_limits": remap["damping_ratio_limits"],
			"tuning_gain_scales": remap["tuning_gain_scales"],
			**(extra or {}),
		}
		if curve is not None:
			self.attrs["curve_len"] = int(len(curve["waypoints"]))

	def record(self, seen_pos, ee_pos, ee_quat, qpos, osc_goal_pos, osc_goal_quat,
			   anchor_pos, action, cursor: int | None = None,
			   gain_action=None, qvel=None, ee_force=None, ee_torque=None) -> np.ndarray:
		"""One step, all base frame. `seen_pos` is the EE the policy acted on,
		`action` the policy's own action, `gain_action` the normalised [a_kp, a_kd]
		sent with it, `qvel` the measured joint velocity when the producer has one,
		`ee_force`/`ee_torque` libfranka's estimated external wrench at the EE.

		Returns the measured EE in world, which a live caller needs for its floor
		and keep-out warnings.
		"""
		self.eef_pos.append(np.asarray(ee_pos, dtype=np.float64))
		self.eef_quat.append(np.asarray(ee_quat, dtype=np.float64))
		self.qpos.append(np.asarray(qpos, dtype=np.float64))
		self.goal_pos.append(np.asarray(osc_goal_pos, dtype=np.float64))
		self.goal_quat.append(np.asarray(osc_goal_quat, dtype=np.float64))
		self.policy_action.append(np.asarray(action, dtype=np.float64))
		# How much fresher send_action's own read was than the pose the policy
		# acted on: the stale-anchor effect, as one number per step.
		self.anchor_gap.append(float(np.linalg.norm(
			np.asarray(anchor_pos, dtype=np.float64) - np.asarray(seen_pos, dtype=np.float64))))
		if cursor is not None:
			self.cursor.append(int(cursor))
		if gain_action is not None:
			a = np.asarray(gain_action, dtype=np.float64).reshape(2)
			self.gains.append(a)
			t = self._trims
			kp6, kd6 = resolve_gains(a[0], a[1], t["kp_ori_scale"], t["kd_ori_scale"],
									 kp_pos_scale=t["kp_pos_scale"], kd_pos_scale=t["kd_pos_scale"])
			self.kp.append(kp6)
			self.kd.append(kd6)
		if qvel is not None:
			self.qvel.append(np.asarray(qvel, dtype=np.float64))
		if ee_force is not None:
			self.ee_force.append(np.asarray(ee_force, dtype=np.float64))
			self.ee_torque.append(np.asarray(ee_torque, dtype=np.float64))
		return base_to_world(self.arm, np.asarray(ee_pos, dtype=np.float64))

	def finish(self, steps: int, success: bool | None = None,
			   cursor: int | None = None) -> tuple:
		n = len(self.eef_pos)
		qpos = np.asarray(self.qpos).reshape(n, 7)
		if self.qvel:
			qvel, qvel_source = np.asarray(self.qvel).reshape(n, 7), "measured"
		else:
			qvel = (np.gradient(qpos, 1.0 / self.fps, axis=0) if n > 1 else np.zeros((n, 7)))
			qvel_source = "central_difference"
		goal_pos = np.asarray(self.goal_pos).reshape(n, 3)
		goal_quat = np.asarray(self.goal_quat).reshape(n, 4)
		arrays = {
			"action": np.concatenate([goal_pos, goal_quat], axis=1),
			"eef_goal_pos": goal_pos, "eef_goal_quat": goal_quat,
			"eef_pos": np.asarray(self.eef_pos).reshape(n, 3),
			"eef_quat": np.asarray(self.eef_quat).reshape(n, 4),
			"qpos": qpos, "qvel": qvel,
			"policy_action": np.asarray(self.policy_action).reshape(n, -1),
			"anchor_gap_m": np.asarray(self.anchor_gap, dtype=np.float64),
		}
		if self.cursor:
			arrays["cursor"] = np.asarray(self.cursor, dtype=np.int32)
		if self.gains:
			arrays["gain_action"] = np.asarray(self.gains).reshape(n, 2)
			arrays["kp"] = np.asarray(self.kp).reshape(n, 6)
			arrays["kd"] = np.asarray(self.kd).reshape(n, 6)
		# Only when every step has one: a partial series would not line up with eef_pos.
		if n and len(self.ee_force) == n:
			arrays["ee_force"] = np.asarray(self.ee_force).reshape(n, 3)
			arrays["ee_torque"] = np.asarray(self.ee_torque).reshape(n, 3)
		attrs = {**self.attrs, "num_samples": n, "steps": int(steps),
				 "qvel_source": qvel_source,
				 "gain_varies": bool(self.gains and np.any(arrays["gain_action"] != arrays["gain_action"][0]))}
		if success is not None:
			attrs["success"] = bool(success)
		if cursor is not None:
			attrs["cursor"] = int(cursor)
		return episode_name(self.episode), arrays, attrs, self.curve

	def dry_run(self) -> tuple:
		"""An episode with no steps: the sampled curve and start pose only."""
		empty = {k: np.zeros((0,) + s) for k, s in (
			("action", (7,)), ("eef_goal_pos", (3,)), ("eef_goal_quat", (4,)),
			("eef_pos", (3,)), ("eef_quat", (4,)), ("qpos", (7,)), ("qvel", (7,)))}
		attrs = {**self.attrs, "num_samples": 0, "steps": 0, "dry_run": True}
		return episode_name(self.episode), empty, attrs, self.curve

	# ---- RealReach conveniences: unpack its obs/info into record() ----------

	def begin_reach(self, env, episode: int) -> None:
		self.begin(episode, env.reset_q, env.reset_quat, curve={
			"goal": env.goal, "waypoints": env._waypoints,
			"velocity_scales": env.velocity_scales[:env._curve_len]},
			ee_pos0=np.asarray(env._waypoints[0]))     # the curve starts at the reset EE

	def record_reach(self, seen_pos, obs: dict, info: dict) -> np.ndarray:
		return self.record(seen_pos, obs["robot0_eef_pos"], obs["robot0_eef_quat"],
						   info["qpos"], info["osc_goal_pos"], info["osc_goal_quat"],
						   info["osc_anchor_pos"], info["action"],
						   cursor=info["next_waypoint_idx"],
						   gain_action=info.get("gain_action"), qvel=info.get("qvel"),
						   ee_force=info.get("ee_force"), ee_torque=info.get("ee_torque"))

	def finish_reach(self, info: dict) -> tuple:
		return self.finish(info["episode_steps"], success=info.get("success", False),
						   cursor=info["next_waypoint_idx"])


# The name the reach scripts were written against.
ReachEpisodeRecorder = EpisodeRecorder
