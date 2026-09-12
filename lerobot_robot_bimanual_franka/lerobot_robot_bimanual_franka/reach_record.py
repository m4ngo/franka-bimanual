"""One episode's on-disk record, for a reach curve or a dataset trajectory.

Shared by `scripts/real_reach_rollout.py`, `scripts/real_trajectory_rollout.py`
and `scripts/check_real_reach_offline.py`, so every producer emits the same
artifact and the sim replay and the figure consume one schema.

Two frames, deliberately disjoint so no quantity is ever written in both:

  * MEASUREMENTS are WORLD -- the frame the worktable floor and the keep-out
    sphere are defined in, and the frame the figure is drawn in.
  * COMMANDS are BASE, under `replay` -- what a sim replay re-issues. Base
    because that is the frame the env produced them in; deriving base from world
    would mean inverting `robot_base_in_world`, which consumers never do.

`replay.osc_goal_*` is the DISPATCHED goal, after `clip_delta`, the `tuning`
fudges, the latched `goal_ori` and the worktable screen -- not an estimate of it.
Under a zero rotation delta `from_delta` never re-latches, so `osc_goal_quat`
is CONSTANT across a position-only episode. That is the real controller's
behaviour and a replay has to reproduce it, which is why the quaternion is
recorded per step rather than assumed.

The reach curve is optional. Without it the record is a plain trajectory and
the task-specific fields (cursor, success) are absent rather than faked.
"""

from __future__ import annotations

import numpy as np

import franka_config as fc

from .real_reach_geometry import base_to_world, base_to_world_quat


class EpisodeRecorder:
	"""Accumulates one episode, then emits the record dict."""

	def __init__(self, arm: str, seed: int | None = None, source: str = "arm"):
		self.arm = arm
		self.seed = None if seed is None else int(seed)
		self.source = source        # "arm" (measured) or "dataset" (as recorded)
		self.fps = float(fc.control_fps())

	def begin(self, episode: int, qpos0: np.ndarray, ee_quat0: np.ndarray,
			  curve: dict | None = None, extra: dict | None = None,
			  ee_pos0: np.ndarray | None = None) -> None:
		"""`curve`, when present, is {goal (3), waypoints (N,3), velocity_scales
		(N)} in base frame. `extra` lands in the record header verbatim."""
		self.episode = int(episode)
		self.trace, self.commanded, self.ee_quat = [], [], []
		self.qpos, self.cursor, self.stale = [], [], []
		self.osc_pos, self.osc_quat, self.actions = [], [], []
		conv_rotvec, conv_pos = fc.sim_ee_convention()
		self.replay = {
			"frame": "base", "arm": self.arm, "seed": self.seed,
			"ee_convention": "O_T_EE", "quat_order": "xyzw",
			"fps": self.fps,
			"qpos0": np.asarray(qpos0, dtype=np.float64).tolist(),
			# O_T_EE orientation at qpos0. A sim replay measures its EE frame
			# against this at the same joints, which is how it learns the rigid
			# offset between robosuite's grip site and the FR3's O_T_EE without
			# a hardcoded constant.
			"ee_quat0": np.asarray(ee_quat0, dtype=np.float64).tolist(),
			# O_T_EE position at qpos0, the other half of that measurement. A
			# reach curve starts here so it is implied there; a trajectory is not.
			"ee_pos0": (None if ee_pos0 is None
						else np.asarray(ee_pos0, dtype=np.float64).tolist()),
			# Copied in by the side that can read config/: the sim venv has no
			# franka_config. Describes the hand BODY for sim-trained policies;
			# the OSC replay measures its own frame and does not use this.
			"sim_ee_convention": {"rotvec_rad": conv_rotvec.tolist(),
								  "pos_tool_m": conv_pos.tolist()},
		}
		if curve is not None:
			self.replay.update({
				"control_mode": "EE_DELTA",
				"goal": np.asarray(curve["goal"], dtype=np.float64).tolist(),
				"waypoints": np.asarray(curve["waypoints"], dtype=np.float64).tolist(),
				"velocity_scales": np.asarray(curve["velocity_scales"], dtype=np.float64).tolist(),
			})
		else:
			self.replay["control_mode"] = "EE_POS"
		self.header = {"episode": self.episode, "arm": self.arm, "frame": "world",
					   "seed": self.seed, "fps": self.fps, "real_source": self.source,
					   # trace[t] is the state after action t acted for one control
					   # period. Records without this key read the state right
					   # after the send, i.e. one step stale.
					   "obs_timing": "post_period",
					   "replay": self.replay, **(extra or {})}
		if curve is not None:
			self.header["curve_len"] = int(len(curve["waypoints"]))

	def record(self, seen_pos, ee_pos, ee_quat, qpos, osc_goal_pos, osc_goal_quat,
			   anchor_pos, action, cursor: int | None = None) -> np.ndarray:
		"""One step, all base frame. `seen_pos` is the EE the policy acted on.

		Returns the measured EE in world, which a live caller needs for its floor
		and keep-out warnings.
		"""
		pos_w = base_to_world(self.arm, ee_pos)
		self.trace.append(pos_w.tolist())
		goal = np.asarray(osc_goal_pos, dtype=np.float64)
		self.commanded.append(base_to_world(self.arm, goal).tolist())
		self.osc_pos.append(goal.tolist())
		self.osc_quat.append(np.asarray(osc_goal_quat, dtype=np.float64).tolist())
		self.actions.append(np.asarray(action, dtype=np.float64).tolist())
		# How much fresher send_action's own read was than the pose the policy
		# acted on: the stale-anchor effect, as one number per step.
		self.stale.append(float(np.linalg.norm(
			np.asarray(anchor_pos, dtype=np.float64) - np.asarray(seen_pos, dtype=np.float64))))
		self.ee_quat.append(base_to_world_quat(self.arm, ee_quat).tolist())
		self.qpos.append(np.asarray(qpos, dtype=np.float64).tolist())
		if cursor is not None:
			self.cursor.append(int(cursor))
		return pos_w

	def finish(self, steps: int, success: bool | None = None,
			   cursor: int | None = None) -> dict:
		self.replay.update({"osc_goal_pos": self.osc_pos, "osc_goal_quat": self.osc_quat,
							"actions": self.actions})
		out = {**self.header, "steps": int(steps),
			   "trace": self.trace, "commanded": self.commanded,
			   "ee_quat": self.ee_quat, "qpos": self.qpos,
			   "stale_anchor_m": self.stale}
		if success is not None:
			out["success"] = bool(success)
		if cursor is not None:
			out["cursor"] = int(cursor)
			out["cursor_trace"] = self.cursor
		return out

	def dry_run(self) -> dict:
		return {**self.header, "dry_run": True}

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
						   cursor=info["next_waypoint_idx"])

	def finish_reach(self, info: dict) -> dict:
		return self.finish(info["episode_steps"], success=info.get("success", False),
						   cursor=info["next_waypoint_idx"])


# The name the reach scripts were written against.
ReachEpisodeRecorder = EpisodeRecorder
