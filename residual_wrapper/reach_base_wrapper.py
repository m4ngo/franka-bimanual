import numpy as np
import torch

class ReachBaseWrapper:
	"""Analytic chunked base policy for the reach task.

	Reads waypoint state from obs (env-tracked: see ReachObservationWrapper):
	  - obs["robot0_eef_pos"]:    (B, 3)
	  - obs["waypoints"]:         (B, MAX_WAYPOINTS, 3) — past-end slots are
	                              filled with the goal so any lookahead idx
	                              clamps cleanly without bounds-checking.
	  - obs["next_waypoint_idx"]: (B, 1) — env auto-advances when EE is within
	                              advance_threshold of the current waypoint.

	Output action layout matches LIBERO/pi0.5 exactly: 7D = 6 OSC (3 trans +
	3 axis-angle rot) + 1 gripper. Rot delta is zero (maintain orientation)
	unless include_orient=True, in which case it tracks the env's per-waypoint
	target quats (obs["waypoint_quats"]) with the same lookahead + forward-sim
	pacing as translation. Gripper command is zero (no manipulation). Same action for all timesteps
	in the chunk — the OSC controller smooths through dense waypoints, and
	the chunk gets re-predicted at every chunk boundary anyway.

	lookahead_k: target waypoints[idx + k] instead of waypoints[idx] so the
	OSC delta has meaningful magnitude on dense curves (waypoints ~1-2 cm
	apart). Tuned via cfg.base_policy.lookahead_k (default 3).

	Unit conversion: the raw `target - eef_pos` is in meters, but the OSC
	controller interprets inputs as normalized [-1, 1] mapped to
	[±osc_output_max] meters of target displacement per control step. So we
	divide the raw delta by osc_output_max to get OSC input units. pi0.5
	emits already-normalized actions, so this scaling is implicit in its
	training — the analytic policy has to do it explicitly.

	Chunk shape: the policy emits an EVOLVING sequence of deltas through the
	chunk, not the same delta repeated. Per inner step, it forward-sims a
	predicted EE position and re-targets a new lookahead waypoint. The
	induced chunk waypoints trace a curve through the Bezier path, mirroring
	how pi0.5's chunked predictions evolve.

	chunk_alpha: fraction of the commanded OSC delta the predicted EE travels
	per inner step in the forward-sim. Theoretical first-order response under
	kp=150, damping=1, control_freq=20Hz is ~0.27 (τ=2/√kp ≈ 0.163s, dt=0.05s),
	so 0.3 is the default. 1.0 = OSC saturates each step → most visibly curved
	chunk but overshoots reality, which can confuse OSC playback. Smaller
	values evolve more conservatively but produce cleaner tracking.
	"""

	def __init__(self, chunk_size, prediction_horizon, max_step, lookahead_k=3,
	             action_dim=7, osc_output_max=0.05, advance_threshold=0.05,
	             chunk_alpha=0.3, velocity_lookahead_window=10,
	             min_velocity_floor=0.2, include_orient=False,
	             osc_rot_output_max=0.5):
		self.chunk_size = int(chunk_size)
		self.prediction_horizon = int(prediction_horizon)
		self.max_step = float(max_step)
		self.lookahead_k = int(lookahead_k)
		self.action_dim = int(action_dim)
		self.osc_output_max = float(osc_output_max)
		self.advance_threshold = float(advance_threshold)
		self.chunk_alpha = float(chunk_alpha)
		# Orientation extension: when True, also command rotation toward the
		# per-waypoint target quats (obs["waypoint_quats"], auto-emitted by
		# ReachObservationWrapper when the env samples an orientation curve).
		# osc_rot_output_max mirrors osc_output_max for the rotation channels
		# (radians of target displacement per unit OSC input).
		self.include_orient = bool(include_orient)
		self.osc_rot_output_max = float(osc_rot_output_max)
		# Window for the velocity-aware speed cap: take MIN over the next N
		# dense waypoints when computing the OSC clip bound, so upcoming slow
		# nodes are seen before the cursor races past them via nearest-next.
		self.velocity_lookahead_window = int(velocity_lookahead_window)
		# Minimum effective velocity floor — prevents EE from fully stopping at
		# v=0 waypoints (which would deadlock cursor advance since cursor needs
		# EE motion to fire nearest-next). 0.05 → ~2.5mm/step floor under
		# kp=150 OSC. EE still slows dramatically at junctions, just doesn't
		# permanently stop.
		self.min_velocity_floor = float(min_velocity_floor)
		# FAST's sample_base_policy reads these to determine which obs keys to
		# materialize before calling __call__.
		self.low_dim_keys = [
			"robot0_eef_pos", "waypoints", "next_waypoint_idx", "velocity_scales",
		]
		if self.include_orient:
			self.low_dim_keys += ["robot0_eef_quat", "waypoint_quats"]
		self.image_keys = []

	def __call__(self, obs, return_numpy=True):
		actions = self.predict_full_chunk(obs, return_numpy=return_numpy)
		return actions[:, :self.chunk_size, :]

	def predict_full_chunk(self, obs, return_numpy=True, horizon=None):
		"""Return (B, horizon, action_dim) — evolving deltas via forward-sim.
		Each chunk step re-targets a fresh lookahead waypoint from the predicted
		EE position, so the chunk's induced waypoints trace a curve through the
		Bezier path.

		`horizon` defaults to self.prediction_horizon (matches training). Pass a
		larger value to look further into the base's intent (e.g. for the
		animate_reach_episode viz). Since the base is deterministic forward-sim,
		any horizon length is well-defined — no model retraining or refit needed.
		"""
		horizon = int(horizon) if horizon is not None else self.prediction_horizon
		eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)         # (B, 3)
		waypoints = np.asarray(obs["waypoints"], dtype=np.float32)             # (B, K, 3)
		idx = np.asarray(obs["next_waypoint_idx"], dtype=np.int32).reshape(-1)  # (B,)
		# Per-dense-waypoint velocity scale [0, 1]. Backwards-compat: if the
		# env doesn't surface this key, default to all-ones (= no velocity cap,
		# matches pre-velocity behavior).
		if "velocity_scales" in obs:
			velocity_scales = np.asarray(obs["velocity_scales"], dtype=np.float32)  # (B, K)
		else:
			velocity_scales = np.ones((waypoints.shape[0], waypoints.shape[1]), dtype=np.float32)

		K = waypoints.shape[1]
		B = waypoints.shape[0]
		bound = min(self.max_step, 1.0)

		# Working state for the forward-sim: predicted EE position + cursor idx
		# that advance through the chunk timesteps. cur_idx mirrors the env's
		# advance logic so the chunk shape matches the env's playback semantics.
		cur_eef = eef_pos.copy()
		cur_idx = idx.copy().astype(np.int32)

		# Orientation extension: forward-simulated EE quat + per-waypoint target
		# quats. Mirrors the position forward-sim so the rotation commands pace
		# with the curve instead of rushing straight to the goal orientation.
		# orient_on = self.include_orient and "waypoint_quats" in obs
		# if orient_on:
		# 	from robosuite.utils import transform_utils as T
		# 	waypoint_quats = np.asarray(obs["waypoint_quats"], dtype=np.float32)  # (B, K, 4)
		# 	cur_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()  # (B, 4)

		actions = np.zeros(
			(B, horizon, self.action_dim), dtype=np.float32,
		)
		batch_range = np.arange(B)

		for t in range(horizon):
			# Advance the virtual cursor for each env in lockstep with the
			# forward-simulated EE. Mirrors Reach._post_action's combined
			# proximity + nearest-next advance — cursor advances when the
			# predicted EE is either within advance_threshold of the current
			# waypoint OR closer to the next waypoint than the current.
			# Robust to corner-cutting at segment junctions.
			for b in range(B):
				while cur_idx[b] < K - 1:
					idx_b = cur_idx[b]
					cur_d = np.linalg.norm(cur_eef[b] - waypoints[b, idx_b])
					next_d = np.linalg.norm(cur_eef[b] - waypoints[b, idx_b + 1])
					if cur_d < self.advance_threshold or next_d < cur_d:
						cur_idx[b] += 1
					else:
						break

			target_idx = np.minimum(cur_idx + self.lookahead_k, K - 1)
			target = waypoints[batch_range, target_idx]                        # (B, 3)

			# Meters -> OSC input units, clipped to a velocity-aware bound.
			# v_window_min is the MIN velocity scale over the next
			# velocity_lookahead_window dense waypoints — so upcoming slow
			# waypoints (e.g. junction v=0) are visible even when nearest-next
			# cursor advance would otherwise skip past them in a single inner
			# step. Robust to corner-cutting at sharp kinks.
			window_end = np.minimum(cur_idx + self.velocity_lookahead_window, K)
			v_window_min = np.array([
				velocity_scales[b, cur_idx[b] : window_end[b]].min()
				if window_end[b] > cur_idx[b] else 0.0
				for b in range(B)
			], dtype=np.float32)                                                # (B,)
			# Floor the effective cap so EE never fully stops — see
			# min_velocity_floor docstring.
			v_window_min = np.maximum(v_window_min, self.min_velocity_floor)
			eff_bound = bound * v_window_min                                    # (B,)
			delta_osc = (target - cur_eef) / self.osc_output_max               # (B, 3)
			delta_osc = np.clip(delta_osc, -eff_bound[:, None], eff_bound[:, None])  # (B, 3)

			actions[:, t, :3] = delta_osc.astype(np.float32)

			# Rotation (3:6): command toward the same lookahead waypoint's target
			# quat, mirroring the position channel. Zero when the orientation
			# extension is off (bit-identical to the original behavior).
			# if orient_on:
			# 	for b in range(B):
			# 		# Axis-angle world-frame error current → target; the same
			# 		# signal OSC_POSE consumes. Normalize to OSC input units and
			# 		# clip by the shared velocity-aware bound so rotation slows
			# 		# at junctions in lockstep with translation.
			# 		rot_err = T.get_orientation_error(
			# 			waypoint_quats[b, target_idx[b]], cur_quat[b],
			# 		)
			# 		rot_osc = np.clip(
			# 			rot_err / self.osc_rot_output_max,
			# 			-eff_bound[b], eff_bound[b],
			# 		).astype(np.float32)
			# 		actions[b, t, 3:6] = rot_osc
			# 		# Forward-sim the quat by the realized fraction of the
			# 		# commanded rotation (world-frame left-multiply).
			# 		step_aa = self.chunk_alpha * rot_osc * self.osc_rot_output_max
			# 		cur_quat[b] = T.quat_multiply(
			# 			T.axisangle2quat(step_aa), cur_quat[b],
			# 		)
			# Gripper (6:) stays zero.

			# Forward-sim: advance predicted EE by chunk_alpha * commanded delta.
			# alpha matches the first-order step response under our OSC gains
			# (~0.27 from τ=2/√kp), so the chunk's induced waypoints track the
			# rate the real EE actually moves.
			cur_eef = cur_eef + self.chunk_alpha * delta_osc * self.osc_output_max

		if return_numpy:
			return actions
		return torch.from_numpy(actions).float()


def _load_reach_policy(cfg):
	"""Construct the analytic ReachBaseWrapper from cfg.base_policy fields.
	The policy-side advance_threshold mirrors the env-side one (reach.advance_threshold)
	by default so the chunk's virtual cursor advancement matches what the env
	actually does at playback time.
	"""
	advance_threshold = cfg.base_policy.get(
		"advance_threshold",
		cfg.get("reach", {}).get("advance_threshold", 0.05),
	)
	return ReachBaseWrapper(
		chunk_size=cfg.base_policy.chunk_size,
		prediction_horizon=cfg.base_policy.get("prediction_horizon", cfg.base_policy.chunk_size),
		max_step=cfg.base_policy.get("max_step", 1.0),
		lookahead_k=cfg.base_policy.get("lookahead_k", 3),
		action_dim=cfg.base_policy.get("action_dim", 7),
		osc_output_max=cfg.base_policy.get("osc_output_max", 0.05),
		advance_threshold=advance_threshold,
		chunk_alpha=cfg.base_policy.get("chunk_alpha", 0.3),
		velocity_lookahead_window=cfg.base_policy.get("velocity_lookahead_window", 10),
		min_velocity_floor=cfg.base_policy.get("min_velocity_floor", 0.2),
		# Orientation extension: keyed off the env-side knob so the base tracks
		# orientation exactly when the env samples an orientation curve.
		include_orient=float(cfg.get("reach", {}).get("orient_delta_max_deg", 0.0)) > 0.0,
		osc_rot_output_max=cfg.base_policy.get("osc_rot_output_max", 0.5),
	)