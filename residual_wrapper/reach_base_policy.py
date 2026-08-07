"""Real-robot adapter around ReachBaseWrapper for the residual eval pipeline.

reach_base_wrapper.ReachBaseWrapper and reach.Reach are sim-training
artifacts: Reach owns curve sampling, per-step cursor advance, and the
robosuite obs/action contract; ReachBaseWrapper is the analytic policy that
consumes that contract. Neither talks to real hardware.

run_episode.py's control loop (_run_episode / _infer_chunk) instead expects a
`base_policy` object shaped like policy_wrapper.BasePolicy:

    base_policy.reset() -> None
    base_policy.infer(obs_no_depth: dict) -> np.ndarray, shape (T, 10)
        [dx, dy, dz, dqx, dqy, dqz, dqw, gripper, kp, kd], physical units
        (metres, unit delta-quaternion xyzw), T >= _RESIDUAL_HORIZON, IN
        ROBOT BASE FRAME -- this is the frame build_action/send_action
        expect (see env_wrapper.build_action).

ReachBasePolicy is that adapter. It:
  - owns the per-episode curve state that Reach._sample_episode_curves /
    _post_action own in sim (dense waypoint curve + cursor index), built from
    a user-supplied goal instead of workspace-bound rejection sampling (no
    simulator to test real reachability against);
  - each .infer() call, reads the real EE pose out of obs, maps it into
    WORLD frame (matching the frame the caller-supplied goal is given in --
    the same frame the point cloud and --proprio-frame world already use),
    advances the cursor with the same proximity-OR-nearest-next rule
    Reach._post_action uses, packages the {robot0_eef_pos, waypoints,
    next_waypoint_idx, velocity_scales} obs ReachBaseWrapper expects (all in
    world frame), and converts its OSC-normalised (T, 7) world-frame output
    chunk into the physical-units (T, 10) ROBOT-FRAME chunk the rest of the
    pipeline (process_chunk, build_action, ResidualPolicy) already assumes.

Gripper and impedance gains are not part of ReachBaseWrapper's action space
(it is position/orientation only -- see its docstring: "Gripper command is
zero"). This adapter holds the gripper at its value from the first observation
of the episode, and commands fixed impedance gains from franka_config, both
overridable at construction time.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

import franka_config as fc
from env_wrapper import current_ee_pose, ee_pose_to_world, _RESIDUAL_HORIZON, _CHUNK_EXEC
from reach_base_wrapper import ReachBaseWrapper

logger = logging.getLogger(__name__)


MAX_WAYPOINTS = 128
SEED = 42

def sample_dense_curve(start, goal, n_points=25, control_offset=0.12, rng=None):
	"""Sample n_points along a smooth cubic Bezier from start to goal.

	Two interior control points at t=1/3 and t=2/3 along the start->goal line,
	each shifted by a random perpendicular offset in the plane normal to the
	line. control_offset (meters) caps the offset magnitude. Sampling within a
	disk uses the standard r = R * sqrt(uniform) trick for uniform-in-disk.

	Workspace bounds are intentionally NOT enforced — the curve is already
	bounded by start, goal, and control_offset, so interior points only stray
	~control_offset (10-15cm) perpendicular to the line. Clipping would create
	flat sections that break cursor-advance on those stretches.

	Returns: [n_points, 3] dense waypoint array along the curve.
    """
	rng = rng if rng is not None else np.random
	rng.seed(SEED)
	start = np.asarray(start, dtype=np.float64)
	goal = np.asarray(goal, dtype=np.float64)
	line = goal - start
	line_len = float(np.linalg.norm(line))
	if line_len < 1e-6:
		return np.tile(start[None, :], (n_points, 1)).astype(np.float64)
	line_dir = line / line_len

	# Two perpendicular unit vectors in the plane normal to line_dir.
	# Pick any reference not parallel to line_dir.
	ref = np.array([0., 0., 1.]) if abs(line_dir[2]) < 0.9 else np.array([1., 0., 0.])
	perp1 = np.cross(line_dir, ref); perp1 /= np.linalg.norm(perp1)
	perp2 = np.cross(line_dir, perp1)

	# Shared bow direction for BOTH control points (independent random
	# magnitudes). Guarantees a single-bow shape — independent θ for P1 and
	# P2 lets the bends cancel and produces near-straight inflection curves
	# ~half the time, which is what we don't want for the testbed.
	theta = rng.uniform(0.0, 2.0 * np.pi)
	bow_dir = perp1 * np.cos(theta) + perp2 * np.sin(theta)

	def _bow(rng_):
		r = control_offset * np.sqrt(rng_.uniform())
		return r * bow_dir

	P0 = start
	P1 = start + line / 3.0 + _bow(rng)
	P2 = start + 2.0 * line / 3.0 + _bow(rng)
	P3 = goal

	ts = np.linspace(0.0, 1.0, n_points)
	pts = np.stack([
		(1 - t) ** 3 * P0
		+ 3 * (1 - t) ** 2 * t * P1
		+ 3 * (1 - t) * t ** 2 * P2
		+ t ** 3 * P3
		for t in ts
	], axis=0)
	return pts


class ReachBasePolicy:
    """BasePolicy-compatible wrapper around the analytic ReachBaseWrapper.

    goal_pos (and the curve built toward it) are in WORLD frame. Internally
    all curve/cursor math runs in world frame; infer() converts the resulting
    position delta back to robot base frame before returning, since that is
    the frame build_action/send_action require. Rotation deltas are frame-
    invariant under a fixed-orientation robot-to-world transform (rotating a
    delta rotation by a constant offset commutes with composing it), so only
    the translational delta needs the explicit back-conversion.

    One instance = one episode's worth of curve state. Construct a fresh
    instance (or call reset_curve) per episode; reset() alone (the method
    _run_episode calls) only resets the underlying analytic policy's
    internal batch state -- it does NOT resample the curve, mirroring how
    BasePolicy.reset() clears model state but not task config.
    """

    def __init__(
        self,
        goal_pos: np.ndarray,
        r_robot_in_world: np.ndarray,
        t_robot_in_world: np.ndarray,
        n_waypoints: int = 25,
        control_offset: float = 0.12,
        advance_threshold: float = 0.02,
        lookahead_k: int = 3,
        osc_output_max: float = 0.05,
        osc_rot_output_max: float = 0.5,
        max_step: float = 1.0,
        chunk_alpha: float = 0.3,
        velocity_lookahead_window: int = 10,
        min_velocity_floor: float = 0.2,
        include_orient: bool = False,
        gripper_value: "float | None" = None,
        kp: "float | None" = None,
        kd: "float | None" = None,
        sim_proprio_convention: bool = True,
    ) -> None:
        if n_waypoints > MAX_WAYPOINTS:
            raise ValueError(f"n_waypoints={n_waypoints} exceeds MAX_WAYPOINTS={MAX_WAYPOINTS}")

        # goal_pos is WORLD frame, per the caller's convention.
        self.goal_pos_world = np.asarray(goal_pos, dtype=np.float64)
        self.n_waypoints = int(n_waypoints)
        self.control_offset = float(control_offset)
        self.advance_threshold = float(advance_threshold)
        self.sim_proprio_convention = sim_proprio_convention

        # Static robot-base -> world calibration (see env_wrapper.ee_pose_to_world).
        # Used both to express the real EE pose in world frame for curve/cursor
        # math, and to rotate the resulting world-frame position delta back to
        # robot frame for build_action/send_action.
        self._r_robot_in_world = np.asarray(r_robot_in_world, dtype=np.float64)
        self._t_robot_in_world = np.asarray(t_robot_in_world, dtype=np.float64)
        self._r_world_in_robot = self._r_robot_in_world.T  # orthonormal -> inverse = transpose

        # Horizon requested from the analytic policy each call: _infer_chunk /
        # process_chunk need at least _RESIDUAL_HORIZON steps of context, and
        # the control loop executes _CHUNK_EXEC of them -- request the max so
        # both are satisfied regardless of which is larger.
        self._horizon = max(_CHUNK_EXEC, _RESIDUAL_HORIZON)

        self._policy = ReachBaseWrapper(
            chunk_size=self._horizon,
            prediction_horizon=self._horizon,
            max_step=max_step,
            lookahead_k=lookahead_k,
            action_dim=7,
            osc_output_max=osc_output_max,
            advance_threshold=advance_threshold,
            chunk_alpha=chunk_alpha,
            velocity_lookahead_window=velocity_lookahead_window,
            min_velocity_floor=min_velocity_floor,
            include_orient=include_orient,
            osc_rot_output_max=osc_rot_output_max,
        )
        self._osc_output_max = float(osc_output_max)
        self._osc_rot_output_max = float(osc_rot_output_max)

        # Fixed action-space fields ReachBaseWrapper doesn't produce.
        self._gripper_override = gripper_value  # None -> latch from first obs
        self._gripper_value: "float | None" = gripper_value
        self._kp = float(kp) if kp is not None else float(fc.policy("sysid.default_kp"))
        self._kd = float(kd) if kd is not None else float(fc.policy("sysid.default_kd"))

        # Curve state (WORLD frame), populated on first .infer() call once we
        # know the real start pose (mirrors Reach._reset_internal pinning the
        # curve start to the post-reset EE position).
        self._waypoints_world: "np.ndarray | None" = None    # (MAX_WAYPOINTS, 3)
        self._velocity_scales: "np.ndarray | None" = None
        self._curve_len = 0
        self._next_waypoint_idx = 0

    # ------------------------------------------------------------------
    # BasePolicy-compatible interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear analytic-policy batch state. Does NOT resample the curve --
        call reset_curve() (or construct a new instance) for a new episode.

        CALLER WARNING: _warmup_policies() in run_episode.py calls
        base_policy.infer() several times before controller.home() runs, to
        pay compile/cuDNN costs off the critical path. For this adapter the
        FIRST infer() call is what anchors the curve to the current EE pose
        (see infer() -> _build_curve). If warmup runs first, the curve gets
        anchored to the pre-homing pose, not the real episode start. Callers
        MUST call reset_curve() again after homing and before _run_episode
        for every episode, including the first -- reset() alone does not fix
        this, by design (see its docstring)."""

    def reset_curve(self, goal_pos_world: "np.ndarray | None" = None) -> None:
        """Start a new episode: forget the sampled curve and gripper latch.

        Call this (not reset()) between episodes, mirroring Reach's per-reset
        _sample_episode_curves. The curve is (re-)built lazily on the next
        infer() call, once the real start pose is known. goal_pos_world, if
        given, replaces the goal (WORLD frame).
        """
        if goal_pos_world is not None:
            self.goal_pos_world = np.asarray(goal_pos_world, dtype=np.float64)
        self._waypoints_world = None
        self._velocity_scales = None
        self._curve_len = 0
        self._next_waypoint_idx = 0
        self._gripper_value = self._gripper_override

    def infer(self, obs: dict) -> np.ndarray:
        """One base-policy pass. See module docstring for the (T, 10) contract."""
        ee_pose_robot = current_ee_pose(obs, sim_convention=self.sim_proprio_convention)
        ee_pose_world = ee_pose_to_world(ee_pose_robot, self._r_robot_in_world, self._t_robot_in_world)
        eef_pos_world = ee_pose_world[:3].astype(np.float64)

        if self._waypoints_world is None:
            self._build_curve(eef_pos_world)
        if self._gripper_value is None:
            self._gripper_value = float(ee_pose_robot[7])

        self._advance_cursor(eef_pos_world)

        policy_obs = {
            "robot0_eef_pos": eef_pos_world.astype(np.float32)[None, :],
            "waypoints": self._waypoints_world[None, ...],
            "next_waypoint_idx": np.array([[self._next_waypoint_idx]], dtype=np.float32),
            "velocity_scales": self._velocity_scales[None, ...],
        }
        chunk = self._policy.predict_full_chunk(
            policy_obs, return_numpy=True, horizon=self._horizon
        )[0]  # (horizon, 7): [dx, dy, dz, rx, ry, rz, grip], WORLD frame

        return self._to_physical_chunk(chunk)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_curve(self, start_pos_world: np.ndarray) -> None:
        """Sample the dense Bezier curve (WORLD frame) from the real start
        pose to the user-supplied world-frame goal, padded to MAX_WAYPOINTS
        (past-end = goal, so any lookahead index clamps cleanly -- mirrors
        Reach._sample_episode_curves). No workspace-bound rejection sampling:
        reachability of goal_pos_world is the caller's responsibility on real
        hardware.
        """
        curve = sample_dense_curve(
            start_pos_world, self.goal_pos_world,
            n_points=self.n_waypoints,
            control_offset=self.control_offset,
        )
        padded = np.zeros((MAX_WAYPOINTS, 3), dtype=np.float32)
        padded[: self.n_waypoints] = curve.astype(np.float32)
        padded[self.n_waypoints :] = self.goal_pos_world.astype(np.float32)

        # Full-speed-to-stop profile (mirrors Reach's single-segment default
        # [1.0, 0.0]): no per-node velocity plan supplied for the real task.
        velocities = np.zeros((MAX_WAYPOINTS,), dtype=np.float32)
        ramp = np.linspace(1.0, 0.0, self.n_waypoints, dtype=np.float32)
        velocities[: self.n_waypoints] = ramp

        self._waypoints_world = padded
        self._velocity_scales = velocities
        self._curve_len = self.n_waypoints
        self._next_waypoint_idx = 0
        logger.info(
            "reach curve built (world frame): start=%s goal=%s n_waypoints=%d\n%s",
            np.round(start_pos_world, 4), np.round(self.goal_pos_world, 4), self.n_waypoints,
            "\n".join(f"  wp {i:3d}  {p[0]: .4f} {p[1]: .4f} {p[2]: .4f}"
                      for i, p in enumerate(curve)),
        )

    def _advance_cursor(self, eef_pos_world: np.ndarray) -> None:
        """Same proximity-OR-nearest-next rule as Reach._post_action, run
        against the real measured EE position, in world frame."""
        wp = self._waypoints_world
        while self._next_waypoint_idx < self._curve_len - 1:
            idx = self._next_waypoint_idx
            cur_d = float(np.linalg.norm(eef_pos_world - wp[idx]))
            next_d = float(np.linalg.norm(eef_pos_world - wp[idx + 1]))
            if cur_d < self.advance_threshold or next_d < cur_d:
                self._next_waypoint_idx += 1
            else:
                break
        if (self._next_waypoint_idx == self._curve_len - 1
                and float(np.linalg.norm(eef_pos_world - wp[self._next_waypoint_idx])) < self.advance_threshold):
            self._next_waypoint_idx += 1

    def _to_physical_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Convert ReachBaseWrapper's OSC-normalised (T, 7) WORLD-frame output
        to the physical-units (T, 10) ROBOT-frame chunk BasePolicy.infer()
        contractually returns.

        chunk columns: [dx, dy, dz, rx, ry, rz, grip] normalised to
        [-osc_output_max/-osc_rot_output_max .. +] via division inside
        predict_full_chunk; multiplying back out gives metres / radians, in
        WORLD frame (since eef_pos/waypoints fed to predict_full_chunk were
        world frame). Both the position delta and the rotation delta are
        rotated by r_world_in_robot to express them in robot base frame --
        deltas transform by rotation only (no translation component), so this
        is the same R^T @ v used by ee_pose_to_world's inverse. Rotation is
        axis-angle -> converted to a delta unit quaternion (xyzw) to match
        process_chunk's Rotation.from_quat(...).as_rotvec() step.
        Gripper channel is ignored (always 0 from the analytic policy);
        replaced with the latched gripper value. Gains are fixed.
        """
        T = chunk.shape[0]
        pos_delta_world = chunk[:, :3].astype(np.float64) * self._osc_output_max
        rotvec_world = chunk[:, 3:6].astype(np.float64) * self._osc_rot_output_max

        pos_delta_robot = (self._r_world_in_robot @ pos_delta_world.T).T
        rotvec_robot = (self._r_world_in_robot @ rotvec_world.T).T
        quat_xyzw_robot = Rotation.from_rotvec(rotvec_robot).as_quat()  # (T, 4)

        out = np.zeros((T, 10), dtype=np.float32)
        out[:, 0:3] = pos_delta_robot
        out[:, 3:7] = quat_xyzw_robot
        out[:, 7] = self._gripper_value
        out[:, 8] = self._kp
        out[:, 9] = self._kd
        return out

    @property
    def waypoints_world(self) -> "np.ndarray | None":
        """The active (n_waypoints, 3) curve in WORLD frame, without the
        goal-valued padding out to MAX_WAYPOINTS. None before the first
        infer() builds it."""
        if self._waypoints_world is None:
            return None
        return np.asarray(self._waypoints_world[: self._curve_len], dtype=np.float64)

    @property
    def curve_progress(self) -> float:
        """Fraction of the current curve's cursor advanced past (for logging)."""
        if self._curve_len == 0:
            return 0.0
        return self._next_waypoint_idx / max(self._curve_len - 1, 1)