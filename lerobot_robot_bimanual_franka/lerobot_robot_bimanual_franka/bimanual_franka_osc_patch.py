"""
Patch: wire OSCVelocityController into BimanualFranka's EE control paths.

Drop-in changes to bimanual_franka.py. Everything below is either a diff
against the existing file or a new method to add -- nothing here needs the
old `_ee_delta` math kept around; OSC-style goal integration replaces it.

Assumes (all already true of your files):
  - `franka_fk(q) -> (pos_xyz, quat_xyzw)` exists (already imported)
  - `franka_jacobian(q) -> (6, NUM_JOINTS)` ndarray, rows = [linear; angular],
    matching KinematicSnapshot's field order (see below)
  - KinematicSnapshot = (q, dq, jacobian, ee_pos, ee_quat_xyzw, ee_twist)
    i.e. the Jacobian is already index [2] of the snapshot -- no separate
    franka_jacobian import needed, use kin[2] directly.
  - osc_velocity_controller.py (the file alongside this patch) is added to
    the same package as bimanual_franka.py.
"""

# ============================================================================
# 1. NEW IMPORT -- add near the top of bimanual_franka.py, with the other
#    local imports:
# ============================================================================
#
#   from .osc_velocity_controller import OSCVelocityController
#
# (franka_jacobian is NOT needed as a separate import -- KinematicSnapshot
#  already carries the Jacobian at index 2: (q, dq, J, pos, quat_xyzw, twist))


# ============================================================================
# 2. __init__ -- add after self.grippers is built, near where
#    self._cached_kin_state is initialized:
# ============================================================================
#
#   self._osc_vel: dict[str, OSCVelocityController] = {
#       arm: OSCVelocityController(num_joints=NUM_JOINTS) for arm in self.active_arms
#   }
#   self._goal_pos: dict[str, np.ndarray] = {}
#   self._goal_quat_xyzw: dict[str, np.ndarray] = {}
#   self._home_q: dict[str, np.ndarray] = {}  # nullspace target; set in home()


# ============================================================================
# 3. connect() -- after homing (grippers.home()), seed the OSC goal from the
#    robot's actual current pose so the first EE_DELTA tick doesn't jump:
# ============================================================================
#
#   for arm in self.active_arms:
#       kin = self.robot_manager.current_kinematic_state(arm)
#       self._goal_pos[arm] = np.asarray(kin[3], dtype=np.float64)
#       self._goal_quat_xyzw[arm] = np.asarray(kin[4], dtype=np.float64)
#
# (self._home_q is populated separately in home(), see #6 below -- until
#  home() is called at least once, nullspace biasing is simply skipped.)


# ============================================================================
# 4. send_action() -- EDIT the EE_DELTA branch. Old code called
#    self._ee_delta(...) as a @staticmethod; replace with a call to the new
#    instance method self._qdot_ee_delta(...) (defined in #7 below), which
#    needs `self` for the persistent goal/controller state.
# ============================================================================
#
# OLD:
#
#   cmds = self.safety.shape_ee(
#       {arm: self._ee_delta(self.kp_gain, self.kd_gain, action, arm, kin[arm],
#                             self.delta_pos, self.delta_rot, self.config.use_noise,
#                             self.config.noise_pos_scale, self.config.noise_rot_scale)
#        for arm in self.active_arms}, kin
#   )
#
# NEW:
#
#   cmds = self.safety.shape_ee(
#       {arm: self._qdot_ee_delta(arm, action, kin[arm], self.delta_pos, self.delta_rot)
#        for arm in self.active_arms}, kin
#   )
#
# The unnormalize-actions block just above (multiplying by
# _EE_TRANSLATION_FUDGE_FACTOR / _EE_ROTATION_FUDGE_FACTOR) stays unchanged --
# _qdot_ee_delta still reads action[f"{arm}_x"] etc. the same way _ee_delta did.


# ============================================================================
# 5. send_action() -- EDIT the EE_POS branch the same way, replacing
#    self._ee_pd(...) with self._qdot_ee_pos(...) (defined in #8 below):
# ============================================================================
#
# OLD:
#
#   cmds = self.safety.shape_ee(
#       {arm: self._ee_pd(self.kp_gain, self.kd_gain, action, arm, kin[arm],
#                          self.delta_pos, self.delta_rot, ignore_action)
#        for arm in self.active_arms}, kin
#   )
#
# NEW:
#
#   cmds = self.safety.shape_ee(
#       {arm: self._qdot_ee_pos(arm, action, kin[arm], ignore_action)
#        for arm in self.active_arms}, kin
#   )


# ============================================================================
# 6. home() -- after a successful homing pass (both the EE-homing branch and
#    the joint-PD branch, right before each `return True`), record the home
#    q as the nullspace target AND reset the OSC goal so any pending
#    EE_DELTA goal state doesn't fight the freshly-homed pose:
# ============================================================================
#
#   for arm in names:
#       self._home_q[arm] = targets_q[arm].copy()
#       kin_arm = self.robot_manager.current_kinematic_state(arm)
#       self._goal_pos[arm] = np.asarray(kin_arm[3], dtype=np.float64)
#       self._goal_quat_xyzw[arm] = np.asarray(kin_arm[4], dtype=np.float64)


# ============================================================================
# 7. NEW METHOD -- replaces the old @staticmethod _ee_delta. Instance method
#    (needs self._goal_pos / self._osc_vel), so add it as a regular method,
#    not a staticmethod.
# ============================================================================

import numpy as np  # already imported in bimanual_franka.py


def _qdot_ee_delta(
    self,
    arm: str,
    action,  # RobotAction
    snap,  # KinematicSnapshot = (q, dq, J, ee_pos, ee_quat_xyzw, ee_twist)
    dpos_cached: np.ndarray,
    drot_cached: np.ndarray,
) -> np.ndarray:
    """Replacement for the old `_ee_delta`. Integrates the incoming delta
    action into a persistent per-arm goal pose (mirroring robosuite OSC's
    set_goal() under use_delta=True), then servos the current EE toward that
    goal with OSCVelocityController -- rather than treating the raw delta as
    a one-shot velocity command every tick.
    """
    action_dpos = np.fromiter(
        (action[f"{arm}_{ax}"] for ax in ("x", "y", "z")),
        dtype=np.float64, count=3,
    )
    action_dquat_xyzw = np.fromiter(
        (action[f"{arm}_{ax}"] for ax in ("qx", "qy", "qz", "qw")),
        dtype=np.float64, count=4,
    )

    # ---- integrate delta into the persistent goal (position: simple sum;
    #      orientation: compose as delta * goal) ----
    self._goal_pos[arm] = self._goal_pos[arm] + action_dpos + dpos_cached

    dq = action_dquat_xyzw / max(float(np.linalg.norm(action_dquat_xyzw)), 1e-12)
    gx, gy, gz, gw = self._goal_quat_xyzw[arm]
    dx, dy, dz, dw = dq
    new_quat = np.array([
        dw * gx + dx * gw + dy * gz - dz * gy,
        dw * gy - dx * gz + dy * gw + dz * gx,
        dw * gz + dx * gy - dy * gx + dz * gw,
        dw * gw - dx * gx - dy * gy - dz * gz,
    ])
    self._goal_quat_xyzw[arm] = new_quat / max(float(np.linalg.norm(new_quat)), 1e-12)

    if drot_cached is not None and np.any(drot_cached):
        angle = float(np.linalg.norm(drot_cached))
        if angle > 1e-9:
            axis = drot_cached / angle
            s, c = np.sin(angle / 2.0), np.cos(angle / 2.0)
            bx, by, bz, bw = axis[0] * s, axis[1] * s, axis[2] * s, c
            gx, gy, gz, gw = self._goal_quat_xyzw[arm]
            composed = np.array([
                bw * gx + bx * gw + by * gz - bz * gy,
                bw * gy - bx * gz + by * gw + bz * gx,
                bw * gz + bx * gy - by * gx + bz * gw,
                bw * gw - bx * gx - by * gy - bz * gz,
            ])
            self._goal_quat_xyzw[arm] = composed / max(float(np.linalg.norm(composed)), 1e-12)

    # ---- servo current EE toward the (now-updated) goal ----
    q, dq_, J, ee_pos, ee_quat_xyzw, ee_twist = snap
    return self._osc_vel[arm].compute_qdot(
        goal_pos=self._goal_pos[arm],
        goal_quat_xyzw=self._goal_quat_xyzw[arm],
        ee_pos=np.asarray(ee_pos, dtype=np.float64),
        ee_quat_xyzw=np.asarray(ee_quat_xyzw, dtype=np.float64),
        ee_twist=np.asarray(ee_twist, dtype=np.float64),
        J=np.asarray(J, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        q_nullspace_target=self._home_q.get(arm),  # None until home() has run once
        kp=self.kp_gain * OSC_BASE_KP,  # variable impedance, see note below
    )


# ============================================================================
# 8. NEW METHOD -- replaces the old @staticmethod _ee_pd. Absolute EE_POS
#    mode: the action target IS the goal each tick (no goal integration, same
#    as robosuite's use_delta=False path), so this is simpler than #7.
# ============================================================================

def _qdot_ee_pos(
    self,
    arm: str,
    action,  # RobotAction
    snap,  # KinematicSnapshot
    ignore_action: bool = False,
) -> np.ndarray:
    """Replacement for the old `_ee_pd`. No goal integration needed: the
    absolute action target becomes the OSC goal directly, every tick."""
    q, dq_, J, ee_pos, ee_quat_xyzw, ee_twist = snap

    if ignore_action:
        target_pos = np.asarray(ee_pos, dtype=np.float64)
        target_quat_xyzw = np.asarray(ee_quat_xyzw, dtype=np.float64)
    else:
        target_pos = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("x", "y", "z")),
            dtype=np.float64, count=3,
        )
        target_quat_xyzw = np.fromiter(
            (action[f"{arm}_{ax}"] for ax in ("qx", "qy", "qz", "qw")),
            dtype=np.float64, count=4,
        )

    # Keep the persistent goal in sync too, so a mode switch back to
    # EE_DELTA (if that ever happens mid-episode) resumes from here instead
    # of an old stale goal.
    self._goal_pos[arm] = target_pos
    self._goal_quat_xyzw[arm] = target_quat_xyzw

    return self._osc_vel[arm].compute_qdot(
        goal_pos=target_pos,
        goal_quat_xyzw=target_quat_xyzw,
        ee_pos=np.asarray(ee_pos, dtype=np.float64),
        ee_quat_xyzw=np.asarray(ee_quat_xyzw, dtype=np.float64),
        ee_twist=np.asarray(ee_twist, dtype=np.float64),
        J=np.asarray(J, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        q_nullspace_target=self._home_q.get(arm),
        kp=self.kp_gain * OSC_BASE_KP,
    )


# ============================================================================
# 9. Variable impedance wiring, and why kd_gain is NOT used here
# ============================================================================
#
# send_action() already computes self._kp_gain / self._kd_gain from the
# action's "kp"/"kd" terms (exponential scaling, see _KP_GAIN_BASE /
# _KD_GAIN_BASE). OSCVelocityController exposes kp as a per-call override
# (see compute_qdot's `kp=` argument), so variable impedance is just:
#
#   kp=self.kp_gain * OSC_BASE_KP
#
# where OSC_BASE_KP plays the same role EE_PD_KP played for the old PD law --
# add it as a module constant near EE_PD_KP/EE_PD_KD:
#
#   OSC_BASE_KP = 3.5   # matches EE_PD_KP's previous scale; retune if needed
#
# self.kd_gain is intentionally NOT wired into OSCVelocityController: this
# controller's task-space law is pure-P (see osc_velocity_controller.py's
# module docstring for why -- a -kd*twist term is a torque-domain OSC
# artifact that destabilizes a velocity-domain loop). damping_ratio is still
# a valid knob on OSCVelocityController (affects the nullspace/derived-kd
# bookkeeping) but there's no natural mapping from the existing "kd" action
# dimension into this law, so it's simplest to leave kd_gain governing
# nothing in the EE paths (it's still used as before for JOINT_POS mode,
# which is untouched by this patch).
