"""Action-screening safety overlays for the bimanual Franka.

The safety screen sits between the parent's PD controller and the per-arm
subprocesses. Every action passes through it before being streamed to franky,
so the same kinematic snapshot that drives the PD also informs the safety
checks.

All checks are pure functions of (action, kinematic_state) and return a new
action of the same shape, so they compose left-to-right: each subsequent
check sees the previous check's output. This keeps the pipeline easy to
reason about and easy to extend with new checks.

Two checks are implemented or stubbed:

- Worktable brake: prevents the EE from impacting a horizontal worktable at
  WORKTABLE_HEIGHT in the arm's base frame. A stopping-distance envelope plus
  overspeed override keep fast / momentum-laden commands safe.
- Bimanual repel (TODO): pushes the arms apart when their EEs get too close.

The kinematic snapshot is the tuple returned by
MultiRobotWrapper.current_kinematic_state_batch:
    (q, dq, ee_translation, jacobian)
all read from the same robot.state and therefore mutually consistent.
"""

import numpy as np

# --- Worktable safety -------------------------------------------------------
# Two layers stack to brake the downward (toward-the-table) component of
# the commanded EE motion:
#   1. Kinematic stopping-distance envelope (always active): caps the
#      commanded downward speed at sqrt(2 * MAX_DECEL * (dist - MIN)) so
#      the arm can always decelerate before reaching the table given the
#      assumed max deceleration. This is what makes fast commands safe.
#   2. Actual-velocity overspeed override: if the *measured* downward speed
#      exceeds the envelope (e.g. due to momentum from a prior fast command),
#      the commanded EE downward velocity is forced to zero so the arm
#      brakes maximally.
# In EE-twist mode lateral / upward / angular motion passes through
# unmodified. In joint-velocity mode the entire dq vector is uniformly
# scaled by the same ratio used to brake v_z, so braking the descent does
# not expose any lateral motion that was incidentally coupled with the
# downward joint command (see _apply_worktable_brake for the rationale).
WORKTABLE_HEIGHT = 0.12  # meters
# Extra vertical reach below the Franka EE frame for the custom end-effector.
# Set this to the added tool length, in meters. The same value is used for
# every arm, assuming the bimanual Frankas have identical end-effectors.
CUSTOM_END_EFFECTOR_Z_EXTENSION = 0.18
WORKTABLE_DISTANCE_MIN = 0.03  # meters; minimum closeness to the table; downward velocity is forced to zero at/past this distance
# Maximum deceleration (m/s^2) we assume the arm can deliver. Used by the
# kinematic envelope: smaller values are more conservative (larger braking
# zone, lower allowed approach speed) and less prone to overshoot.
WORKTABLE_MAX_DECEL = 0.5
# Commanded EE z-velocities with magnitude below this threshold are treated
# as "no descent" and the brake is skipped. Catches floating-point noise
# from e.g. Jacobian-pseudoinverse joint commands whose intended z-velocity
# was zero -- without this the joint-mode uniform scaling could halt the
# arm on what was supposed to be a purely lateral command.
WORKTABLE_VELOCITY_EPS = 1.0e-4  # m/s

# --- Bimanual safety (TODO) -------------------------------------------------
# Repel the two arms when their EEs get too close to each other. Currently
# stubbed; constants kept here so all safety knobs live in one place.
# Left arm's position relative to the right arm's base.
RELATIVE_ARM_POSITION_ROTATION = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
BIMANUAL_DISTANCE_THRESHOLD = 0.05  # meters; how close can arms be before starting to slow?
BIMANUAL_DISTANCE_MIN = 0.02  # meters; minimum closeness between arms; repel immediately at/past this distance
BIMANUAL_REPEL_FORCE = 1.0

# --- Joint-velocity ceiling -------------------------------------------------
# Hardware-safe ceiling on the L2 norm of the 7-DoF joint-velocity command.
# Applied AFTER all per-arm checks so they can never push past it.
JOINT_VELOCITY_MAX = 2.0  # rad/s


# Type alias for the kinematic snapshot returned by
# MultiRobotWrapper.current_kinematic_state(_batch). Documented here so
# callers don't have to spell the tuple out everywhere.
# Layout: (q, dq, ee_translation, jacobian)
#   q              : (7,)   joint positions     (rad)
#   dq             : (7,)   joint velocities    (rad/s)
#   ee_translation : (3,)   EE position in base (m)
#   jacobian       : (6, 7) base-frame Jacobian
KinematicSnapshot = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ActionSafetyScreen:
    """Screen actions before they are dispatched to the arm subprocesses.

    All checks are pure functions of (action, kinematic_state) and return a
    new action of the same shape. Composition is a left-to-right pipe inside
    `screen_ee_actions` / `screen_joint_actions` so adding a new check is a
    one-line edit there.
    """

    def __init__(
        self,
        end_effector_z_extension: float = CUSTOM_END_EFFECTOR_Z_EXTENSION,
    ) -> None:
        if end_effector_z_extension < 0.0:
            raise ValueError("end_effector_z_extension must be non-negative.")
        self.end_effector_z_extension = float(end_effector_z_extension)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def screen_ee_actions(
        self,
        ee_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
    ) -> dict[str, np.ndarray]:
        """Apply worktable / bimanual safety to EE-twist actions."""
        ee_by_arm = self._apply_worktable_brake(ee_by_arm, kin_state, is_ee=True)
        # ee_by_arm = self._apply_bimanual_repel(ee_by_arm, kin_state, is_ee=True)  # TODO
        return ee_by_arm

    def screen_joint_actions(
        self,
        joint_velocities_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
    ) -> dict[str, np.ndarray]:
        """Apply worktable / bimanual safety to joint-velocity actions."""
        joint_velocities_by_arm = self._apply_worktable_brake(
            joint_velocities_by_arm, kin_state, is_ee=False
        )
        # joint_velocities_by_arm = self._apply_bimanual_repel(joint_velocities_by_arm, kin_state, is_ee=False)  # TODO
        return joint_velocities_by_arm

    @staticmethod
    def clamp_joint_velocity_magnitude(velocity: np.ndarray) -> np.ndarray:
        """Scale a 7-DoF velocity vector down to JOINT_VELOCITY_MAX in L2 norm.

        Applied as the final stage of the joint pipeline, after every other
        screen, so no upstream check can push the command past the hardware-
        safe ceiling.
        """
        norm = float(np.linalg.norm(velocity))
        if norm > JOINT_VELOCITY_MAX:
            return velocity * (JOINT_VELOCITY_MAX / norm)
        return velocity

    # ------------------------------------------------------------------
    # Worktable brake
    # ------------------------------------------------------------------
    def _apply_worktable_brake(
        self,
        action_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
        is_ee: bool,
    ) -> dict[str, np.ndarray]:
        """Brake motion that drives the EE toward the worktable.

        Two layers compute a target downward EE Z-speed in the arm's base
        frame:

            v_target_z = max(v_target_z, -v_envelope)        # envelope cap
            if -v_actual_z > v_envelope:                     # overspeed
                v_target_z = max(v_target_z, 0.0)            # active brake

        How the action is shaped to hit v_target_z differs by mode:

        - is_ee=True (EE-twist action): the linear-z entry of the action is
          overwritten with v_target_z. Lateral, upward, and angular
          components pass through unmodified -- the operator commanded each
          axis explicitly, so preserving them respects intent and lets the
          arm slide along or pull away from the table near the limit.

        - is_ee=False (joint-velocity action): the entire 7-DoF velocity
          vector is uniformly scaled by v_target_z / v_commanded_z. This
          preserves the commanded joint-space direction (and therefore the
          EE-space direction) while slowing the motion proportionally as
          the arm approaches the limit. Uniform scaling is what prevents
          the arm from drifting laterally when the joint command was
          supposed to be "purely downward" but the teleop's joint mapping
          has incidental lateral coupling: any lateral component that came
          along with the down command is braked at the same rate, instead
          of being preserved while only z is zeroed.

        Trade-off in joint mode: an operator who wants to slide along the
        table near the limit must command joint targets that produce
        ~zero EE z-velocity so the brake does not engage; a downward-
        plus-lateral joint command will see both components slowed.
        """
        out: dict[str, np.ndarray] = {}
        for arm, action in action_by_arm.items():
            action = np.asarray(action, dtype=np.float64)
            _, dq, ee_translation, jacobian = kin_state[arm]
            ee_z = float(np.asarray(ee_translation)[2])
            contact_z = self._worktable_contact_z(ee_z)
            jacobian = np.asarray(jacobian, dtype=np.float64)
            dq = np.asarray(dq, dtype=np.float64)

            # Kinematic stopping-distance envelope. This bounds downward
            # speed by sqrt(2 * MAX_DECEL * (dist - MIN)), so the arm can
            # decelerate before reaching WORKTABLE_DISTANCE_MIN.
            safe_dist = contact_z - WORKTABLE_HEIGHT - WORKTABLE_DISTANCE_MIN
            v_envelope = (
                0.0
                if safe_dist <= 0.0
                else float(np.sqrt(2.0 * WORKTABLE_MAX_DECEL * safe_dist))
            )
            v_actual_z = float(jacobian[2, :] @ dq)
            if is_ee:
                v_commanded_z = float(action[2])
            else:
                v_commanded_z = float(jacobian[2, :] @ action)

            # Pass through any command that isn't a meaningful descent. The
            # epsilon catches floating-point noise (e.g. a "purely lateral"
            # joint command may carry a v_z of ~1e-17 from pinv math) so
            # the joint-mode uniform scaling below doesn't halt the arm on
            # numerical jitter.
            if v_commanded_z >= -WORKTABLE_VELOCITY_EPS:
                out[arm] = action
                continue

            v_target_z = v_commanded_z
            # Layer 1: kinematic stopping-distance envelope.
            if v_target_z < -v_envelope:
                v_target_z = -v_envelope
            # Layer 2: overspeed override on measured velocity.
            if -v_actual_z > v_envelope:
                v_target_z = max(v_target_z, 0.0)

            if v_target_z == v_commanded_z:
                # No brake layer changed the command.
                out[arm] = action
                continue

            if is_ee:
                action = action.copy()
                action[2] = v_target_z
            else:
                # Uniform scale that drives v_z to v_target_z while
                # preserving joint-space direction. v_commanded_z is
                # safely below -EPS thanks to the early-out above, so the
                # division is well-defined and the ratio lies in [0, 1].
                scale = max(0.0, v_target_z / v_commanded_z)
                action = action * scale

            out[arm] = action
        return out

    def _worktable_contact_z(self, ee_z: float) -> float:
        """Lowest relevant tool height used by the worktable safety checks."""
        return ee_z - self.end_effector_z_extension

    # ------------------------------------------------------------------
    # Bimanual repel (TODO)
    # ------------------------------------------------------------------
    # def _apply_bimanual_repel(
    #     self,
    #     action_by_arm: dict[str, np.ndarray],
    #     kin_state: dict[str, KinematicSnapshot],
    #     is_ee: bool,
    # ) -> dict[str, np.ndarray]:
    #     """Push the arms apart when their EEs get too close. Returns
    #     modified actions with the same shape as the input. Not yet
    #     implemented.
    #     """
    #     pass
