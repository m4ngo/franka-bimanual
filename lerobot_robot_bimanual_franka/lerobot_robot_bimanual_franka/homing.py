"""Joint-space homing: the speed budget, and the ramp that respects it.

Homing always runs server-side joint impedance -- the only law that reaches a
joint configuration directly -- in every control mode. It is not screened by
`safety.ActionSafetyScreen`: the worktable floor bounds an EE goal pose, and a
saved home configuration has none.

Split out of `bimanual_franka` because the budget below is a closed-form result
about the joint-impedance gains, and reading it wedged between camera plumbing
and OSC goal composition is how it came to be re-derived twice in one method.
"""

from __future__ import annotations

import numpy as np

import franka_config as fc  # type: ignore

from .osc_torque_controller import DEFAULT_JOINT_KD, DEFAULT_JOINT_KP, JOINT_TORQUE_LIMITS

MAX_QDOT = fc.control("homing.max_qdot_rad_s")        # ramp rate of the commanded goal
SETTLE_QDOT = fc.control("homing.settle_qdot_rad_s")  # home() is not done until this still
LEAD_MARGIN = fc.control("homing.lead_margin")        # keeps the stall clamp off the ramp
# Fraction of each joint's torque clamp that homing may spend on kp*lead + kd*qdot
# together. Both terms scale with the speed, so this caps the speed per joint: the
# wrist (kd 10 against a 20 Nm clamp) cannot be damped at MAX_QDOT at all, and a
# saturated joint stops tracking the ramp and never converges.
TAU_FRACTION = fc.control("homing.tau_fraction")

IMPEDANCE_KP = DEFAULT_JOINT_KP
IMPEDANCE_KD = DEFAULT_JOINT_KD


def speed_budget() -> np.ndarray:
    """Per-joint cap on the commanded ramp rate (rad/s).

    kp*lead + kd*qdot = qdot*kd*(1 + LEAD_MARGIN) at the stall lead, so bounding
    that sum by TAU_FRACTION of the clamp fixes the speed.
    """
    return np.minimum(
        MAX_QDOT,
        TAU_FRACTION * np.asarray(JOINT_TORQUE_LIMITS) / (IMPEDANCE_KD * (1.0 + LEAD_MARGIN)),
    )


def max_lead() -> np.ndarray:
    """How far the commanded goal may run ahead of the measured q.

    A stall guard, not the speed limit -- the ramp is. Sustaining `speed_budget()`
    needs exactly qdot/(kp/kd) of lead, so this must sit above it with margin or it
    binds every tick and re-introduces the sawtooth the ramp exists to remove.
    """
    return LEAD_MARGIN * speed_budget() * IMPEDANCE_KD / IMPEDANCE_KP


class HomingRamp:
    """The commanded goal, ramped toward the target and re-anchored on the arm.

    RAMPS rather than re-deriving the goal from the measured q each tick: the
    latter sawtooths the error by v/rate, a 25% torque ripple felt as vibration.
    """

    def __init__(self, start_q: dict[str, np.ndarray], targets_q: dict[str, np.ndarray], rate_hz: float):
        self._goal = {arm: np.asarray(q, dtype=np.float64).copy() for arm, q in start_q.items()}
        self._targets = targets_q
        self._step = speed_budget() / float(rate_hz)
        self._max_lead = max_lead()

    def advance(self, measured_q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One tick of the ramp; returns the goal to command per arm."""
        for arm in list(self._goal):
            goal = self._goal[arm]
            lead = goal + np.clip(self._targets[arm] - goal, -self._step, self._step)
            q = measured_q[arm]
            # Re-anchors only once the arm has actually fallen behind.
            self._goal[arm] = q + np.clip(lead - q, -self._max_lead, self._max_lead)
        return self._goal
