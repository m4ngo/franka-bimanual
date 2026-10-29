"""The workstation half of robosuite's OSC: ``set_goal``.

``osc_torque_controller.OSCTorqueController`` is the other half, ``run_controller``,
and it runs on the NUC at 500 Hz. This module is what turns one policy step's
action into the goal pose that controller holds between steps, and it is
deliberately the *whole* of that job -- pose composition, the delta envelope and
the latched goal orientation -- so the port can be read against
``multi-fast/robosuite/robosuite/controllers/osc.py`` in one sitting.

Pure and stateless apart from ``goal_ori``, which osc.py is stateful about too and
for the same reason: see :meth:`OSCGoalBuilder.from_delta`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .osc_torque_controller import clip_delta

# (pos(3), quat_xyzw(4)) in the arm's base frame -- what move_osc_goal_batch takes.
Goal = tuple[np.ndarray, np.ndarray]


def delta_rotvec(dquat_xyzw: np.ndarray) -> np.ndarray:
    """Delta quaternion (xyzw) -> axis-angle, the representation osc.py's
    set_goal_orientation takes. A degenerate all-zero quat means no rotation."""
    norm = float(np.linalg.norm(dquat_xyzw))
    if norm < 1e-9:
        return np.zeros(3)
    return Rotation.from_quat(dquat_xyzw / norm).as_rotvec()


class OSCGoalBuilder:
    """Composes each arm's OSC goal pose, holding osc.py's latched ``goal_ori``.

    One instance per robot; arms are keyed by the ``l_``/``r_`` prefix and share
    the configured trims, which are per-rig rather than per-arm.
    """

    def __init__(
        self,
        translation_fudge: float,
        rotation_fudge: float,
    ) -> None:
        # Sim-to-real delta scaling. Applied to the axis-angle rotation delta, NOT
        # to the quaternion: scaling all four components uniformly is undone by
        # normalisation. 1.0 on both is exactly what the policy emitted.
        self._trans_fudge = float(translation_fudge)
        self._rot_fudge = float(rotation_fudge)
        # osc.py's goal_ori: an ABSOLUTE orientation carried across steps. See
        # from_delta for why it is not recomputed every step.
        self._goal_ori: dict[str, Rotation] = {}

    def reset(self, arm: str, ee_quat_xyzw: np.ndarray) -> None:
        """osc.py reset_goal(): park the held orientation on the current pose, so
        the next delta is relative to where the arm actually is."""
        self._goal_ori[arm] = Rotation.from_quat(np.asarray(ee_quat_xyzw, dtype=np.float64))

    def from_delta(
        self,
        arm: str,
        delta_pos: np.ndarray,
        delta_rotvec_: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat_xyzw: np.ndarray,
    ) -> Goal:
        """osc.py set_goal() with ``use_delta=True``.

        goal_pos is rebuilt from the *current* EE pose every policy step and never
        accumulated onto the previous goal -- that is what makes a released
        joystick or a zero-delta policy step hold position instead of drifting.

        goal_ori is the opposite, and the asymmetry is osc.py's: it is rewritten
        ONLY when the commanded rotation delta is nonzero (osc.py tests it with
        ``math.isclose(elem, 0.0)`` -- exact zero). Re-anchoring it every step
        would make the orientation error identically zero on a pure-translation
        command, so nothing would hold the EE's orientation and it would tumble as
        the arm translates.
        """
        dpos = np.asarray(delta_pos, dtype=np.float64)
        drot = np.asarray(delta_rotvec_, dtype=np.float64)

        rotation_commanded = bool(np.any(drot))

        # Clip to the envelope a policy could have emitted, THEN apply the hardware
        # fudge. The other order lets clip_delta eat the fudge -- at tf=3 a 0.05 m
        # delta became 0.15 m and was clipped straight back to 0.05, so any fudge
        # above 1.0 was a silent no-op.
        dpos, drot = clip_delta(dpos, drot)
        dpos = dpos * self._trans_fudge
        drot = drot * self._rot_fudge

        if arm not in self._goal_ori or rotation_commanded:
            self._goal_ori[arm] = Rotation.from_rotvec(drot) * Rotation.from_quat(ee_quat_xyzw)
        return np.asarray(ee_pos, dtype=np.float64) + dpos, self._goal_ori[arm].as_quat()

    @staticmethod
    def absolute(goal_pos: np.ndarray, goal_quat_xyzw: np.ndarray) -> Goal:
        """EE_POS: the action already carries an absolute pose, so it becomes the
        OSC goal directly -- no envelope, because a pose is not a step, and no
        latched orientation, because the caller supplies one every step."""
        quat = np.asarray(goal_quat_xyzw, dtype=np.float64)
        return (
            np.asarray(goal_pos, dtype=np.float64),
            quat / max(float(np.linalg.norm(quat)), 1e-12),
        )

    @staticmethod
    def delta_from_absolute(
        goal_pos: np.ndarray,
        goal_quat_xyzw: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The arithmetic inverse of ``from_delta``'s pose composition.

        Ignores clip_delta and the fudges -- neither is invertible -- so
        this is the delta a bare policy step would have needed, not a replay of
        what from_delta applies.
        """
        dpos = np.asarray(goal_pos, dtype=np.float64) - np.asarray(ee_pos, dtype=np.float64)
        drot = (
            Rotation.from_quat(np.asarray(goal_quat_xyzw, dtype=np.float64))
            * Rotation.from_quat(np.asarray(ee_quat_xyzw, dtype=np.float64)).inv()
        ).as_rotvec()
        return dpos, drot

    @staticmethod
    def offset(goal: Goal, offset_pos: np.ndarray, offset_rotvec: np.ndarray) -> Goal:
        """Add a residual correction (``BimanualFranka.cache_delta``) to an
        ABSOLUTE goal.

        The delta path does not use this: there the residual is summed into the
        delta before ``from_delta`` sees it, so it passes through ``clip_delta``
        and counts toward "was a rotation commanded" like any other delta. An
        absolute pose has no envelope to pass through, so it is composed here.
        """
        pos, quat = goal
        rot = Rotation.from_rotvec(np.asarray(offset_rotvec, dtype=np.float64))
        return pos + np.asarray(offset_pos, dtype=np.float64), (rot * Rotation.from_quat(quat)).as_quat()
