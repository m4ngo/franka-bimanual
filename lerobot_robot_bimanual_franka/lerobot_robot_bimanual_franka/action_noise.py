"""Action noise, added to a goal just before it leaves the workstation.

Leaders and policies emit clean actions, and the action a recording stores is
always that clean one: only the goal the arm pursues carries the noise. It is
applied to the composed goal -- after the delta envelope, the fudges and the
residual, before the worktable screen, so the floor still bounds a noisy goal.

Every scale defaults to 0.0 (off) in the robot configs. ``control.yaml``'s
``noise:`` block holds the scales a run turns on (``configured_scales``).
"""

from __future__ import annotations

import franka_config as fc
import numpy as np
from scipy.spatial.transform import Rotation

# (pos(3), quat_xyzw(4)) in the arm's base frame -- what move_osc_goal_batch takes.
Goal = tuple[np.ndarray, np.ndarray]


def configured_scales() -> dict[str, float]:
    """control.yaml's ``noise:`` scales as robot config overrides, for a run
    that adds noise: ``SingleArmFrankaConfig(**configured_scales())``."""
    return {
        "noise_pos_scale": float(fc.control("noise.pos_scale_m")),
        "noise_rot_scale": float(fc.control("noise.rot_scale_rad")),
        "noise_joint_scale": float(fc.control("noise.joint_scale_rad")),
    }


class ActionNoise:
    """Zero-mean Gaussian noise on an OSC goal pose or a joint-position goal,
    std dev per axis per policy step. A zero scale returns its input untouched
    and draws nothing."""

    def __init__(self, pos_scale: float, rot_scale: float, joint_scale: float) -> None:
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.joint_scale = float(joint_scale)

    def pose(self, goal: Goal) -> Goal:
        pos, quat = goal
        if self.pos_scale > 0.0:
            pos = pos + np.random.normal(0.0, self.pos_scale, 3)
        if self.rot_scale > 0.0:
            rot = Rotation.from_euler("xyz", np.random.normal(0.0, self.rot_scale, 3))
            quat = (rot * Rotation.from_quat(quat)).as_quat()
        return pos, quat

    def joints(self, q: np.ndarray) -> np.ndarray:
        if self.joint_scale > 0.0:
            return q + np.random.normal(0.0, self.joint_scale, q.shape)
        return q
