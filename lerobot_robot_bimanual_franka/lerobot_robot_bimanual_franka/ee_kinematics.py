"""Joint angles -> Cartesian O_T_EE pose.

franka_fk returns the FLANGE; O_T_EE is that frame rotated -45 deg about its
own z by the Franka Hand mount. Position is unaffected. Verified to 1.2e-7 rad
against the excite_panda datasets, which record O_T_EE directly.
"""

from __future__ import annotations

import numpy as np

from .franka_fk import franka_fk

_FLANGE_TO_EE_QUAT_XYZW = np.array([0.0, 0.0, -np.sin(np.pi / 8), np.cos(np.pi / 8)])


def flange_quat_to_o_t_ee(quat_xyzw: np.ndarray) -> np.ndarray:
    """Post-multiply each row by the tool-frame Hand offset (Hamilton, xyzw)."""
    x1, y1, z1, w1 = quat_xyzw.T
    x2, y2, z2, w2 = _FLANGE_TO_EE_QUAT_XYZW
    out = np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def eef_poses_from_qpos(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """qpos (N, num_joints) -> (eef_pos (N,3), eef_quat_xyzw (N,4)), both O_T_EE."""
    fk = [franka_fk(q) for q in qpos]
    pos = np.array([p for p, _ in fk], dtype=np.float64)
    quat = flange_quat_to_o_t_ee(np.array([q for _, q in fk], dtype=np.float64))
    return pos, quat
