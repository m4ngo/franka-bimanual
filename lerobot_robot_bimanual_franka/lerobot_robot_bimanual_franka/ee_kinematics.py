"""Joint angles -> Cartesian O_T_EE pose.

franka_fk returns the FLANGE; O_T_EE is that frame rotated -45 deg about its
own z by the Franka Hand mount. Position is unaffected. Verified to 1.2e-7 rad
against the excite_panda datasets, which record O_T_EE directly.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .franka_fk import _DH

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


def _dh_batch(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """franka_fk._dh_matrix over a batch of joint angles, (N,) -> (N, 4, 4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros((len(theta), 4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = ct, -st, a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st * ca, ct * ca, -sa, -d * sa
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = st * sa, ct * sa, ca, d * ca
    T[:, 3, 3] = 1.0
    return T


def eef_poses_from_qpos(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """qpos (N, num_joints) -> (eef_pos (N,3), eef_quat_xyzw (N,4)), both O_T_EE.

    franka_fk over every row at once; the same chain, so the same numbers.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    T = np.broadcast_to(np.eye(4), (len(qpos), 4, 4))
    for i in range(len(_DH) - 1):
        T = T @ _dh_batch(*_DH[i], qpos[:, i])
    T = T @ _dh_batch(*_DH[-1], np.zeros(len(qpos)))
    quat = Rotation.from_matrix(T[:, :3, :3]).as_quat() if len(qpos) else np.empty((0, 4))
    return T[:, :3, 3].copy(), flange_quat_to_o_t_ee(quat)
