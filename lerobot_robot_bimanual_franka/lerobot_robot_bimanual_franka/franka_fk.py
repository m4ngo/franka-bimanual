"""Franka FR3 / Panda forward kinematics (modified DH, Craig convention).

Mirrors `lerobot_teleoperator_gello.franka_fk` so the robot package can resolve
saved joint-space home poses to EE targets without depending on the teleop
wheel stack. Keep the two files aligned if DH parameters change.
"""

import numpy as np
from scipy.spatial.transform import Rotation

_DH = np.array([
    [0.0, 0.333, 0.0],
    [0.0, 0.0, -np.pi / 2],
    [0.0, 0.316, np.pi / 2],
    [0.0825, 0.0, np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0, 0.0, np.pi / 2],
    [0.088, 0.0, np.pi / 2],
    [0.0, 0.107, 0.0],
])


def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def franka_fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EE pose from 7 joint angles: position (m), unit quaternion xyzw."""
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _DH[i]
        T = T @ _dh_matrix(a, d, alpha, float(q[i]))
    a, d, alpha = _DH[7]
    T = T @ _dh_matrix(a, d, alpha, 0.0)
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()
