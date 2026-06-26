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
    [0.0, 0.107+0.1034, 0.0],
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


def franka_fk_chain(q: np.ndarray) -> np.ndarray:
    """Per-link cumulative transforms base→frame{1..7} plus base→EE.

    Returns shape (8, 4, 4).  out[i] = cumulative base→frame{i+1} for i<7;
    out[7] includes the fixed flange-to-EE step.
    """
    out = np.empty((8, 4, 4), dtype=np.float64)
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _DH[i]
        T = T @ _dh_matrix(a, d, alpha, float(q[i]))
        out[i] = T
    a, d, alpha = _DH[7]
    T = T @ _dh_matrix(a, d, alpha, 0.0)
    out[7] = T
    return out


def franka_jacobian(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian (6×7) expressed in the base frame.

    Returns J such that [v; omega] = J @ dq, with v and omega in the
    base frame.  Rows 0-2 are linear velocity (m/s), rows 3-5 angular (rad/s).

    Uses the local DH parameters — no dependency on libfranka or the CBRobot
    wrapper (which returns zeros for robot.model.zero_jacobian via RPyC).
    """
    chain = franka_fk_chain(q)   # out[i] = base→frame{i+1}, out[7] = base→EE
    p_ee = chain[7][:3, 3]

    J = np.zeros((6, 7), dtype=np.float64)
    for i in range(7):
        if i == 0:
            z = np.array([0.0, 0.0, 1.0])
            p = np.zeros(3)
        else:
            z = chain[i - 1][:3, 2]
            p = chain[i - 1][:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J
