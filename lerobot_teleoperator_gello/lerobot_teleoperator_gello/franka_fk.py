"""Franka FR3 / Panda forward kinematics via modified DH parameters (Craig convention).

Returns EE position (metres) and orientation (unit quaternion, xyzw convention).
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Modified DH parameters (Craig convention) for Franka FR3/Panda.
# Columns: [a_{i-1} (m), d_i (m), alpha_{i-1} (rad)]
# theta_i is the variable joint angle.
_DH = np.array([
    [ 0.0,     0.333,   0.0      ],  # joint 1
    [ 0.0,     0.0,    -np.pi/2  ],  # joint 2
    [ 0.0,     0.316,   np.pi/2  ],  # joint 3
    [ 0.0825,  0.0,     np.pi/2  ],  # joint 4
    [-0.0825,  0.384,  -np.pi/2  ],  # joint 5
    [ 0.0,     0.0,     np.pi/2  ],  # joint 6
    [ 0.088,   0.0,     np.pi/2  ],  # joint 7
    [ 0.0,     0.107+0.1034,   0.0      ],  # flange-to-EE (fixed)
])


def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Modified DH (Craig) transform from frame i-1 to frame i."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,    -st,     0.0,    a   ],
        [st*ca,  ct*ca, -sa,    -d*sa],
        [st*sa,  ct*sa,  ca,     d*ca],
        [0.0,    0.0,    0.0,   1.0  ],
    ])


def franka_fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Franka FR3/Panda EE pose from 7 joint angles.

    Args:
        q: Joint angles in radians, shape (7,), in the Franka joint convention.

    Returns:
        pos: EE Cartesian position [x, y, z] in metres relative to the robot base.
        quat_xyzw: EE orientation as a unit quaternion [qx, qy, qz, qw].
    """
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _DH[i]
        T = T @ _dh_matrix(a, d, alpha, float(q[i]))
    # Fixed flange-to-EE transform (theta=0).
    a, d, alpha = _DH[7]
    T = T @ _dh_matrix(a, d, alpha, 0.0)

    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def franka_jacobian(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian (6×7) for Franka FR3/Panda, expressed in the base frame.

    Uses the same modified DH parameters as franka_fk / franka_fk_chain so the
    result is guaranteed to be consistent with the FK.  Avoids libfranka's
    robot.model.zero_jacobian, which returns zeros through the CBRobot wrapper.

    Returns:
        J: (6, 7) ndarray.  Rows 0-2 are linear velocity (m/s), rows 3-5 are
           angular velocity (rad/s).  J @ dq gives the EE twist in the base frame.
    """
    chain = franka_fk_chain(q)      # out[i] = base→frame{i+1}, out[7] = base→EE
    p_ee = chain[7][:3, 3]          # EE position in base frame

    J = np.zeros((6, 7), dtype=np.float64)
    for i in range(7):
        # z-axis and origin of the frame BEFORE joint i+1 (0-indexed)
        if i == 0:
            z = np.array([0.0, 0.0, 1.0])   # base frame z-axis
            p = np.zeros(3)
        else:
            z = chain[i - 1][:3, 2]          # z-axis of frame{i} in base frame
            p = chain[i - 1][:3, 3]          # origin of frame{i} in base frame
        J[:3, i] = np.cross(z, p_ee - p)    # linear velocity contribution
        J[3:, i] = z                         # angular velocity contribution
    return J


def franka_fk_chain(q: np.ndarray) -> np.ndarray:
    """Per-link cumulative transforms base→j1, base→j2, ..., base→EE.

    Returns:
        Shape (8, 4, 4). Entries 0..6 are the frames after each of the 7
        joint rotations; entry 7 includes the fixed flange-to-EE step.
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
