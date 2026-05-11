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
    [ 0.0,     0.107,   0.0      ],  # flange-to-EE (fixed)
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
