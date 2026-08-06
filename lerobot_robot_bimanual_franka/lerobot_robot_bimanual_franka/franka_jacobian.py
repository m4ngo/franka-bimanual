"""Analytic Franka FR3/Panda geometric Jacobian, computed on the workstation.

franky's ``model.zero_jacobian`` returns all zeros on this net_franky build
(verified on hardware: every frame/state/overload -> norm 0, while q and O_T_EE
are valid). The joint_ik loop needs a real Jacobian, so we compute the base-frame
geometric 6x7 Jacobian ourselves from the measured joint angles using the same
modified-DH (Craig) chain as ``lerobot_teleoperator_gello.franka_fk``.

Validated to machine precision against a finite-difference of the FK: joint k's
screw axis is the z-axis of frame k+1 in the chain (``chain[k][:3,2]``) with
origin ``chain[k][:3,3]`` -- the standard modified-DH result. The linear block is
``z_k x (o_e - o_k)`` where ``o_e`` is the EE point; the angular block is ``z_k``.

The EE point ``o_e`` is supplied by the caller (the measured franky O_T_EE
translation), NOT the FK's own tool point, so the Jacobian stays consistent with
the pose error the IK loop forms from the measured pose. The geometric Jacobian's
joint axes/origins are tool-independent, so anchoring its linear part on the
measured EE is exact.

Quaternion/rotation conventions follow the rest of the stack; this module only
produces a base-frame geometric Jacobian (linear-on-top, angular-on-bottom),
matching the sim/isaaclab and franky ``zero_jacobian`` row ordering.
"""

from __future__ import annotations

import numpy as np

# Modified DH parameters (Craig convention) for Franka FR3/Panda, same as
# lerobot_teleoperator_gello.franka_fk. Columns: [a_{i-1} (m), d_i (m), alpha_{i-1}].
_DH = np.array([
    [ 0.0,     0.333,          0.0      ],  # joint 1
    [ 0.0,     0.0,           -np.pi/2  ],  # joint 2
    [ 0.0,     0.316,          np.pi/2  ],  # joint 3
    [ 0.0825,  0.0,            np.pi/2  ],  # joint 4
    [-0.0825,  0.384,         -np.pi/2  ],  # joint 5
    [ 0.0,     0.0,            np.pi/2  ],  # joint 6
    [ 0.088,   0.0,            np.pi/2  ],  # joint 7
    [ 0.0,     0.107 + 0.1034, 0.0      ],  # flange-to-EE (fixed)
])


def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Modified DH (Craig) transform from frame i-1 to frame i."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,    -st,     0.0,    a    ],
        [st*ca,  ct*ca, -sa,    -d*sa ],
        [st*sa,  ct*sa,  ca,     d*ca ],
        [0.0,    0.0,    0.0,    1.0  ],
    ])


def fk_chain(q: np.ndarray) -> np.ndarray:
    """Per-link cumulative base->frame transforms. Shape (8, 4, 4).

    Entries 0..6 are the frames after each of the 7 joint rotations
    (``out[k] = 0_T_(k+1)``); entry 7 includes the fixed flange-to-EE step.
    """
    out = np.empty((8, 4, 4), dtype=np.float64)
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _DH[i]
        T = T @ _dh_matrix(a, d, alpha, float(q[i]))
        out[i] = T
    a, d, alpha = _DH[7]
    out[7] = T @ _dh_matrix(a, d, alpha, 0.0)
    return out


def zero_jacobian(q: np.ndarray, ee_pos_base: np.ndarray | None = None) -> np.ndarray:
    """Base-frame geometric 6x7 Jacobian [linear(3); angular(3)] at the EE.

    ``ee_pos_base`` anchors the linear block (pass the measured franky O_T_EE
    translation to stay consistent with the measured-pose error). If omitted, the
    FK's own EE point is used. Validated == finite-difference of the FK.
    """
    chain = fk_chain(np.asarray(q, dtype=np.float64))
    o_e = np.asarray(ee_pos_base, dtype=np.float64) if ee_pos_base is not None else chain[7][:3, 3]
    J = np.zeros((6, 7), dtype=np.float64)
    for k in range(7):
        z_k = chain[k][:3, 2]      # joint k screw axis = z of frame k+1
        o_k = chain[k][:3, 3]      # origin of frame k+1
        J[:3, k] = np.cross(z_k, o_e - o_k)
        J[3:, k] = z_k
    return J