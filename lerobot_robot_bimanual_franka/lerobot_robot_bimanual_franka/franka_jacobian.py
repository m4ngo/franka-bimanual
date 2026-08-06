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


# Constant per-row DH terms, hoisted so the per-tick path only evaluates the
# joint-angle-dependent entries. This runs inside the NUC's 1 kHz loop, where
# the original per-call `np.array([[...]])` construction cost ~150 us/tick and
# was a large part of what pushed the loop past its deadline.
_A = np.ascontiguousarray(_DH[:, 0])
_D = np.ascontiguousarray(_DH[:, 1])
_CA = np.cos(_DH[:, 2])
_SA = np.sin(_DH[:, 2])

_N_LINKS = _DH.shape[0]

# Everything in the DH transform that does not depend on theta.
_TEMPLATE = np.zeros((_N_LINKS, 4, 4), dtype=np.float64)
_TEMPLATE[:, 0, 3] = _A
_TEMPLATE[:, 1, 2] = -_SA
_TEMPLATE[:, 1, 3] = -_D * _SA
_TEMPLATE[:, 2, 2] = _CA
_TEMPLATE[:, 2, 3] = _D * _CA
_TEMPLATE[:, 3, 3] = 1.0


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
    theta = np.zeros(_N_LINKS, dtype=np.float64)
    theta[:7] = np.asarray(q, dtype=np.float64)[:7]
    ct, st = np.cos(theta), np.sin(theta)

    # Build all 8 link transforms at once, then chain them.
    b = _TEMPLATE.copy()
    b[:, 0, 0] = ct
    b[:, 0, 1] = -st
    b[:, 1, 0] = st * _CA
    b[:, 1, 1] = ct * _CA
    b[:, 2, 0] = st * _SA
    b[:, 2, 1] = ct * _SA

    out = np.empty((_N_LINKS, 4, 4), dtype=np.float64)
    out[0] = b[0]
    for i in range(1, _N_LINKS):
        np.matmul(out[i - 1], b[i], out=out[i])
    return out


def zero_jacobian(q: np.ndarray, ee_pos_base: np.ndarray | None = None) -> np.ndarray:
    """Base-frame geometric 6x7 Jacobian [linear(3); angular(3)] at the EE.

    ``ee_pos_base`` anchors the linear block (pass the measured O_T_EE
    translation to stay consistent with the measured-pose error). If omitted, the
    FK's own EE point is used. Validated == finite-difference of the FK.
    """
    chain = fk_chain(q)
    o_e = np.asarray(ee_pos_base, dtype=np.float64) if ee_pos_base is not None else chain[7][:3, 3]
    z = chain[:7, :3, 2]           # joint k screw axis = z of frame k+1
    o = chain[:7, :3, 3]           # origin of frame k+1
    J = np.empty((6, 7), dtype=np.float64)
    J[:3, :] = np.cross(z, o_e - o).T
    J[3:, :] = z.T
    return J
