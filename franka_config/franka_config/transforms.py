"""Rigid-transform helpers.

Every quaternion in config/ is wxyz (scalar first). scipy is scalar-LAST, so
the two conversion helpers here are the only place the orders may be mixed —
confusing them is what produced the 180-degree sim-alignment error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_wxyz_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(r, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return (float(w), float(x), float(y), float(z))


def quat_wxyz_to_xyzw(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)


def quat_xyzw_to_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in q)
    return (w, x, y, z)


@dataclass(frozen=True)
class Pose:
    """A rigid transform: p_parent = rotation @ p_child + translation."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_quat_wxyz(cls, quat_wxyz, translation_m) -> "Pose":
        return cls(
            quat_wxyz_to_matrix(tuple(quat_wxyz)),
            np.asarray(translation_m, dtype=np.float64).reshape(3),
        )

    @classmethod
    def from_matrix_translation(cls, rotation, translation) -> "Pose":
        return cls(
            np.asarray(rotation, dtype=np.float64).reshape(3, 3),
            np.asarray(translation, dtype=np.float64).reshape(3),
        )

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        return matrix_to_quat_wxyz(self.rotation)

    @property
    def quat_xyzw(self) -> tuple[float, float, float, float]:
        return quat_wxyz_to_xyzw(self.quat_wxyz)

    @property
    def matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def inverse(self) -> "Pose":
        r_inv = self.rotation.T
        return Pose(r_inv, -r_inv @ self.translation)

    def compose(self, other: "Pose") -> "Pose":
        """self ∘ other: maps other's child frame into self's parent frame."""
        return Pose(self.rotation @ other.rotation,
                    self.rotation @ other.translation + self.translation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform (3,) or (N, 3) points from the child frame to the parent."""
        p = np.asarray(points, dtype=np.float64)
        if p.ndim == 1:
            return self.rotation @ p + self.translation
        return p @ self.rotation.T + self.translation

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate (3,) or (N, 3) vectors — no translation (velocities, twists)."""
        v = np.asarray(vectors, dtype=np.float64)
        return self.rotation @ v if v.ndim == 1 else v @ self.rotation.T

    def apply_twist(self, twist: np.ndarray) -> np.ndarray:
        """Rotate a [linear(3), angular(3)] twist into the parent frame."""
        t = np.asarray(twist, dtype=np.float64).reshape(6)
        return np.concatenate([self.rotation @ t[:3], self.rotation @ t[3:]])
