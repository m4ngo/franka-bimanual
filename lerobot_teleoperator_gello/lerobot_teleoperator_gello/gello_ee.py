"""GELLO EE teleoperator: reads Dynamixel joint positions and converts to an
absolute EE pose via Franka FR3 forward kinematics.

Action keys: x, y, z, qx, qy, qz, qw, gripper
(compatible with BimanualFranka use_ee_pos=True mode)
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.errors import DeviceNotConnectedError

from .config_gello_ee import GelloEEConfig
from .franka_fk import franka_fk
from .gello import Gello


class GelloEE(Gello):
    """GELLO leader that outputs an absolute EE pose via FR3 forward kinematics."""

    config_class = GelloEEConfig
    name = "gello_ee"

    AXIS_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")

    @property
    def action_features(self) -> dict[str, type]:
        return self._maybe_prefix({axis: float for axis in self.AXIS_NAMES} | {"gripper": float})

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_action = self._get_raw_action()  # unprefixed; we apply prefix once at the bottom
        q = np.array([joint_action[f"joint_{i}"] for i in range(1, 8)])
        pos, quat_xyzw = franka_fk(q)
        return self._maybe_prefix({
            "x":      float(pos[0]),
            "y":      float(pos[1]),
            "z":      float(pos[2]),
            "qx":     float(quat_xyzw[0]),
            "qy":     float(quat_xyzw[1]),
            "qz":     float(quat_xyzw[2]),
            "qw":     float(quat_xyzw[3]),
            "gripper": joint_action["gripper"],
        })
