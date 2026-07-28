#!/usr/bin/env python3
"""Print what the SpaceMouse actually sends, and the goal delta it turns into.

Robot-free: opens only the HID device, so it isolates teleop-side problems (a
dead axis, a sign, an axis swap) from the controller. Columns are the raw
pyspacemouse channels, then the base-frame delta BimanualFranka._osc_goal_delta
would build from the emitted action dict.

    python scripts/check_spacemouse.py --hidraw /dev/hidraw3

Push each axis in turn and check the highlighted column matches what you moved.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_spacemouse import SpaceMouse, SpaceMouseConfig
from lerobot_robot_bimanual_franka.osc_torque_controller import clip_delta

_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidraw", default="/dev/hidraw3")
    ap.add_argument("--translation-scale", type=float, default=0.05)
    ap.add_argument("--rotation-scale", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--deadzone", type=float, default=0.02, help="hide rows quieter than this")
    args = ap.parse_args()

    cfg = SpaceMouseConfig(
        hidraw_path=args.hidraw,
        prefix="r_",
        use_delta=True,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
    )
    teleop = SpaceMouse(cfg)
    teleop.connect()
    print(f"connected to {args.hidraw}; move one axis at a time, ctrl-c to stop\n")
    print(f"{'raw x':>7}{'y':>7}{'z':>7}{'roll':>7}{'pitch':>7}{'yaw':>7}   |"
          f"{'dpos x':>9}{'y':>9}{'z':>9}   |{'drot x':>9}{'y':>9}{'z':>9}   peak")

    seen = {a: 0.0 for a in _AXES}
    period = 1.0 / args.fps
    try:
        while True:
            t0 = time.perf_counter()
            action = teleop.get_action()
            raw = teleop._device.read()
            raw_v = np.array([raw.x, raw.y, raw.z, raw.roll, raw.pitch, raw.yaw], dtype=np.float64)
            for a, v in zip(_AXES, raw_v):
                seen[a] = max(seen[a], abs(float(v)))

            dpos = np.array([action["r_x"], action["r_y"], action["r_z"]])
            dq = np.array([action["r_qx"], action["r_qy"], action["r_qz"], action["r_qw"]])
            n = float(np.linalg.norm(dq))
            drot = np.zeros(3) if n < 1e-9 else Rotation.from_quat(dq / n).as_rotvec()
            dpos, drot = clip_delta(dpos, drot)

            if max(np.abs(raw_v).max(), np.abs(dpos).max(), np.abs(drot).max()) > args.deadzone:
                dom = _AXES[int(np.argmax(np.abs(raw_v)))]
                print("".join(f"{v:7.2f}" for v in raw_v)
                      + "   |" + "".join(f"{v:9.4f}" for v in dpos)
                      + "   |" + "".join(f"{v:9.4f}" for v in drot)
                      + f"   {dom}")

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        print("\nmax |deflection| seen per raw channel:")
        for a in _AXES:
            flag = "   <- never moved" if seen[a] < args.deadzone else ""
            print(f"  {a:<6} {seen[a]:.3f}{flag}")


if __name__ == "__main__":
    main()
