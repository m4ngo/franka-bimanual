"""On-robot check that measured_ee_twist_world (J @ dq) is trustworthy.

Phase 1: arm at rest — twist must read ~0.
Phase 2: gentle single-joint velocity — J @ dq must agree with the finite
difference of consecutive measured EE poses.

Attended run only (the arm moves):
    python residual_wrapper/check_ee_vel.py [--joint 5] [--vel 0.15] [--secs 2.0]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_wrapper import DEFAULT_HOME_Q, measured_ee_twist_world, start_controller  # noqa: E402

_FPS = 20.0


def _sample(controller):
    return time.perf_counter(), controller.robot_manager.current_kinematic_state_batch(["r"])["r"]


def _pose_fd(t0, s0, t1, s1):
    """Finite-difference twist between two snapshots: [v(3), w(3)] base frame."""
    dt = t1 - t0
    v = (np.asarray(s1[3], dtype=np.float64) - np.asarray(s0[3], dtype=np.float64)) / dt
    r0 = Rotation.from_quat(np.asarray(s0[4], dtype=np.float64))
    r1 = Rotation.from_quat(np.asarray(s1[4], dtype=np.float64))
    w = (r1 * r0.inv()).as_rotvec() / dt
    return np.concatenate([v, w])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--joint", type=int, default=5, help="joint index 0-6 to move (default 5 = wrist)")
    p.add_argument("--vel", type=float, default=0.15, help="joint velocity rad/s (keep gentle)")
    p.add_argument("--secs", type=float, default=2.0, help="motion duration per direction")
    p.add_argument("--no-home", action="store_true",
                   help="skip homing; move from the current configuration")
    args = p.parse_args()
    assert 0 <= args.joint <= 6 and abs(args.vel) <= 0.3

    print("connecting (no cameras)...")
    controller = start_controller(with_cameras=False)
    dt = 1.0 / _FPS
    try:
        if not args.no_home:
            print("homing to DEFAULT_HOME_Q...")
            if not controller.home(home_q_left=None, home_q_right=np.asarray(DEFAULT_HOME_Q)):
                print("WARNING: homing did not converge; continuing from current pose")

        # Phase 1: rest — brake explicitly and let post-homing motion decay
        # (home() exits on position tolerance, not zero velocity).
        controller.robot_manager.move_joint_velocity_batch({"r": [0.0] * 7})
        time.sleep(2.0)
        rest = [_sample(controller)[1] for _ in range(20)]
        rest_tw = np.array([measured_ee_twist_world(s, np.eye(3)) for s in rest])
        rest_v = float(np.median(np.linalg.norm(rest_tw[:, :3], axis=1)))
        rest_w = float(np.median(np.linalg.norm(rest_tw[:, 3:], axis=1)))
        print(f"rest: median |v| = {rest_v:.4f} m/s, |w| = {rest_w:.4f} rad/s (expect ~0)")

        # Phase 2: gentle motion, out and back
        snaps = []
        for sign in (+1.0, -1.0):
            cmd = [0.0] * 7
            cmd[args.joint] = sign * args.vel
            n = int(args.secs * _FPS)
            for _ in range(n):
                t0 = time.perf_counter()
                snaps.append(_sample(controller))
                controller.robot_manager.move_joint_velocity_batch({"r": cmd})
                sleep = dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        try:
            controller.robot_manager.stop_all_motion()
        except Exception:
            pass
        controller.disconnect()

    # Compare J@dq against pose finite-difference (skip spin-up/reversal edges).
    errs_v, errs_w, mags_v, mags_w = [], [], [], []
    for (t0, s0), (t1, s1) in zip(snaps[:-1], snaps[1:]):
        if t1 - t0 > 2.5 * dt:
            continue
        fd = _pose_fd(t0, s0, t1, s1)
        tw = measured_ee_twist_world(s0, np.eye(3))
        errs_v.append(np.linalg.norm(tw[:3] - fd[:3]))
        errs_w.append(np.linalg.norm(tw[3:] - fd[3:]))
        mags_v.append(np.linalg.norm(tw[:3]))
        mags_w.append(np.linalg.norm(tw[3:]))

    med = lambda a: float(np.median(a))
    print(f"motion: median |v| {med(mags_v):.4f} m/s, |w| {med(mags_w):.4f} rad/s "
          f"(commanded joint vel {args.vel} rad/s on joint {args.joint})")
    print(f"agreement vs pose finite-diff: median err v {med(errs_v):.4f} m/s, w {med(errs_w):.4f} rad/s")

    ok = (
        rest_v < 0.01 and rest_w < 0.02
        and med(mags_w) > 0.3 * abs(args.vel)          # twist actually reflects the motion
        and med(errs_v) < 0.02 and med(errs_w) < 0.06  # FD at 20 Hz is noisy; loose bounds
    )
    print("RESULT:", "PASS — measured dq / J@dq trustworthy" if ok else "FAIL — inspect values above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
