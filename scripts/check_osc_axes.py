#!/usr/bin/env python3
"""Command one known OSC axis at a time and measure what the arm actually does.

Ground truth for "are the axes right": bypasses the teleop entirely and pushes
goal poses straight at the controller, then reads the resulting motion back from
the robot's own O_T_EE. The printed matrix is commanded-vs-measured, so
off-diagonal terms are real cross-coupling, not inference.

Each probe is a fixed ABSOLUTE goal (not re-anchored), so the arm steps to
start+delta and stops there; it is returned to the start pose between probes and
at the end.

    python scripts/check_osc_axes.py                 # right arm, defaults
    python scripts/check_osc_axes.py --pos-step 0.02 --rot-step 0.1

Clear the workspace first -- this moves the arm.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka.franka_process import MultiRobotWrapper
from lerobot_robot_bimanual_franka.osc_torque_controller import resolve_gains

ARM = "r"
AXES = ("+X", "+Y", "+Z", "roll(+X)", "pitch(+Y)", "yaw(+Z)")


def pose(mgr):
    _, _, _, p, quat, _ = mgr.current_kinematic_state(ARM)
    return np.asarray(p, dtype=np.float64), Rotation.from_quat(np.asarray(quat, dtype=np.float64))


def drive(mgr, goal_p, goal_r, kp, kd, ns_q, secs, fps=30.0):
    period = 1.0 / fps
    for _ in range(int(secs * fps)):
        t0 = time.perf_counter()
        mgr.move_osc_goal_batch({ARM: (goal_p, goal_r.as_quat(), kp, kd, ns_q)})
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", default="192.168.3.10")
    ap.add_argument("--robot-ip", default="192.168.201.10")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--pos-step", type=float, default=0.03, help="metres")
    ap.add_argument("--rot-step", type=float, default=0.15, help="radians")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds per probe")
    ap.add_argument("--kp", type=float, default=0.0,
                    help="action kp in [-1,1]; 0 = robosuite default 150, +1 = 1500")
    ap.add_argument("--kd", type=float, default=0.0,
                    help="action kd in [-1,1]; damping_ratio = 10**kd")
    ap.add_argument("--uncouple", choices=("true", "false"), default=None,
                    help="osc.py's uncouple_pos_ori. false applies Lambda_full to the whole "
                         "wrench: zero cross-coupling and far more torque per axis")
    ap.add_argument("--kp-ori-scale", type=float, default=1.0,
                    help="multiply the orientation gains only; leaves translation at the "
                         "sim default so the arm stays gentle")
    ap.add_argument("--friction-kc", type=float, default=None,
                    help="set the server's Coulomb friction feedforward before probing (0..1); "
                         "omit to leave it as-is. Sweep this to find the smallest value that "
                         "un-stalls the rotation axes.")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if not args.yes:
        resp = input(f"This MOVES the arm ({args.pos_step*100:.0f} cm / "
                     f"{np.degrees(args.rot_step):.0f} deg per probe). Workspace clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    mgr = MultiRobotWrapper()
    mgr.add_robot(ARM, args.server_ip, args.robot_ip, args.port, use_ee_delta=True)
    kp, kd = resolve_gains(args.kp, args.kd, args.kp_ori_scale)
    if args.friction_kc is not None or args.uncouple is not None:
        unc = None if args.uncouple is None else (args.uncouple == "true")
        mgr.set_tuning_all(friction_kc=args.friction_kc, uncouple_pos_ori=unc)
        print(f"friction_kc={args.friction_kc}  uncouple_pos_ori={unc}")
    try:
        q0, _, _, _, _, _ = mgr.current_kinematic_state(ARM)
        ns_q = np.asarray(q0, dtype=np.float64)
        home_p, home_r = pose(mgr)
        print(f"\nstart pose  p = {np.round(home_p, 4)}  rpy = {np.round(np.degrees(home_r.as_euler('xyz')), 1)} deg")
        print(f"commanding {args.pos_step} m / {args.rot_step} rad per axis, kp={kp[0]:.0f}\n")

        header = (f"{'commanded':<12}{'measured dpos (mm)':>26}{'measured drot (deg)':>26}")
        print(header)
        print("-" * len(header))
        rows = []
        for i, name in enumerate(AXES):
            drive(mgr, home_p, home_r, kp, kd, ns_q, args.settle)   # return to start
            p0, r0 = pose(mgr)

            dp = np.zeros(3)
            dr = np.zeros(3)
            if i < 3:
                dp[i] = args.pos_step
            else:
                dr[i - 3] = args.rot_step
            drive(mgr, p0 + dp, Rotation.from_rotvec(dr) * r0, kp, kd, ns_q, args.settle)

            p1, r1 = pose(mgr)
            meas_p = (p1 - p0) * 1000.0
            meas_r = np.degrees((r1 * r0.inv()).as_rotvec())
            rows.append((name, meas_p, meas_r))
            print(f"{name:<12}{np.array2string(np.round(meas_p, 1), precision=1):>26}"
                  f"{np.array2string(np.round(meas_r, 1), precision=1):>26}")

        drive(mgr, home_p, home_r, kp, kd, ns_q, 2.0)
        print("\nreturned to start pose")

        # Direction AND magnitude: an axis that points the right way but only
        # achieves 5% of the command is the interesting failure, and an
        # argmax-only verdict scores a 0.0 response as a pass.
        print("\nverdict per axis:")
        axis_names = "XYZ"
        for i, (name, mp, mr) in enumerate(rows):
            vec = mp if i < 3 else mr
            want = i if i < 3 else i - 3
            cmd = args.pos_step * 1000.0 if i < 3 else np.degrees(args.rot_step)
            frac = vec[want] / cmd
            k = int(np.argmax(np.abs(vec)))
            cross = np.linalg.norm(np.delete(vec, want)) / max(abs(vec[want]), 1e-9)
            if k != want or frac < 0.0:
                verdict = f"WRONG AXIS (peaked on {axis_names[k]})"
            elif frac < 0.25:
                verdict = "STALLED - torque likely below the joint friction floor"
            elif frac < 0.7:
                verdict = "sluggish"
            else:
                verdict = "OK"
            print(f"  {name:<12} {100*frac:5.0f}% of command, cross-axis {100*cross:4.0f}%   {verdict}")
    finally:
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
