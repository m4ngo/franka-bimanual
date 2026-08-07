#!/usr/bin/env python3
"""Command one known OSC axis at a time and measure what the arm actually does.

Ground truth for "are the axes right": bypasses the teleop entirely and pushes
goal poses straight at the controller, then reads the resulting motion back from
the robot's own O_T_EE. The printed matrix is commanded-vs-measured, so
off-diagonal terms are real cross-coupling, not inference.

Each probe is a fixed ABSOLUTE goal (not re-anchored), so the arm steps to
start+delta and stops there; it is returned to the start pose between probes and
at the end.

RESULTS ARE ONLY COMPARABLE AT THE SAME POSE. lambda_pos rotates with the arm, so
the commanded force for a given axis -- and which joints clear breakaway -- both
change with configuration; X read 42/62/65% across three runs at three different
kc values purely because the arm had drifted. Pass --home-q (or --poses) to pin it.

    python scripts/check_osc_axes.py                     # wherever the arm stands
    python scripts/check_osc_axes.py --poses sim         # homed, repeatable
    python scripts/check_osc_axes.py --poses all         # sweep the workspace

Clear the workspace first -- this moves the arm, and --poses/--home-q may move it
a long way to reach the reference configuration.
"""

from __future__ import annotations

import argparse
import time

import franka_config as fc
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka import (
    ControlMode, SingleArmFranka, SingleArmFrankaConfig)
from lerobot_robot_bimanual_franka.osc_torque_controller import resolve_gains

ARM = "r"
AXES = ("+X", "+Y", "+Z", "roll(+X)", "pitch(+Y)", "yaw(+Z)")

# Reference configurations spanning the working reach; "sim" is sysid's init_qpos.
# All three clear the worktable by >0.1 m -- check before adding one.
POSES: dict[str, list[float]] = {
    "folded":   [0.0, -0.15, 0.0, -2.40, 0.0, 1.85, 0.785],   # reach 0.46 m
    "sim":      [0.0, -0.15, 0.0, -1.70, 0.0, 1.55, 0.785],   # reach 0.74 m
    "extended": [0.0, -0.15, 0.0, -1.10, 0.0, 1.25, 0.785],   # reach 0.89 m
}


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


def run_probe(mgr, kp, kd, args) -> list[tuple]:
    """One full six-axis probe from wherever the arm currently stands."""
    q0, _, _, _, _, _ = mgr.current_kinematic_state(ARM)
    ns_q = np.asarray(q0, dtype=np.float64)
    home_p, home_r = pose(mgr)
    print(f"  start p = {np.round(home_p, 4)}  reach {np.linalg.norm(home_p):.2f} m"
          f"  rpy = {np.round(np.degrees(home_r.as_euler('xyz')), 1)} deg")

    rows = []
    for i, name in enumerate(AXES):
        drive(mgr, home_p, home_r, kp, kd, ns_q, args.settle)   # return to start
        p0, r0 = pose(mgr)
        dp, dr = np.zeros(3), np.zeros(3)
        if i < 3:
            dp[i] = args.pos_step
        else:
            dr[i - 3] = args.rot_step
        drive(mgr, p0 + dp, Rotation.from_rotvec(dr) * r0, kp, kd, ns_q, args.settle)
        p1, r1 = pose(mgr)
        rows.append((name, (p1 - p0) * 1000.0,
                     np.degrees((r1 * r0.inv()).as_rotvec())))
    drive(mgr, home_p, home_r, kp, kd, ns_q, 2.0)
    return rows


def verdicts(rows, args) -> list[tuple[str, float, float, str]]:
    """Direction AND magnitude: an axis pointing the right way at 5% of command is
    the interesting failure, and an argmax-only check scores a 0.0 response a pass."""
    out = []
    for i, (name, mp, mr) in enumerate(rows):
        vec = mp if i < 3 else mr
        want = i if i < 3 else i - 3
        cmd = args.pos_step * 1000.0 if i < 3 else np.degrees(args.rot_step)
        frac = vec[want] / cmd
        k = int(np.argmax(np.abs(vec)))
        cross = np.linalg.norm(np.delete(vec, want)) / max(abs(vec[want]), 1e-9)
        if k != want or frac < 0.0:
            v = f"WRONG AXIS (peaked on {'XYZ'[k]})"
        elif frac < 0.25:
            v = "STALLED - below the joint friction floor"
        elif frac < 0.7:
            v = "sluggish"
        else:
            v = "OK"
        out.append((name, frac, cross, v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", default="192.168.3.11")
    ap.add_argument("--robot-ip", default="192.168.200.2")
    ap.add_argument("--port", type=int, default=18813)
    ap.add_argument("--pos-step", type=float, default=0.03, help="metres")
    ap.add_argument("--rot-step", type=float, default=0.15, help="radians")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds per probe")
    ap.add_argument("--kp", type=float, default=0.0,
                    help="action kp in [-1,1]; 0 = robosuite default 150, +1 = 1500")
    ap.add_argument("--kd", type=float, default=0.0,
                    help="action kd in [-1,1]; damping_ratio = 10**kd")
    ap.add_argument("--kp-ori-scale", type=float, nargs="+",
                    default=list(fc.control("tuning.kp_ori_scale")),
                    help="multiply the orientation gains only (scalar or rx ry rz); leaves "
                         "translation at the sim default so the arm stays gentle. Capped at "
                         "10x by KP_LIMITS -- yaw needs its own value, roll/pitch saturate "
                         "the wrist torque clamp long before it does")
    ap.add_argument("--kp-pos-scale", type=float, nargs="+",
                    default=list(fc.control("tuning.kp_pos_scale")),
                    help="same for the translation gains (scalar or x y z). X is the weak "
                         "one here: lambda_pos is ~0.85 kg along it against 6.3 along Z")
    ap.add_argument("--friction-kc", type=float, default=fc.control("tuning.friction_kc"),
                    help="server-side Coulomb friction feedforward (0..1); defaults to the "
                         "robot config's value so a probe measures what teleop flies. Sweep it "
                         "to find the smallest value that un-stalls the rotation axes.")
    ap.add_argument("--poses", choices=(*POSES, "all"), default=None,
                    help="home to a named reference configuration before probing, so runs "
                         "are comparable; 'all' sweeps them and prints the spread")
    ap.add_argument("--home-q", type=float, nargs=7, default=None,
                    help="explicit 7-joint reference pose instead of a named one")
    ap.add_argument("--home-time-s", type=float, default=20.0)
    ap.add_argument("--home-tol-rad", type=float, default=0.02,
                    help="joint 7 settles at friction/kp = 0.008 rad, so tighter "
                         "than ~0.015 cannot converge")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if not args.yes:
        extra = " AND HOMES IT, possibly a long way," if (args.poses or args.home_q) else ""
        resp = input(f"This MOVES the arm{extra} ({args.pos_step*100:.0f} cm / "
                     f"{np.degrees(args.rot_step):.0f} deg per probe). Workspace clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    def _scale3(v):
        a = np.asarray(v, dtype=np.float64)
        return np.full(3, a[0]) if a.size == 1 else a

    ori_scale, pos_scale = _scale3(args.kp_ori_scale), _scale3(args.kp_pos_scale)
    kp, kd = resolve_gains(args.kp, args.kd, ori_scale, kp_pos_scale=pos_scale)

    if args.poses == "all":
        targets = list(POSES)
    elif args.poses:
        targets = [args.poses]
    elif args.home_q:
        targets = ["--home-q"]
    else:
        targets = [None]                       # wherever the arm stands

    cfg = SingleArmFrankaConfig(
        r_server_ip=args.server_ip, r_robot_ip=args.robot_ip,
        r_gripper_ip=args.robot_ip, r_port=args.port,
        control_mode=ControlMode.EE_DELTA, cameras={}, depth=False, depth_cam={})
    robot = SingleArmFranka(cfg)
    robot.connect()
    mgr = robot.robot_manager
    # ALWAYS push: server sessions outlive their clients.
    mgr.set_tuning_all(friction_kc=args.friction_kc)
    print(f"friction_kc={args.friction_kc}  "
          f"kp_pos_scale={pos_scale}  kp_ori_scale={ori_scale}  kp={kp[0]:.0f}")
    print(f"commanding {args.pos_step} m / {args.rot_step} rad per axis")

    results: dict[str, list] = {}
    try:
        for name in targets:
            q_target = (np.asarray(args.home_q, dtype=np.float64) if name == "--home-q"
                        else np.asarray(POSES[name], dtype=np.float64) if name else None)
            label = name or "as-found"
            print(f"\n=== pose: {label} ===")
            if q_target is not None:
                print(f"  homing to {np.round(q_target, 3)} ...")
                if not robot.home(home_q_left=None, home_q_right=q_target,
                                  max_time_s=args.home_time_s, tol_rad=args.home_tol_rad):
                    print("  HOMING DID NOT CONVERGE - skipping this pose")
                    continue
            results[label] = verdicts(run_probe(mgr, kp, kd, args), args)
            for axis, frac, cross, v in results[label]:
                print(f"  {axis:<12}{100*frac:5.0f}% of command, cross-axis {100*cross:4.0f}%   {v}")

        if len(results) > 1:
            print("\n=== % of command, by pose ===")
            print(f"{'axis':<12}" + "".join(f"{k:>12}" for k in results))
            for i, axis in enumerate(AXES):
                print(f"{axis:<12}" + "".join(f"{100*results[k][i][1]:11.0f}%" for k in results))
            print("\nSpread across poses is the pose-sensitivity of each axis; tune only")
            print("against a fixed pose, and check the others before adopting a value.")
    finally:
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
