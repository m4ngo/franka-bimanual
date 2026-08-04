#!/usr/bin/env python3
"""Calibrate friction_kc_joint so every joint clears breakaway equally.

What is being equalised is NOT displacement per unit torque -- that is 1069x
apart across these joints by design, since qddot = M^-1 tau and M runs 2.03 down
to 0.0019 kg m^2. robosuite does not assume equal joint inertia; mujoco
integrates the real M. What it assumes is ZERO FRICTION, so the target here is
residual friction -> 0 on every joint, which is what makes the arm behave like
the frictionless plant the policies trained against.

The burden is F/M, the acceleration a joint loses to friction, and it is joint 7
that is crippled: 216 rad/s^2 against 0.5-0.9 on the shoulder, 452x worse,
despite carrying LESS friction in Nm. That is why it needs the largest factor.

Method: hold every joint under joint impedance, then creep ONE joint's goal so
its commanded torque kp*e rises slowly until the joint breaks away. The EXTRA
kp*e over the standing hold torque at rest is the friction the law had to supply
by itself; baselining that way drops gravity before the two directions are even
averaged. `asym` is the leftover +/- difference -- large means a bad measurement.

Unlike measure_joint_friction.py this works on the wrist: there is no velocity
servo, so there is no loop whose pole (kv > F/(M*v), 105 rad/s for joint 5 and
~2000 for joint 7) has to be stable. The joint is held, and the setpoint creeps
slowly enough that the per-tick torque step is under 0.06 Nm -- no dither.
Breakaway is detected by POSITION departure: at this creep rate a joint that lets
go slides at the creep rate, so there is no dq spike to threshold on.

Run with the assist OFF (the default) to measure true breakaway; the printed
factors then make kc*_FRICTION_COULOMB match it per joint. --verify re-measures
with those factors applied, where the residual should collapse toward zero.

Each joint moves a few mrad past breakaway and is returned. Clear the workspace.

    python scripts/calibrate_joint_friction.py --yes
    python scripts/calibrate_joint_friction.py --yes --verify
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lerobot_robot_bimanual_franka.franka_process import NUM_JOINTS, MultiRobotWrapper
from lerobot_robot_bimanual_franka.osc_torque_controller import DEFAULT_JOINT_KP

ARM = "r"

# Literature-free: this is the table friction_kc multiplies, mirrored from the
# control loop so the printed factors are directly pasteable into a config.
_FRICTION_COULOMB = np.array([1.19, 1.20, 0.83, 1.19, 0.26, 0.44, 0.41])
# Measured libfranka mass-matrix diagonal, for the F/M burden column only.
_M_DIAG = np.array([1.4267, 2.0302, 1.4044, 1.1216, 0.0248, 0.0414, 0.0019])


def hold(mgr, goal, seconds, fps=50.0) -> None:
    for _ in range(int(seconds * fps)):
        mgr.move_joint_goal_batch({ARM: (goal, 1.0, 1.0)})
        time.sleep(1.0 / fps)


def settle(mgr, goal, args) -> np.ndarray:
    """Hold `goal` until the arm is genuinely still; return the rest q.

    Re-read every time: each creep leaves the joint displaced, and measuring
    kp*e against a stale reference reports the standing offset, not breakaway.
    """
    hold(mgr, goal, 0.4, args.fps)
    for _ in range(int(args.settle_timeout * args.fps)):
        mgr.move_joint_goal_batch({ARM: (goal, 1.0, 1.0)})
        q, dq, _, _, _, _ = mgr.current_kinematic_state(ARM)
        if np.max(np.abs(np.asarray(dq))) < args.still_dq:
            return np.asarray(q, dtype=np.float64)
        time.sleep(1.0 / args.fps)
    return np.asarray(mgr.current_kinematic_state(ARM)[0], dtype=np.float64)


def creep_to_breakaway(mgr, base_goal, joint, direction, args) -> float:
    """Ramp one joint's goal until it moves; return the EXTRA kp*e that took.

    Baselined against the standing hold torque at rest, so gravity and any
    residual offset drop out before the half difference even sees them.
    """
    kp = float(DEFAULT_JOINT_KP[joint])
    q_rest = settle(mgr, base_goal, args)
    tau0 = kp * float(base_goal[joint] - q_rest[joint])
    q_start = float(q_rest[joint])

    goal = base_goal.copy()
    result, why = float("nan"), "timeout"
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t > args.timeout:
            break
        goal[joint] = base_goal[joint] + direction * args.creep * t
        mgr.move_joint_goal_batch({ARM: (goal, 1.0, 1.0)})
        q, _, _, _, _, _ = mgr.current_kinematic_state(ARM)
        moved = direction * (float(np.asarray(q)[joint]) - q_start)
        if abs(moved) > args.max_travel:
            why = "ran away"
            break
        # Position departure, NOT a dq spike: at this creep rate a joint that
        # breaks away slides at the creep rate itself, so dq never spikes.
        if moved > args.pos_thresh:
            result, why = kp * float(goal[joint] - q[joint]) - tau0, "ok"
            break
        time.sleep(1.0 / args.fps)
    if why != "ok":
        print(f"      joint {joint + 1} dir {direction:+.0f}: no breakaway ({why}, "
              f"goal led {abs(float(goal[joint]) - base_goal[joint]):.4f} rad = "
              f"{kp * abs(float(goal[joint]) - base_goal[joint]):.2f} Nm)")

    # Ramp the goal home rather than stepping it: a 0.05 rad step into kp=600
    # is a 30 Nm yank.
    reached, steps = float(goal[joint]), max(int(0.8 * args.fps), 1)
    for i in range(steps):
        goal[joint] = reached + (base_goal[joint] - reached) * (i + 1) / steps
        mgr.move_joint_goal_batch({ARM: (goal, 1.0, 1.0)})
        time.sleep(1.0 / args.fps)
    return result * direction


def measure(mgr, q_home, joint, args) -> dict:
    """Breakaway torque for one joint, averaged over both directions."""
    plus, minus = [], []
    for _ in range(args.repeats):
        for sign, out in ((+1.0, plus), (-1.0, minus)):
            out.append(creep_to_breakaway(mgr, q_home, joint, sign, args))
    plus, minus = np.asarray(plus), np.asarray(minus)
    if np.all(np.isnan(plus)) or np.all(np.isnan(minus)):
        return dict(friction=float("nan"), spread=float("nan"), asym=float("nan"))
    # Each direction already returns a positive breakaway magnitude; averaging
    # them cancels whatever direction-dependent bias survived the baseline.
    f = 0.5 * (np.nanmedian(plus) + np.nanmedian(minus))
    both = np.concatenate([plus, minus])
    return dict(friction=float(f), spread=float(np.nanmax(both) - np.nanmin(both)),
                asym=float(np.nanmedian(plus) - np.nanmedian(minus)))


def run_pass(mgr, q_home, args, label) -> np.ndarray:
    print(f"\n{label}")
    header = (f"{'joint':<7}{'breakaway Nm':>14}{'spread':>9}{'asym Nm':>10}"
              f"{'F/M rad/s^2':>13}{'factor':>9}")
    print(header)
    print("-" * len(header))
    out = np.full(NUM_JOINTS, np.nan)
    for j in args.joints:
        r = measure(mgr, q_home, j, args)
        out[j] = r["friction"]
        factor = r["friction"] / _FRICTION_COULOMB[j] if np.isfinite(r["friction"]) else np.nan
        print(f"{j + 1:<7}{r['friction']:>14.3f}{r['spread']:>9.3f}{r['asym']:>10.3f}"
              f"{r['friction'] / _M_DIAG[j]:>13.1f}{factor:>9.2f}")
        hold(mgr, q_home, 0.3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", default="192.168.3.10")
    ap.add_argument("--robot-ip", default="192.168.201.10")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(NUM_JOINTS)))
    ap.add_argument("--creep", type=float, default=0.004,
                    help="rad/s the GOAL advances; kp*creep is the torque ramp rate")
    ap.add_argument("--pos-thresh", type=float, default=0.0005,
                    help="rad of departure counted as breakaway; above the elastic "
                         "deflection of the drive (~1e-4) and far above encoder noise")
    ap.add_argument("--still-dq", type=float, default=0.006,
                    help="rad/s the whole arm must be under before a creep starts")
    ap.add_argument("--settle-timeout", type=float, default=3.0)
    ap.add_argument("--max-travel", type=float, default=0.08, help="rad, abort bound")
    ap.add_argument("--timeout", type=float, default=15.0, help="s per direction")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--verify", action="store_true",
                    help="re-measure with the computed factors applied; residual should fall")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if not args.yes:
        resp = input(f"This MOVES the arm (creeps each joint to breakaway, "
                     f"<= {np.degrees(args.max_travel):.0f} deg). Workspace clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    mgr = MultiRobotWrapper()
    mgr.add_robot(ARM, args.server_ip, args.robot_ip, args.port)
    try:
        mgr.set_tuning_all(friction_kc=0.0)          # measure the RAW arm
        q_home = np.asarray(mgr.current_kinematic_state(ARM)[0], dtype=np.float64)
        print(f"start q = {np.round(q_home, 4)}")
        raw = run_pass(mgr, q_home, args, "assist OFF -- true breakaway friction")

        factors = raw / _FRICTION_COULOMB
        good = np.isfinite(factors)
        print("\n_FRICTION_COULOMB = np.array(["
              + ", ".join(f"{v:.2f}" if np.isfinite(v) else "nan" for v in raw) + "])")
        print("friction_kc_joint = ("
              + ", ".join(f"{v:.2f}" if np.isfinite(v) else "1.00" for v in factors) + ")")
        if good.any():
            print(f"\nfriction burden F/M spans {np.nanmin(raw[good] / _M_DIAG[good]):.1f} to "
                  f"{np.nanmax(raw[good] / _M_DIAG[good]):.1f} rad/s^2 -- that ratio, not the "
                  f"raw Nm, is what each factor has to equalise.")

        if args.verify:
            applied = np.where(good, factors, 1.0)
            mgr.set_tuning_all(friction_kc=applied)
            run_pass(mgr, q_home, args, f"assist ON at {np.round(applied, 2)} -- residual")
            print("\nResidual should be near zero on every joint. A joint still high is "
                  "under-assisted; one that trips instantly (or will not hold) is over.")
    finally:
        mgr.set_tuning_all(friction_kc=0.0)
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
