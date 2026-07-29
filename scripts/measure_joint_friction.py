#!/usr/bin/env python3
"""Measure this arm's per-joint Coulomb friction.

The sim plant has no Coulomb friction; the FR3 has enough to swallow the torques
an OSC command produces in its low-inertia directions. `friction_kc` cancels it
against `pylibfranka_control._FRICTION_COULOMB` -- re-run this after any change
that alters the load.

Drives one joint at a time at a constant joint-VELOCITY setpoint while the rest
are damped to zero. libfranka compensates gravity, so at constant velocity the
commanded torque IS the friction torque plus a pose-dependent bias; sweeping the
SAME interval in both directions cancels that bias in the half difference.

Two constraints, both learned the hard way:

- The setpoint must be a VELOCITY. A position goal restepped each tick is a
  staircase into a kp=600 Nm/rad servo; the resulting dither suppresses stiction
  and reads 40-70% low on joints 1-4.
- It DOES NOT WORK FOR THE WRIST (joints 5-7), structurally. Breaking friction F
  at speed v needs kv > F/(M*v) -- 105 rad/s for joint 5, ~2000 for joint 7 --
  while kv=200 (--kd-scale 8) already chatters into joint_velocity_violation.
  A velocity servo cannot measure a joint whose friction dominates its inertia;
  that wants an open-loop torque ramp, needing a feedforward field in shm.

Reported as the median half difference over the band, not a Coulomb+viscous line
fit: friction still falls with speed here (Stribeck), so a line extrapolates to a
negative viscous slope. Keep friction_kc below 1 by roughly the printed spread --
over-compensation is negative damping, under-compensation only residual friction.

Each joint returns to where it started. Clear the workspace -- this moves the arm.

    python scripts/measure_joint_friction.py --yes
    python scripts/measure_joint_friction.py --joints 3 4 --speeds 0.05 0.1 0.2
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lerobot_robot_bimanual_franka.franka_process import NUM_JOINTS, MultiRobotWrapper

ARM = "r"

# Averaging window as a fraction of the sweep, centred: the leading edge is the
# servo transient and the trailing edge is the deceleration into the endpoint.
_WINDOW = (0.25, 0.85)


def hold(mgr, q, seconds, fps=30.0) -> np.ndarray:
    """Position-hold at a CONSTANT goal; used only between sweeps to re-anchor."""
    for _ in range(int(seconds * fps)):
        mgr.move_joint_goal_batch({ARM: (q, 1.0, 1.0)})
        time.sleep(1.0 / fps)
    return np.asarray(mgr.current_kinematic_state(ARM)[0], dtype=np.float64)


def drive(mgr, joint, speed, seconds, fps, stop_at=None, kd_scale=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Hold a constant velocity setpoint on `joint`; log its (dq, tau_cmd)."""
    vel = np.zeros(NUM_JOINTS)
    vel[joint] = speed
    period = 1.0 / fps
    t0 = time.perf_counter()
    t_log, dq_log, tau_log = [], [], []
    while True:
        t = time.perf_counter() - t0
        if t >= seconds:
            break
        mgr.move_joint_velocity_batch({ARM: vel}, kd_scale=kd_scale)
        q, dq, _, _, _, _ = mgr.current_kinematic_state(ARM)
        tau_cmd, _, _ = mgr.torque_snapshot(ARM)
        if stop_at is not None and np.sign(speed) * (float(np.asarray(q)[joint]) - stop_at) >= 0.0:
            break
        t_log.append(t)
        dq_log.append(float(np.asarray(dq)[joint]))
        tau_log.append(float(np.asarray(tau_cmd)[joint]))
        rest = period - (time.perf_counter() - t0 - t)
        if rest > 0:
            time.sleep(rest)
    t_log = np.asarray(t_log)
    if t_log.size == 0:
        return np.zeros(0), np.zeros(0)
    span_s = t_log[-1]
    keep = (t_log >= _WINDOW[0] * span_s) & (t_log <= _WINDOW[1] * span_s)
    return np.asarray(dq_log)[keep], np.asarray(tau_log)[keep]


def goto(mgr, q_home, joint, target, speed, fps=50.0, kd_scale=1.0) -> None:
    """Drive one joint to `target`, then re-anchor the whole arm's position."""
    q_now = np.asarray(mgr.current_kinematic_state(ARM)[0], dtype=np.float64)
    gap = target - q_now[joint]
    if abs(gap) > 1e-3:
        drive(mgr, joint, np.sign(gap) * speed, abs(gap / speed) + 1.0, fps,
              stop_at=target, kd_scale=kd_scale)
    anchor = q_home.copy()
    anchor[joint] = float(np.asarray(mgr.current_kinematic_state(ARM)[0])[joint])
    hold(mgr, anchor, 0.4)


def measure_joint(mgr, q_home, joint, speeds, span, fps, repeats, kd_scale) -> dict:
    """Friction torque for one joint from matched up/down sweeps."""
    lo, hi = q_home[joint] - 0.5 * span, q_home[joint] + 0.5 * span
    half, reached, bias = [], [], []
    per_speed = {v: [] for v in speeds}
    for _ in range(repeats):
        for v in speeds:
            # Distance-terminated, not time-terminated: a velocity setpoint is
            # tracked with a friction/kd shortfall, so equal durations would
            # cover unequal intervals and the gravity bias would stop cancelling.
            budget = 3.0 * span / v
            goto(mgr, q_home, joint, lo, max(speeds), kd_scale=kd_scale)
            dq_up, tau_up = drive(mgr, joint, +v, budget, fps, stop_at=hi, kd_scale=kd_scale)
            goto(mgr, q_home, joint, hi, max(speeds), kd_scale=kd_scale)
            dq_dn, tau_dn = drive(mgr, joint, -v, budget, fps, stop_at=lo, kd_scale=kd_scale)
            # tau(+v) - tau(-v) = 2*friction; the pose-dependent bias cancels.
            h = 0.5 * float(np.mean(tau_up) - np.mean(tau_dn))
            half.append(h)
            per_speed[v].append(h)
            reached.append(0.5 * float(np.mean(dq_up) - np.mean(dq_dn)))
            bias.append(0.5 * float(np.mean(tau_up) + np.mean(tau_dn)))
    goto(mgr, q_home, joint, q_home[joint], max(speeds), kd_scale=kd_scale)

    half = np.asarray(half)
    # Fraction of the commanded speed actually reached. A velocity setpoint is
    # always short by friction/kd, so this is not an error -- but a joint that
    # barely moved has not been measured, whatever its torque says.
    cmd = np.asarray([v for _ in range(repeats) for v in speeds])
    track = float(np.min(np.abs(reached) / cmd))
    return dict(coulomb=float(np.median(half)), spread=float(np.ptp(half)),
                bias=float(np.mean(bias)), per_speed=per_speed, track=track)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", default="192.168.3.10")
    ap.add_argument("--robot-ip", default="192.168.201.10")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(NUM_JOINTS)))
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                    help="rad/s; each is swept in both directions")
    ap.add_argument("--span", type=float, default=0.20,
                    help="rad swept per direction, centred on the start pose")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--kd-scale", type=float, default=1.0,
                    help="velocity-loop bandwidth scale (kv = 25 * this). Joints 1-4 measure "
                         "fine at 1. Do not raise it for the wrist: 8 chatters into "
                         "joint_velocity_violation, and anything low enough to be stable "
                         "cannot break those joints' friction. See the module docstring.")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if not args.yes:
        resp = input(f"This MOVES the arm ({np.degrees(args.span):.0f} deg per joint, "
                     f"returned each time). Workspace clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    mgr = MultiRobotWrapper()
    mgr.add_robot(ARM, args.server_ip, args.robot_ip, args.port)
    # Sessions outlive their clients; without this we measure the compensated plant.
    mgr.set_tuning_all(friction_kc=0.0)
    try:
        q_home = hold(mgr, np.asarray(mgr.current_kinematic_state(ARM)[0],
                                      dtype=np.float64), 0.5)
        print(f"start q = {np.round(q_home, 4)}\n")
        header = (f"{'joint':<7}{'friction Nm':>13}{'spread':>9}{'bias Nm':>10}"
                  f"{'track':>8}   median per speed (Nm)")
        print(header)
        print("-" * len(header))
        results = {}
        for j in args.joints:
            r = measure_joint(mgr, q_home, j, args.speeds, args.span, args.fps,
                              args.repeats, args.kd_scale)
            results[j] = r
            detail = "  ".join(f"{v:.2f}:{np.median(r['per_speed'][v]):+.3f}"
                               for v in args.speeds)
            print(f"{j + 1:<7}{r['coulomb']:>13.3f}{r['spread']:>9.3f}{r['bias']:>10.3f}"
                  f"{r['track']:>8.2f}   {detail}")
            hold(mgr, q_home, 0.4)

        coulomb = np.array([results[j]["coulomb"] if j in results else np.nan
                            for j in range(NUM_JOINTS)])
        # Untracked joints did not move as commanded, so their spread says
        # nothing about the measurement noise on the ones that did.
        tracked = [j for j in results if results[j]["track"] >= 0.5]
        worst = max((results[j]["spread"] / max(results[j]["coulomb"], 1e-6)
                     for j in tracked), default=float("nan"))
        print("\n_FRICTION_COULOMB = np.array(["
              + ", ".join(f"{c:.2f}" for c in coulomb) + "])")
        print(f"worst relative spread over tracked joints {worst:.2f} -> keep "
              f"friction_kc near {max(0.0, 1.0 - worst):.1f}")
        print("track is reached/commanded speed; below ~0.5 the joint barely "
              "moved and that row is not a measurement.")
    finally:
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
