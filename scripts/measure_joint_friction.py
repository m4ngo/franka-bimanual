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


def split_direction(tau_up: float, tau_dn: float) -> tuple[float, float]:
    """(symmetric friction, half-sum) from one matched pair of sweeps.

    At constant velocity the commanded torque is friction plus a pose-dependent
    model bias, and friction opposes motion:

        tau_up = +F_pos + b
        tau_dn = -F_neg + b

    so the half DIFFERENCE is the symmetric part with b cancelled,

        (tau_up - tau_dn)/2 = (F_pos + F_neg)/2

    and the half SUM carries the directional asymmetry,

        (tau_up + tau_dn)/2 = (F_pos - F_neg)/2 + b

    but carries `b` with it. Those two are perfectly CONFOUNDED in a single
    matched pair: no amount of averaging, repeating, or sweeping more speeds
    separates a constant friction asymmetry from a constant model bias, because
    both sweeps cover the same q interval and see the same mean b.

    `asymmetry_from_poses` is what breaks the confound.
    """
    return 0.5 * (tau_up - tau_dn), 0.5 * (tau_up + tau_dn)


def asymmetry_from_poses(half_sums, gravity_signs) -> dict:
    """Separate directional friction from model bias using gravity sign flips.

    Each entry is `s_k = A + b_k`, where A is the (pose-independent) friction
    asymmetry (F_pos - F_neg)/2 and b_k is that pose's model bias. With poses
    whose gravity torque on this joint has OPPOSITE sign, b flips while A does
    not, so averaging the two groups isolates A:

        mean(s | g=+1) = A + b       mean(s | g=-1) = A - b
        =>  A = (mean_+ + mean_-)/2      b = (mean_+ - mean_-)/2

    With only one sign present A cannot be separated -- `separable` is False and
    `asymmetry` is returned as the raw half-sum, which is an UPPER BOUND on the
    true asymmetry, not an estimate of it.
    """
    s = np.asarray(half_sums, dtype=np.float64)
    g = np.asarray(gravity_signs, dtype=np.float64)
    pos, neg = s[g > 0], s[g < 0]
    if pos.size == 0 or neg.size == 0:
        return dict(asymmetry=float(np.mean(s)), bias=float("nan"),
                    separable=False, spread=float(np.ptp(s)) if s.size > 1 else 0.0)
    mp, mn = float(np.mean(pos)), float(np.mean(neg))
    return dict(asymmetry=0.5 * (mp + mn), bias=0.5 * (mp - mn),
                separable=True, spread=float(np.ptp(s)))


def measure_joint(mgr, q_home, joint, speeds, span, fps, repeats, kd_scale) -> dict:
    """Friction torque for one joint from matched up/down sweeps."""
    lo, hi = q_home[joint] - 0.5 * span, q_home[joint] + 0.5 * span
    half, reached, bias = [], [], []
    # Per-direction means, kept so the directional hypothesis can be tested at
    # all: measure_joint used to collapse them into `half` and throw the sign
    # information away.
    tau_pos_log, tau_neg_log = [], []
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
            # The half SUM keeps the directional asymmetry, confounded with the
            # bias -- see split_direction.
            mu_up, mu_dn = float(np.mean(tau_up)), float(np.mean(tau_dn))
            h, s = split_direction(mu_up, mu_dn)
            half.append(h)
            per_speed[v].append(h)
            reached.append(0.5 * float(np.mean(dq_up) - np.mean(dq_dn)))
            bias.append(s)
            tau_pos_log.append(mu_up)
            tau_neg_log.append(mu_dn)
    goto(mgr, q_home, joint, q_home[joint], max(speeds), kd_scale=kd_scale)

    half = np.asarray(half)
    # Fraction of the commanded speed actually reached. A velocity setpoint is
    # always short by friction/kd, so this is not an error -- but a joint that
    # barely moved has not been measured, whatever its torque says.
    cmd = np.asarray([v for _ in range(repeats) for v in speeds])
    track = float(np.min(np.abs(reached) / cmd))
    sym = float(np.median(half))
    # F_pos = sym + A and F_neg = sym - A, with A the half-sum. Reported as a
    # RATIO because that is the units friction_kc_joint_{pos,neg} is in.
    asym = float(np.median(bias))
    return dict(coulomb=sym, spread=float(np.ptp(half)),
                bias=float(np.mean(bias)), per_speed=per_speed, track=track,
                half_sums=list(bias),
                tau_pos=float(np.median(tau_pos_log)),
                tau_neg=float(np.median(tau_neg_log)),
                f_pos=sym + asym, f_neg=sym - asym,
                asym_ratio=(abs(asym) / sym if sym > 1e-6 else float("nan")))


def _report_directional(results, mirrored, tracked, args) -> None:
    """Per-direction friction and a ready-to-paste directional gain pair."""
    print("\n" + "=" * 78)
    print("DIRECTIONAL FRICTION  (hypothesis: breakaway differs by rotation direction)")
    print("=" * 78)
    if not mirrored:
        print("NO --gravity-flip-joint given. The half-sum below is "
              "(F+ - F-)/2 PLUS the model bias, and the two are perfectly\n"
              "confounded, so treat it as an UPPER BOUND on the asymmetry. Re-run "
              "with e.g. --gravity-flip-joint 3 to separate them.")
    header = (f"{'joint':<7}{'F sym':>9}{'F+':>9}{'F-':>9}{'asym':>9}"
              f"{'asym/F':>9}{'bias':>9}  verdict")
    print(header)
    print("-" * len(header))
    kc_pos, kc_neg = np.ones(NUM_JOINTS), np.ones(NUM_JOINTS)
    for j in sorted(results):
        r = results[j]
        sym = r["coulomb"]
        if mirrored and j in mirrored:
            # Two poses with opposite gravity sign on the flipped joint.
            est = asymmetry_from_poses([r["bias"], mirrored[j]["bias"]], [+1.0, -1.0])
            a, b, sep = est["asymmetry"], est["bias"], est["separable"]
        else:
            a, b, sep = r["bias"], float("nan"), False
        ratio = abs(a) / sym if sym > 1e-6 else float("nan")
        if j not in tracked:
            verdict = "not measured (track < 0.5)"
        elif not sep:
            verdict = f"<= {ratio:.0%} of F, confounded with bias"
        elif ratio > 0.15:
            verdict = f"DIRECTIONAL: {'+' if a > 0 else '-'} direction is harder"
        else:
            verdict = "symmetric within noise"
        print(f"{j + 1:<7}{sym:>9.3f}{sym + a:>9.3f}{sym - a:>9.3f}{a:>9.3f}"
              f"{ratio:>9.2f}{b:>9.3f}  {verdict}")
        # F_pos = sym*(1 + a/sym): express the split as a gain either side of
        # the single measured coulomb_nm, which is what the assist multiplies.
        if j in tracked and sym > 1e-6:
            kc_pos[j] = max(0.0, 1.0 + a / sym)
            kc_neg[j] = max(0.0, 1.0 - a / sym)

    def _fmt(v):
        return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"

    print("\npaste into config/control.yaml `tuning:` (then redeploy the NUC):")
    print(f"  friction_kc_joint_pos: {_fmt(kc_pos)}")
    print(f"  friction_kc_joint_neg: {_fmt(kc_neg)}")
    print("\n'+' is the direction of POSITIVE COMMANDED TORQUE on that joint, which is "
          "what\nthe assist keys off -- not the sign of dq. See _friction_feedforward.")
    if not mirrored:
        print("These gains absorb the model bias as if it were friction. That is a "
              "real\nhardware trim and may well help, but it is not a friction "
              "measurement.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", default="192.168.3.11")
    ap.add_argument("--robot-ip", default="192.168.200.2")
    ap.add_argument("--port", type=int, default=18813)
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
    ap.add_argument("--directional", action="store_true",
                    help="report per-direction friction (F+ / F-) and the asymmetry, and "
                         "print a ready-to-paste tuning.friction_kc_joint_{pos,neg} pair")
    ap.add_argument("--gravity-flip-joint", type=int, default=None,
                    help="with --directional: repeat every measurement at a second pose "
                         "with this joint mirrored about 0, so its gravity torque changes "
                         "sign. This is the ONLY thing that separates a real directional "
                         "friction from a constant model bias -- without it the reported "
                         "asymmetry is an upper bound, not an estimate. Joint 4 (index 3) "
                         "is usually the effective one.")
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

        # Second pass at a mirrored pose. Gravity torque on the flipped joint
        # changes sign, the friction asymmetry does not, so the two passes
        # separate them (asymmetry_from_poses).
        mirrored = {}
        if args.directional and args.gravity_flip_joint is not None:
            gj = args.gravity_flip_joint
            q_mirror = q_home.copy()
            q_mirror[gj] = -q_home[gj]
            print(f"\nsecond pass with joint {gj + 1} mirrored "
                  f"({q_home[gj]:+.3f} -> {q_mirror[gj]:+.3f} rad) to flip its gravity sign")
            goto(mgr, q_home, gj, q_mirror[gj], max(args.speeds), kd_scale=args.kd_scale)
            q_m = hold(mgr, q_mirror, 0.5)
            for j in args.joints:
                mirrored[j] = measure_joint(mgr, q_m, j, args.speeds, args.span,
                                            args.fps, args.repeats, args.kd_scale)
                hold(mgr, q_m, 0.4)
            goto(mgr, q_m, gj, q_home[gj], max(args.speeds), kd_scale=args.kd_scale)
            hold(mgr, q_home, 0.5)

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

        if args.directional:
            _report_directional(results, mirrored, tracked, args)
    finally:
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
