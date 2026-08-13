#!/usr/bin/env python3
"""Command one known OSC axis at a time and measure what the arm actually does.

Ground truth for "are the axes right": bypasses the teleop entirely and pushes
goal poses straight at the controller, then reads the resulting motion back from
the robot's own O_T_EE. The printed matrix is commanded-vs-measured, so
off-diagonal terms are real cross-coupling, not inference.

Each probe is a fixed ABSOLUTE goal (not re-anchored), so the arm steps to
start+delta and stops there, then re-homes through joint impedance before the next
one (--no-home-between to skip).

RESULTS ARE ONLY COMPARABLE AT THE SAME POSE. lambda_pos rotates with the arm, so
the commanded force for a given axis -- and which joints clear breakaway -- both
change with configuration; X read 42/62/65% across three runs at three different
kc values purely because the arm had drifted. Two things keep a run honest, and
both are on by default: re-homing between probes, because the OSC return has to
fight the very friction under test and so lands short and creeps; and the printed
start-pose drift, which is the evidence that it worked. Pass --home-q (or --poses)
to pin the configuration itself.

Nothing here resolves a small imbalance in ONE pass -- the axes worth arguing
about scatter by more than they differ. --repeat pools passes into mean +/- sd and
withholds a DIRECTIONAL verdict until the gap clears 2 sd.

    python scripts/osc_check/check_osc_axes.py                  # wherever it stands
    python scripts/osc_check/check_osc_axes.py --poses sim      # homed, repeatable
    python scripts/osc_check/check_osc_axes.py --poses all      # sweep the workspace
    python scripts/osc_check/check_osc_axes.py --both-signs --repeat 4   # +/- gap

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


def aggregate(all_rows, args) -> dict:
    """Pool repeats into per-(axis, sign) mean, sd and n of the response fraction.

    The sd is the point: a single probe cannot tell a real 20% imbalance from
    drift, and the axes worth arguing about here scatter by more than that
    between passes. Pooling makes the imbalance a measurement with an error bar
    instead of one number.
    """
    per: dict[str, list] = {}
    for rows in all_rows:
        for label, frac, cross, _ in verdicts(rows, args):
            per.setdefault(label, []).append((frac, cross))
    out = {}
    for label, v in per.items():
        f = np.asarray([x for x, _ in v])
        out[label] = (float(np.mean(f)), float(np.std(f)),
                      float(np.mean([c for _, c in v])), len(f))
    return out


def aggregate_to_rows(all_rows, args) -> list[tuple]:
    """Pooled means in the (label, frac, cross, note) shape `verdicts` returns, so the
    by-pose comparison table reads the same whether there was one pass or ten."""
    return [(label, m, cross, f"n={n}")
            for label, (m, _sd, cross, n) in aggregate(all_rows, args).items()]


def report_aggregate(agg, args) -> None:
    """Per-axis +/- response with error bars, and an imbalance in units of sigma.

    A verdict of DIRECTIONAL is only earned when the gap clears twice its own
    combined sd. Below that the honest answer is that this many repeats cannot
    resolve it -- which is a different statement from "the axis is symmetric",
    and is reported as such so it does not get read as evidence either way.
    """
    axes = [a for a in dict.fromkeys(k[:-1] for k in agg) if f"{a}+" in agg and f"{a}-" in agg]
    print(f"\n  {'axis':<12}{'+ %':>12}{'- %':>12}{'imbalance':>14}   verdict")
    for a in axes:
        mp, sp, _, n = agg[f"{a}+"]
        mn, sn, _, _ = agg[f"{a}-"]
        imb = mp - mn
        # Independent passes, so the gap's sd is the quadrature sum.
        sd = float(np.hypot(sp, sn))
        weak = "-" if imb > 0 else "+"
        # No spread to divide by -- one pass, or passes that came back identical --
        # means there is no significance to quote. Quoting it anyway would make
        # sigma infinite and stamp DIRECTIONAL on every axis, which is exactly the
        # false confidence --repeat exists to remove. Report the gap, flag it as
        # unreplicated, and let the caller add passes.
        unreplicated = n < 2 or sd <= 1e-9
        if min(mp, mn) < 0.25:
            v = "one direction stalled - assist too small there"
        elif unreplicated:
            why = "n=1" if n < 2 else f"n={n} but zero spread"
            v = (f"{weak} weaker by {100*abs(imb):.0f}%, UNREPLICATED ({why})"
                 if abs(imb) > 0.15 else f"no large gap, but {why} cannot rule one out")
        elif abs(imb) < 2.0 * sd:
            v = (f"unresolved at n={n} ({abs(imb)/sd:.1f} sigma); "
                 "more repeats or --home-between")
        else:
            v = f"DIRECTIONAL ({abs(imb)/sd:.1f} sigma): {weak} is weaker"
        sd_s = "n/a" if unreplicated else f"{100*sd:.0f}"
        print(f"  {a:<12}{100*mp:6.0f}+-{100*sp:<4.0f}{100*mn:6.0f}+-{100*sn:<4.0f}"
              f"{100*imb:7.0f}+-{sd_s:<5}{v}")
    print("  Imbalance is what a directional kc has to remove, +/- one sd over "
          f"{args.repeat} passes.\n  It is a Cartesian readout of a joint-space "
          "effect: it scores a change, it does not attribute it.")


def run_probe(mgr, kp, kd, args, rehome=None) -> list[tuple]:
    """One full six-axis probe from wherever the arm currently stands.

    With --both-signs each axis is probed in + and -, which is the only way to see
    a DIRECTIONAL friction shortfall: a symmetric assist that is too small shows up
    as both directions sluggish, while a directional one shows up as a gap between
    them. The single-sign probe cannot distinguish those two.
    """
    q0, _, _, _, _, _ = mgr.current_kinematic_state(ARM)
    ns_q = np.asarray(q0, dtype=np.float64)
    home_p, home_r = pose(mgr)
    print(f"  start p = {np.round(home_p, 4)}  reach {np.linalg.norm(home_p):.2f} m"
          f"  rpy = {np.round(np.degrees(home_r.as_euler('xyz')), 1)} deg")

    rows = []
    drift, unconverged = [], []
    signs = (1.0, -1.0) if args.both_signs else (1.0,)
    for i, name in enumerate(AXES):
        for sign in signs:
            # Returning by OSC has to fight the same friction being measured, so it
            # lands short and the start pose creeps probe to probe. `rehome` goes
            # back through joint impedance instead, which does not.
            #
            # A home that did not converge is worse than none: the probe then departs
            # from an unknown configuration while the run looks clean. Count them.
            if rehome is not None and not rehome():
                unconverged.append(f"{name}{'+' if sign > 0 else '-'}")
            drive(mgr, home_p, home_r, kp, kd, ns_q, args.settle)   # return to start
            p0, r0 = pose(mgr)
            drift.append(np.linalg.norm(p0 - home_p))
            dp, dr = np.zeros(3), np.zeros(3)
            if i < 3:
                dp[i] = sign * args.pos_step
            else:
                dr[i - 3] = sign * args.rot_step
            drive(mgr, p0 + dp, Rotation.from_rotvec(dr) * r0, kp, kd, ns_q, args.settle)
            p1, r1 = pose(mgr)
            rows.append((name, sign, i, (p1 - p0) * 1000.0,
                         np.degrees((r1 * r0.inv()).as_rotvec())))
    drive(mgr, home_p, home_r, kp, kd, ns_q, 2.0)
    # Printed every run, not just under --home-between: it is the number that says
    # whether homing was needed. Comparable to the commanded step means the start
    # pose moved as much as the probe did.
    print(f"  start-pose drift over {len(drift)} probes: "
          f"mean {1000*np.mean(drift):.1f} mm, max {1000*np.max(drift):.1f} mm"
          f"  (command {1000*args.pos_step:.0f} mm)")
    if unconverged:
        print(f"  WARNING: homing did not converge before {len(unconverged)} probe(s): "
              f"{', '.join(unconverged)}\n           those departed from an unknown "
              "configuration -- treat their rows as unmeasured, and check "
              "homing.tau_fraction / joint_impedance.kd for the wrist.")
    return rows


def verdicts(rows, args) -> list[tuple[str, float, float, str]]:
    """Direction AND magnitude: an axis pointing the right way at 5% of command is
    the interesting failure, and an argmax-only check scores a 0.0 response a pass."""
    out = []
    for name, sign, i, mp, mr in rows:
        vec = mp if i < 3 else mr
        want = i if i < 3 else i - 3
        cmd = args.pos_step * 1000.0 if i < 3 else np.degrees(args.rot_step)
        # Signed by the command, so a correct response is positive either way and
        # the two directions are directly comparable.
        frac = sign * vec[want] / cmd
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
        label = name if not args.both_signs else f"{name}{'+' if sign > 0 else '-'}"
        out.append((label, frac, cross, v))
    return out


def symmetry(rows, args) -> None:
    """Per-axis + vs - response: the directional assist's actual objective.

    Friction is what breaks this symmetry -- the OSC law is sign-agnostic, so a gap
    between the two directions is the plant, not the controller. Reported as the
    imbalance (plus - minus) so it is signed the same way as
    tuning.friction_kc_joint_{pos,neg}: a positive imbalance means the MINUS
    direction is the one being eaten and wants the larger gain.

    This is a JOINT-SPACE effect read out in Cartesian space, so it does not name
    the joint responsible: J^T mixes all seven into every axis. Use it to decide
    whether the split is worth chasing and to score a kc change, not to attribute.
    """
    per = {}
    for label, frac, _, _ in verdicts(rows, args):
        per.setdefault(label[:-1], {})[label[-1]] = frac
    print(f"\n  {'axis':<12}{'+ %':>7}{'- %':>7}{'imbalance':>11}   verdict")
    for axis, d in per.items():
        if set(d) != {"+", "-"}:
            continue
        p, n = d["+"], d["-"]
        imb = p - n
        worst = min(p, n)
        if worst < 0.25:
            v = "one direction stalled - assist too small there"
        elif abs(imb) > 0.15:
            v = f"DIRECTIONAL: {'-' if imb > 0 else '+'} is the weak direction"
        else:
            v = "symmetric within noise"
        print(f"  {axis:<12}{100*p:6.0f}%{100*n:6.0f}%{100*imb:>10.0f}%   {v}")
    print("  Imbalance is the quantity a directional kc has to remove. It is a "
          "Cartesian\n  readout of a joint-space effect, so it scores a change "
          "but does not attribute it.")


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
    ap.add_argument("--friction-kc-pos", type=float, nargs=7,
                    default=fc.control("tuning.friction_kc_joint_pos"),
                    help="per-joint directional assist gain for POSITIVE commanded "
                         "torque; defaults to config. Pass seven 1.0s to A/B against "
                         "the symmetric assist without editing control.yaml.")
    ap.add_argument("--friction-kc-neg", type=float, nargs=7,
                    default=fc.control("tuning.friction_kc_joint_neg"),
                    help="same, for negative commanded torque")
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes over the full axis set per pose. Results are pooled "
                         "into mean +/- sd, and an imbalance only reads DIRECTIONAL "
                         "once it clears 2 sd. Costs one full pass each; 3-5 is enough "
                         "to separate a 20%% imbalance from drift.")
    ap.add_argument("--home-between", action=argparse.BooleanOptionalAction, default=True,
                    help="re-home through joint impedance before EVERY probe (default). "
                         "The OSC return alone lands short, because it has to fight the "
                         "same friction being measured, so the start pose creeps probe to "
                         "probe and every axis is then measured from a different "
                         "configuration -- which is what made X read 42/62/65%% across "
                         "runs. Costs a homing settle per probe. --no-home-between "
                         "restores the old OSC-only return; the printed drift is what "
                         "says whether that is acceptable.")
    ap.add_argument("--both-signs", action="store_true",
                    help="probe each axis in + and -, and report the imbalance. Doubles "
                         "the probe count. This is the only readout that separates a "
                         "symmetric assist that is too small from a directional one.")
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
    # ALWAYS push: server sessions outlive their clients. Push the full 2x7 -- a
    # bare scalar silently discards the directional vectors, so the run would score
    # a symmetric assist while control.yaml says otherwise.
    kc_pos = np.asarray(args.friction_kc_pos, dtype=np.float64)
    kc_neg = np.asarray(args.friction_kc_neg, dtype=np.float64)
    mgr.set_tuning_all(friction_kc=args.friction_kc * np.stack([kc_pos, kc_neg]))
    directional = not (np.allclose(kc_pos, kc_neg))
    print(f"friction_kc={args.friction_kc}  "
          f"{'DIRECTIONAL' if directional else 'symmetric'} "
          f"pos={list(kc_pos)} neg={list(kc_neg)}")
    print(f"kp_pos_scale={pos_scale}  kp_ori_scale={ori_scale}  kp={kp[0]:.0f}")
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
            # Without an explicit reference, re-home to the configuration the arm was
            # actually in when this pose started. That still removes the creep, which
            # is the point -- every probe departs from one configuration, not from
            # wherever the previous OSC return happened to land.
            q_ref = (q_target if q_target is not None else
                     np.asarray(mgr.current_kinematic_state(ARM)[0], dtype=np.float64))

            def rehome(q_ref=q_ref) -> bool:
                return bool(robot.home(home_q_left=None, home_q_right=q_ref,
                                       max_time_s=args.home_time_s,
                                       tol_rad=args.home_tol_rad))

            all_rows = []
            for r in range(args.repeat):
                if args.repeat > 1:
                    print(f"  --- pass {r + 1}/{args.repeat} ---")
                # A recoverable fault re-arms the loop holding wherever the arm ended
                # up, so the probe that spans it measures reflex recovery, not the
                # control law -- and it reads as a perfectly ordinary percentage. This
                # ran a faulting gain set for three poses and reported it clean.
                f0 = mgr.recovery_counts().get(ARM, 0)
                all_rows.append(run_probe(mgr, kp, kd, args,
                                          rehome=rehome if args.home_between else None))
                faulted = mgr.recovery_counts().get(ARM, 0) - f0
                if faulted:
                    print(f"    WARNING: {faulted} recoverable fault(s) this pass -- "
                          f"these rows are suspect, not a measurement")
            # Single pass keeps the per-probe detail; pooled passes report the mean
            # and its sd instead, since that is the whole point of repeating.
            results[label] = aggregate_to_rows(all_rows, args)
            if args.repeat == 1:
                for axis, frac, cross, v in verdicts(all_rows[0], args):
                    print(f"  {axis:<12}{100*frac:5.0f}% of command, "
                          f"cross-axis {100*cross:4.0f}%   {v}")
            if args.both_signs:
                report_aggregate(aggregate(all_rows, args), args)

        if len(results) > 1:
            print("\n=== % of command, by pose ===")
            print(f"{'axis':<12}" + "".join(f"{k:>12}" for k in results))
            for i, axis in enumerate(next(iter(results.values()))):
                print(f"{axis[0]:<12}"
                      + "".join(f"{100*results[k][i][1]:11.0f}%" for k in results))
            print("\nSpread across poses is the pose-sensitivity of each axis; tune only")
            print("against a fixed pose, and check the others before adopting a value.")
    finally:
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
