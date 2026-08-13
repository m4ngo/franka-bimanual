#!/usr/bin/env python3
"""Measure this arm's per-joint Coulomb friction.

The sim plant has no Coulomb friction; the FR3 has enough to swallow the torques
an OSC command produces in its low-inertia directions. `friction_kc` cancels it
against `pylibfranka_control._FRICTION_COULOMB` -- re-run this after any change
that alters the load.

Two methods, selected by `--method`. Both hold every other joint, both read the
COMMANDED torque (libfranka compensates gravity, so it is friction plus a
pose-dependent model bias), and both cancel that bias by measuring the same
interval in each direction -- see `split_direction`.

`velocity` (default) measures the KINETIC curve: a constant joint-velocity
setpoint, torque averaged over the sweep. Reported as the median half difference
rather than a Coulomb+viscous line fit, because friction still falls with speed
here (Stribeck) and a line extrapolates to a NEGATIVE viscous slope -- which as a
compensator is negative damping. Two constraints, both learned the hard way:

- The setpoint must be a VELOCITY. A position goal restepped each tick is a
  staircase into a kp=600 Nm/rad servo; the resulting dither suppresses stiction
  and reads 40-70% low on joints 1-4.
- It DOES NOT WORK FOR THE WRIST (joints 5-7), structurally. Breaking friction F
  at speed v needs kv > F/(M*v) -- 105 rad/s for joint 5, ~2000 for joint 7 --
  while kv=200 (--kd-scale 8) already chatters into joint_velocity_violation.
  A velocity servo cannot measure a joint whose friction dominates its inertia.

`ramp` measures BREAKAWAY, and does cover the wrist. It ramps the joint-impedance
goal monotonically at a fixed torque rate and reports the commanded torque at the
first sustained motion. This is the quantity the assist must clear -- breakaway is
>= the kinetic curve, and sizing the assist at the kinetic zero-speed intercept
instead under-assists (it cost X 80% -> 39% of command). It is not the dither
failure above: that came from RE-ANCHORING the goal on the measured q each tick,
which sawtooths the error. A monotone absolute ramp adds ~0.02 Nm per tick, which
is quasi-static and never reverses.

Onset is thresholded on VELOCITY, not displacement. The joint deflects elastically
before it slides (presliding), so a displacement trigger fires on gear compliance
and reads breakaway low; a velocity trigger fires just after true breakaway and
reads slightly high, which is the safe direction given friction_kc < 1.

Each joint returns to where it started. Clear the workspace -- this moves the arm.

    python scripts/measure_joint_friction.py --yes
    python scripts/measure_joint_friction.py --joints 3 4 --speeds 0.05 0.1 0.2
    python scripts/measure_joint_friction.py --method ramp --directional \
        --gravity-flip-joint 3 --yes
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

import franka_config as fc
from lerobot_robot_bimanual_franka.franka_process import NUM_JOINTS, MultiRobotWrapper

ARM = "r"

# Averaging window as a fraction of the sweep, centred: the leading edge is the
# servo transient and the trailing edge is the deceleration into the endpoint.
_WINDOW = (0.25, 0.85)

# The ramp knob is a TORQUE rate, so the quasi-static condition is the same on
# every joint; the goal rate it needs is that divided by the joint's stiffness.
_JOINT_KP = np.asarray(fc.control("torque.joint_impedance.kp"), dtype=np.float64)
# Datasheet joint speed limits. MODE_TORQUE's speed bound is a fraction of these, and
# it is the bound that matters on the wrist: exceeding them is what the robot itself
# faults on (joint_velocity_violation), and a travel window cannot catch it in time.
_JOINT_VMAX = np.asarray(fc.control("franka.max_joint_velocity_rad_s"), dtype=np.float64)


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


def torque_step(mgr, q_hold, joint, tau_nm, cfg, window_rad) -> tuple[bool, float, int]:
    """Apply one feedforward torque step; return (moved, mean dq while moving, trips).

    Pushed ONCE, not per tick. Re-pushing re-latches the loop's travel window and
    clears its trip, so a joint that keeps being commanded after a trip ratchets
    outward by up to one window per push -- 1 rad/s at 50 Hz. Pushing once instead
    makes the loop's stale-goal timeout the dead man: torque stops on its own if this
    process dies, which is why `dwell_s` must stay under it.
    """
    trips0 = mgr.torque_trips(ARM)
    tau = np.zeros(NUM_JOINTS)
    tau[joint] = tau_nm
    mgr.move_torque_goal(ARM, tau, joint, q_hold, window_rad,
                         cfg.dq_max_frac * _JOINT_VMAX[joint])
    dq_log, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < cfg.dwell_s:
        _, dq, _, _, _, _ = mgr.current_kinematic_state(ARM)
        dq_log.append(float(np.asarray(dq)[joint]))
        time.sleep(1.0 / cfg.fps)
    trips = mgr.torque_trips(ARM) - trips0
    dq_arr = np.asarray(dq_log) * np.sign(tau_nm)
    # A window trip is a positive detection, not a failure: the joint travelled far
    # enough to cross it. That is a cleaner "it moved" than any velocity threshold,
    # because it does not have to clear the encoder noise floor.
    moved = bool(trips > 0 or (dq_arr.size and np.max(dq_arr) >= cfg.onset_dq_abs))
    return moved, float(np.mean(dq_arr[dq_arr > 0])) if np.any(dq_arr > 0) else 0.0, trips


def torque_breakaway(mgr, q_hold, joint, direction, cfg,
                     start_nm=None) -> tuple[float, float]:
    """Staircase feedforward torque until the joint moves. (breakaway Nm, dq there).

    This is the instrument the servo-based ramp is a workaround for. tau is the
    independent variable here -- it is not kp*e, so it does not fall as the joint
    yields, there is no creep equilibrium, and the result does not depend on
    joint_impedance.kp at all.

    Reads HIGH by however much torque it takes to reach the detector's threshold, so
    it is an upper bound on breakaway; --response extrapolates the kinetic curve back
    to zero velocity, which is the better estimate of the same quantity.

    `start_nm` skips the steps a previous pass already showed do not move it.
    """
    first = 1 if start_nm is None else max(
        1, int(start_nm / cfg.tau_step) - cfg.warm_backoff_steps)
    for i in range(first, int(cfg.tau_max / cfg.tau_step) + 1):
        tau_nm = direction * i * cfg.tau_step
        moved, dq, _ = torque_step(mgr, q_hold, joint, tau_nm, cfg, cfg.window_rad)
        hold(mgr, q_hold, 0.5)          # back inside the window before the next step
        if moved:
            return abs(tau_nm), dq
    return float("nan"), 0.0


def torque_response(mgr, q_hold, joint, breakaway, direction, cfg) -> list[tuple]:
    """(tau, steady dq) above breakaway -- the kinetic friction curve, directly.

    Needs a WIDER window than the breakaway staircase: the joint has to be allowed to
    travel long enough to reach a steady velocity, where tau balances friction plus
    viscous drag. This is the measurement a velocity servo cannot make on the wrist
    (it would need kv of 105-2000 rad/s), and it is what decides whether the viscous
    term is worth modelling at all.
    """
    out = []
    for mult in cfg.response_mults:
        tau_nm = direction * breakaway * mult
        if abs(tau_nm) > cfg.tau_max:
            break
        moved, dq, _ = torque_step(mgr, q_hold, joint, tau_nm, cfg,
                                   cfg.response_window_rad)
        hold(mgr, q_hold, 0.8)
        out.append((abs(tau_nm), dq if moved else 0.0))
    return out


def ramp_rates(joint, cfg) -> tuple[float, float]:
    """(goal rate, onset threshold) in rad/s for one joint. See `ramp_breakaway`."""
    rate = cfg.tau_rate / _JOINT_KP[joint]
    return rate, cfg.onset_frac * rate


def settle(mgr, q_anchor, joint, cfg, timeout_s=3.0) -> bool:
    """Hold until this joint is genuinely still; False on timeout.

    A fixed settle delay is not enough. `ramp_breakaway`'s recovery moves the
    joint OPPOSITE to the ramp it just ran, and the next ramp goes that same way,
    so the signed onset check cannot reject the tail of it -- the transient and
    the ramp share a sign. The result reads as breakaway at near-zero torque.
    Stillness has to be measured, against the same threshold onset uses.
    """
    _, onset_dq = ramp_rates(joint, cfg)
    still = 0
    for _ in range(int(timeout_s * cfg.fps)):
        mgr.move_joint_goal_batch({ARM: (q_anchor, cfg.kp_scale, 1.0)})
        dq = abs(float(np.asarray(mgr.current_kinematic_state(ARM)[1])[joint]))
        still = still + 1 if dq < onset_dq else 0
        if still >= cfg.onset_ticks:
            return True
        time.sleep(1.0 / cfg.fps)
    return False


def ramp_to(mgr, q_anchor, joint, target, cfg, fps=50.0) -> np.ndarray:
    """Walk one joint's impedance goal to `target` at a bounded rate, then settle.

    The only repositioning primitive the ramp method uses: `goto` drives a
    velocity setpoint, which is the thing the wrist cannot do.
    """
    goal = np.asarray(q_anchor, dtype=np.float64).copy()
    q0 = float(goal[joint])
    gap = target - q0
    steps = max(1, int(abs(gap) * fps / cfg.reposition_qdot))
    for i in range(1, steps + 1):
        goal[joint] = q0 + gap * i / steps
        mgr.move_joint_goal_batch({ARM: (goal, cfg.kp_scale, 1.0)})
        time.sleep(1.0 / fps)
    return hold(mgr, goal, 0.4)


def ramp_breakaway(mgr, q_anchor, joint, direction, cfg) -> tuple[float, float, bool]:
    """Ramp `joint`'s impedance goal until it breaks away; return its torque there.

    Returns (tau_at_onset, travel_at_onset, ok). `ok` is False if the ramp hit
    `tau_max` or `max_travel` without a sustained velocity crossing, which is not
    a measurement -- the caller must not average it in.

    The goal only ever moves in `direction`, so the commanded torque is monotone:
    no dither to suppress stiction. `torque_snapshot` reads it exactly, the ramp
    being far inside both limit layers (2 Nm/s against a 1000 Nm/s rate limit,
    ~1 Nm against a >=20 Nm clamp), and needs no -kd*dq correction because it is
    the torque actually applied and at onset inertia takes none of it.

    Three constraints:

    - The threshold is a FRACTION OF THE GOAL RATE, never an absolute speed. The
      goal rate is the fastest the joint can steadily slide -- once sliding it
      creeps at exactly that rate, so the error, and the torque, stop growing. A
      threshold above it is never crossed and the ramp aborts with the joint
      visibly moving. 0.005 rad/s looks conservative and is unreachable on joints
      1-4, whose goal rate at 1 Nm/s is 0.0033.
    - Report the FIRST crossing tick, not the confirming one. `onset_ticks` is
      debounce against encoder noise, not part of breakaway.
    - The reading is high by (threshold x viscous friction) plus at most one ramp
      step -- high, never low. Both terms are symmetric, so they cancel in the half
      difference and leave the asymmetry exact, inflating only the magnitude, the
      way the assist prefers (breakaway >= kinetic). Halve `--onset-frac` to size
      it: whatever the symmetric number moves by is this, not friction.
    """
    goal = np.asarray(q_anchor, dtype=np.float64).copy()
    q0 = float(goal[joint])
    rate, onset_dq = ramp_rates(joint, cfg)
    period = 1.0 / cfg.fps
    if not settle(mgr, goal, joint, cfg):
        return float("nan"), float("nan"), False
    # One entry per debounce tick, so the tail is the tick dq first crossed.
    recent = deque(maxlen=cfg.onset_ticks)
    crossings = 0
    t0 = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            goal[joint] = q0 + direction * rate * t
            mgr.move_joint_goal_batch({ARM: (goal, cfg.kp_scale, 1.0)})
            q, dq, _, _, _, _ = mgr.current_kinematic_state(ARM)
            tau_cmd, _, _ = mgr.torque_snapshot(ARM)
            tau = float(np.asarray(tau_cmd)[joint])
            travel = float(np.asarray(q)[joint]) - q0
            recent.append((tau, travel))
            # Signed: the joint must move the way we are pushing it. An unsigned
            # threshold triggers on a settling transient from the previous ramp.
            if direction * float(np.asarray(dq)[joint]) >= onset_dq:
                crossings += 1
                if crossings >= cfg.onset_ticks:
                    return (*recent[0], True)
            else:
                crossings = 0
            if abs(tau) >= cfg.tau_max or abs(travel) >= cfg.max_travel:
                return tau, travel, False
            rest = period - (time.perf_counter() - t0 - t)
            if rest > 0:
                time.sleep(rest)
    finally:
        # Drop the stored kp*error before moving anything, on every exit including
        # the aborts: a joint that releases with the goal still ahead of it gets
        # snapped there. Then walk back rather than stepping.
        resume = np.asarray(q_anchor, dtype=np.float64).copy()
        resume[joint] = float(np.asarray(mgr.current_kinematic_state(ARM)[0])[joint])
        hold(mgr, resume, 0.2)
        ramp_to(mgr, resume, joint, q0, cfg)


def measure_joint_torque(mgr, q_anchor, joint, cfg) -> dict:
    """Breakaway per direction from open-loop torque, plus the kinetic curve.

    Same return shape and the same `split_direction` algebra as the other methods, so
    the confound analysis and the reports are shared: the half difference is still
    bias-free and the half sum still carries the bias. What changes is only that the
    numbers came from a commanded torque rather than a position servo's kp*e.
    """
    hold(mgr, q_anchor, 0.8)
    tau_hold = float(np.asarray(mgr.torque_snapshot(ARM)[0])[joint])
    ups, dns, curves = [], [], []
    by_order = {+1.0: [], -1.0: []}
    # Warm start per direction. The staircase from 0.05 Nm re-walks the same dozens of
    # steps on every repeat, which is the long silence: 28 steps x 0.8 s x 2 directions
    # x 4 repeats. Resuming a few steps below the last answer keeps the staircase
    # monotone (it only skips torques already known not to move) while cutting repeats
    # 2..n to a handful of steps. Backed off, not resumed exactly, so a genuinely
    # lower breakaway on a later repeat is still found.
    warm = {+1.0: None, -1.0: None}
    for rep in range(cfg.repeats):
        first = +1.0 if rep % 2 == 0 else -1.0
        got = {}
        for direction in (first, -first):
            b, _dq = torque_breakaway(mgr, q_anchor, joint, direction, cfg,
                                      start_nm=warm[direction])
            if not np.isfinite(b):
                break
            got[direction] = b
            warm[direction] = b
            if cfg.response and rep == 0:
                curves.append((direction, torque_response(mgr, q_anchor, joint, b,
                                                          direction, cfg)))
        if len(got) < 2:
            continue
        ups.append(got[+1.0])
        dns.append(-got[-1.0])     # signed, so split_direction's algebra is unchanged
        by_order[first].append(split_direction(got[+1.0], -got[-1.0])[1])

    if not ups:
        return dict(coulomb=float("nan"), spread=float("nan"), bias=float("nan"),
                    track=0.0, half_sums=[], tau_pos=float("nan"), tau_neg=float("nan"),
                    f_pos=float("nan"), f_neg=float("nan"), asym_ratio=float("nan"),
                    tau_hold=tau_hold, asym_by_order={}, curves=[],
                    detail=f"never broke away below {cfg.tau_max:.1f} Nm")

    halves, sums = zip(*(split_direction(u, d) for u, d in zip(ups, dns)))
    sym, asym = float(np.median(halves)), float(np.median(sums))
    return dict(coulomb=sym, spread=float(np.ptp(halves)), bias=asym,
                track=len(ups) / cfg.repeats, half_sums=list(sums),
                tau_pos=float(np.median(ups)), tau_neg=float(np.median(dns)),
                f_pos=sym + asym, f_neg=sym - asym, tau_hold=tau_hold,
                asym_ratio=(abs(asym) / sym if sym > 1e-6 else float("nan")),
                asym_by_order={k: float(np.median(v)) for k, v in by_order.items() if v},
                curves=curves,
                detail=f"hold {tau_hold:+.2f} Nm, {cfg.tau_step:.2f} Nm steps")


def measure_joint_ramp(mgr, q_anchor, joint, cfg) -> dict:
    """Breakaway torque for one joint from matched +/- goal ramps.

    Also reports the HOLDING torque at the anchor. Read it as a LOWER BOUND on the
    pose's torque bias and nothing more: friction absorbs any offset inside its own
    band with zero commanded torque, so a joint reading ~0 Nm of hold can still
    carry a bias of up to F -- and a vertical axis, which gravity cannot torque at
    all, can still carry a cable-bundle torque of that size. Only a hold that is
    already large is informative, and what it says is that this pose cannot measure
    this joint: past breakaway the joint sags to a kp*e balance instead of being
    held by friction, the ramp going with the bias trips on the sag, and the
    arithmetic hands back a negative F+ or F-. `split_direction` still cancels the
    bias in the half difference, so F sym degrades rather than dying, but the
    asymmetry is gone.
    """
    hold(mgr, q_anchor, 0.6)
    tau_hold = float(np.asarray(mgr.torque_snapshot(ARM)[0])[joint])
    tau_pos_log, tau_neg_log, travel_log, ok = [], [], [], 0
    # Half-sums split by which direction was ramped FIRST. Any residue of the
    # order -- a settle that did not fully take, hysteresis in the gear -- lands
    # in these two with opposite sign, while real directional friction lands in
    # both the same way. Comparing them is the only way to tell the two apart.
    by_order = {+1.0: [], -1.0: []}
    for rep in range(cfg.repeats):
        first = +1.0 if rep % 2 == 0 else -1.0
        got = {}
        for direction in (first, -first):
            tau, travel, ok_d = ramp_breakaway(mgr, q_anchor, joint, direction, cfg)
            if not ok_d:
                break
            got[direction] = (tau, travel)
        # A pair is only usable whole: the bias cancels in the half difference of
        # matched directions, so half a pair is not half a measurement.
        if len(got) < 2:
            continue
        ok += 1
        tau_up, tr_up = got[+1.0]
        tau_dn, tr_dn = got[-1.0]
        tau_pos_log.append(tau_up)
        tau_neg_log.append(tau_dn)
        travel_log.extend((abs(tr_up), abs(tr_dn)))
        by_order[first].append(split_direction(tau_up, tau_dn)[1])

    if not ok:
        return dict(coulomb=float("nan"), spread=float("nan"), bias=float("nan"),
                    track=0.0, half_sums=[], tau_pos=float("nan"),
                    tau_neg=float("nan"), f_pos=float("nan"), f_neg=float("nan"),
                    asym_ratio=float("nan"), tau_hold=tau_hold,
                    detail=f"no onset detected (holding {tau_hold:+.2f} Nm)")

    halves, sums = zip(*(split_direction(u, d) for u, d in zip(tau_pos_log, tau_neg_log)))
    sym, asym = float(np.median(halves)), float(np.median(sums))
    travel = float(np.median(travel_log))
    return dict(coulomb=sym, spread=float(np.ptp(halves)), bias=asym,
                # Reuses the velocity method's "did this joint actually move"
                # gate: fraction of ramp pairs that reached onset at all.
                track=ok / cfg.repeats, half_sums=list(sums),
                tau_pos=float(np.median(tau_pos_log)),
                tau_neg=float(np.median(tau_neg_log)),
                f_pos=sym + asym, f_neg=sym - asym,
                asym_ratio=(abs(asym) / sym if sym > 1e-6 else float("nan")),
                tau_hold=tau_hold,
                asym_by_order={k: float(np.median(v)) for k, v in by_order.items() if v},
                # Travel at onset is the presliding check: much above ~0.01 rad and
                # the trigger fired on sliding, not on breakaway. A travel an order
                # of magnitude above the other joints' is the sag above, and pairs
                # with a large holding torque.
                detail=f"hold {tau_hold:+.2f} Nm, onset after {np.degrees(travel):.3f} deg")


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


def asymmetry_usable(sym: float, asym: float) -> bool:
    """Whether a (symmetric, half-sum) pair can become a directional gain.

    |asym| >= sym puts F+ or F- at or below zero, which no friction can be: the
    pose's gravity residual has swamped this joint. The asymmetry is then not
    recoverable from it, and F sym is degraded too (watch its spread). Refuse
    rather than clamp -- clamping the negative side to 0 and letting the other
    reach 2x+ is how a bad pose turns into a pasted 4x over-assist.
    """
    return sym > 1e-6 and abs(asym) < sym


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
                bias=float(np.mean(bias)), track=track,
                half_sums=list(bias),
                detail="  ".join(f"{v:.2f}:{np.median(per_speed[v]):+.3f}" for v in speeds),
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
    capped = []
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
        usable = j in tracked and asymmetry_usable(sym, a)
        if j not in tracked:
            verdict = "not measured (no onset)"
        elif not usable:
            # The holding torque is a LOWER BOUND on the bias, never a measurement
            # of it: friction absorbs any offset inside its own band with zero
            # commanded torque, so a joint reading ~0 Nm of hold can still carry a
            # bias of up to F. Only a hold that is already large says anything.
            hold_nm = r.get("tau_hold", float("nan"))
            cause = ("bias exceeds friction, try another pose"
                     if abs(hold_nm) > 0.5 * sym else
                     f"only {hold_nm:+.2f} Nm of hold, so the bias is under F and "
                     "invisible here -- vary the pose to find it")
            verdict = f"INVALID: {cause}"
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
        # Unusable joints keep 1.0, the symmetric no-op.
        #
        # CAPPED. The ratio is only as good as the half-sum's repeatability, which
        # is +/-0.2-0.3 Nm session to session -- the SIGN reproduces, the magnitude
        # does not. An uncapped ratio near 1.0 sends one direction's gain to ~0,
        # i.e. no assist at all that way, which measured WORSE than symmetric:
        # commanded roll- fell from 27% of command to 0. Capping keeps the
        # direction the measurement is confident about and discards the precision
        # it does not have.
        if usable:
            ratio_c = float(np.clip(a / sym, -args.max_asym_ratio, args.max_asym_ratio))
            if abs(a / sym) > args.max_asym_ratio:
                capped.append(j + 1)
            kc_pos[j] = 1.0 + ratio_c
            kc_neg[j] = 1.0 - ratio_c

    # Direction-order control. Real directional friction is a property of the
    # joint, so it must not care which direction was ramped first; anything that
    # flips sign with the order is an artifact of the measurement, not friction.
    split = {j: results[j]["asym_by_order"] for j in sorted(results)
             if len(results[j].get("asym_by_order") or ()) == 2}
    if split:
        print("\norder control -- half-sum measured with + ramped first vs - first:")
        deltas = []
        for j, s in split.items():
            p, n = s[+1.0], s[-1.0]
            deltas.append(p - n)
            flag = "  ORDER ARTIFACT: flips sign" if p * n < 0 else ""
            print(f"  joint {j + 1}: +first {p:+.3f}   -first {n:+.3f}"
                  f"   delta {p - n:+.3f}{flag}")
        print("  A joint's friction cannot depend on which way it was measured "
              "first. Anything that\n  flips here is measurement residue; raise "
              "--repeats and re-read before trusting its gain.")
        # A delta of consistent SIGN across independent joints is not noise, it is
        # the order leaking in. Averaging the two orders cancels it, which is why
        # asym is a median over both, but half of it is the honest error bar on
        # every asymmetry -- and any asym smaller than that is not a measurement.
        agree = max(sum(d > 0 for d in deltas), sum(d < 0 for d in deltas))
        if len(deltas) > 2 and agree == len(deltas):
            print(f"  ALL {agree} deltas share a sign (p ~ {2.0 ** -(len(deltas) - 1):.3f} "
                  f"by chance): a systematic order effect of {np.mean(deltas):+.3f} Nm "
                  f"survives\n  the settle. Averaged out of asym, but treat "
                  f"+/-{abs(np.mean(deltas)) / 2:.3f} Nm as its error bar and "
                  "distrust any asym under that.")
    elif results:
        print("\nno order control: --repeats 1 measures + first only, so an order "
              "artifact would be\nindistinguishable from directional friction. "
              "Use --repeats 2 or more.")

    def _fmt(v):
        return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"

    # No redeploy: these ride shm via set_tuning, unlike coulomb_nm, which is in
    # the torque: block the NUC keeps its own resolved copy of.
    print("\npaste into config/control.yaml `tuning:` (takes effect next session, "
          "no NUC redeploy):")
    print(f"  friction_kc_joint_pos: {_fmt(kc_pos)}")
    print(f"  friction_kc_joint_neg: {_fmt(kc_neg)}")
    if capped:
        print(f"joints {capped} hit the +/-{args.max_asym_ratio:.2f} cap. Their DIRECTION "
              f"is what reproduces;\nthe magnitude does not, and an uncapped ratio near 1 "
              "leaves one direction unassisted,\nwhich measures worse than symmetric. "
              "Raise --max-asym-ratio only with repeatability\nevidence: three identical "
              "passes whose half-sums agree to well inside the gap.")
    print("\n'+' is the direction of POSITIVE COMMANDED TORQUE on that joint, which is "
          "what\nthe assist keys off -- not the sign of dq. See _friction_feedforward.")
    if not mirrored:
        print("These gains absorb the model bias as if it were friction. That is a "
              "real\nhardware trim and may well help, but it is not a friction "
              "measurement.")


def _asym_floor(*passes) -> float:
    """Smallest asymmetry worth believing, in Nm, measured rather than assumed.

    Taken from the direction-order control: half the mean |+first - -first| gap over
    every joint and pose measured. That gap is residue the settle did not remove, so
    it bounds what any single asymmetry can be trusted to. 0.0 when there is no
    order data (--repeats 1), which gates nothing -- as it should, since a run with
    no control has no evidence about its own noise.
    """
    deltas = []
    for p in passes:
        for r in (p or {}).values():
            by = r.get("asym_by_order") or {}
            if len(by) == 2:
                deltas.append(abs(by[+1.0] - by[-1.0]))
    return 0.5 * float(np.mean(deltas)) if deltas else 0.0


def _report_pose_sweep(results, swept, tracked, args) -> None:
    """Is each joint's asymmetry a property of the joint, or of the pose?

    Friction asymmetry cannot move when an unrelated joint moves. A torque bias can,
    and one smaller than breakaway leaves the holding torque at ~0, so this is the
    only readout that catches it. The decisive case is joint 1: gravity cannot
    torque a vertical axis at all, yet a cable-bundle torque about it varies with
    q1 -- so a joint 1 asymmetry that moves under --sweep-joint 0 was never friction.
    """
    offs = sorted(swept)
    print("\n" + "=" * 78)
    print(f"POSE SWEEP  (joint {args.sweep_joint + 1} offset by "
          f"{', '.join(f'{o:+.2f}' for o in offs)} rad)")
    print("=" * 78)
    header = f"{'joint':<7}{'base':>9}" + "".join(f"{f'{o:+.2f}':>9}" for o in offs) \
             + f"{'spread':>9}{'|mean|':>9}  verdict"
    print(header)
    print("-" * len(header))
    for j in sorted(results):
        vals = [results[j]["bias"]] + [swept[o][j]["bias"] for o in offs]
        ok = [v for v in vals if np.isfinite(v)]
        if j not in tracked or len(ok) < 2:
            verdict, spread, mean = "not measured at every pose", float("nan"), float("nan")
        else:
            spread, mean = float(np.ptp(ok)), float(abs(np.mean(ok)))
            # Order of tests matters. A joint with no asymmetry to begin with has a
            # mean near zero, so ANY noise makes spread > mean -- calling that
            # "pose-dependent" would dress up a symmetric joint as a bias. Rule it
            # out first, against an absolute floor rather than a relative one.
            floor = _asym_floor(results, *swept.values())
            if mean < floor:
                verdict = f"symmetric (under the {floor:.3f} Nm floor); no gain needed"
            elif spread > mean:
                verdict = "POSE-DEPENDENT: this is a torque bias, not friction"
            elif spread > 0.5 * mean:
                verdict = "partly pose-dependent; do not adopt the gain yet"
            else:
                verdict = "stable -> friction, gain is safe to adopt"
        print(f"{j + 1:<7}" + "".join(f"{v:>9.3f}" for v in vals)
              + f"{spread:>9.3f}{mean:>9.3f}  {verdict}")
    print("\nOnly joints marked stable have an asymmetry that belongs to the joint. For "
          "the rest\nthe half-sum is a per-pose trim: it may still help where it was "
          "measured, but it will\nbe wrong, possibly in sign, elsewhere in the workspace.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", default="192.168.3.11")
    ap.add_argument("--robot-ip", default="192.168.200.2")
    ap.add_argument("--port", type=int, default=18813)
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(NUM_JOINTS)))
    ap.add_argument("--tau-step", type=float, default=0.05,
                    help="torque method: Nm per staircase step. This is the measurement's "
                         "resolution, so it bounds the answer directly.")
    ap.add_argument("--dwell-s", type=float, default=0.30,
                    help="torque method: seconds each step is held. MUST stay under "
                         "torque.loop.stale_goal_timeout_s -- the goal is pushed once per "
                         "step, so that timeout is the dead man that stops the torque if "
                         "this process dies. Checked at startup.")
    ap.add_argument("--window-rad", type=float, default=0.02,
                    help="torque method: travel past the hold pose at which the CONTROL "
                         "LOOP latches back to hold. Doubles as the motion detector.")
    ap.add_argument("--onset-dq-abs", type=float, default=0.01,
                    help="torque method: rad/s counting as moving, if the window has not "
                         "already caught it")
    ap.add_argument("--response", action="store_true",
                    help="torque method: after breakaway, sweep torque above it and record "
                         "the steady velocity -- the kinetic friction curve. This is the "
                         "measurement a velocity servo cannot make on the wrist, and what "
                         "decides whether the viscous term needs modelling.")
    ap.add_argument("--response-mults", type=float, nargs="+",
                    default=[1.2, 1.5, 2.0, 3.0],
                    help="torque method: multiples of breakaway for --response")
    ap.add_argument("--dq-max-frac", type=float, default=0.25,
                    help="torque method: speed bound as a fraction of each joint's "
                         "datasheet limit, enforced by the control loop. This is what "
                         "protects the wrist -- a travel window cannot, because joint 7 "
                         "passes 2.61 rad/s after ~0.03 rad and the ROBOT faults first. "
                         "0.25 leaves ~2x headroom over the fastest velocity the kinetic "
                         "curve has actually needed (0.29 rad/s).")
    ap.add_argument("--warm-backoff-steps", type=int, default=4,
                    help="torque method: steps below the previous breakaway that repeats "
                         "2..n restart from. 0 disables the warm start and re-walks the "
                         "staircase from zero every repeat, which is ~4x slower.")
    ap.add_argument("--response-window-rad", type=float, default=0.25,
                    help="torque method: wider window for --response, so the joint can "
                         "travel far enough to reach a steady velocity")
    ap.add_argument("--method", choices=("velocity", "ramp", "torque"), default="velocity",
                    help="velocity: kinetic friction from a constant-speed sweep; joints "
                         "1-4 only. ramp: breakaway from a monotone impedance-goal ramp, "
                         "which is what the assist must clear and the only one of the two "
                         "that works on the wrist. See the module docstring.")
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                    help="velocity method: rad/s, each swept in both directions")
    ap.add_argument("--span", type=float, default=0.20,
                    help="rad swept per direction, centred on the start pose")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--kd-scale", type=float, default=1.0,
                    help="velocity-loop bandwidth scale (kv = 25 * this). Joints 1-4 measure "
                         "fine at 1. Do not raise it for the wrist: 8 chatters into "
                         "joint_velocity_violation, and anything low enough to be stable "
                         "cannot break those joints' friction. See the module docstring.")
    ap.add_argument("--fps", type=float, default=50.0)
    # --- ramp method only -------------------------------------------------
    ap.add_argument("--tau-rate", type=float, default=2.0,
                    help="Nm/s the commanded torque climbs. Divided by each joint's "
                         "impedance kp to get its goal rate, so every joint is equally "
                         "quasi-static. Raising it does NOT inflate the reading, and it "
                         "raises the onset velocity off the encoder noise floor.")
    ap.add_argument("--tau-max", type=float, default=6.0,
                    help="Nm at which a ramp gives up. Well under every joint's clamp; a "
                         "joint that needs more than this is not breaking away for a "
                         "reason friction explains.")
    ap.add_argument("--onset-frac", type=float, default=0.3,
                    help="onset velocity as a fraction of the joint's goal rate, signed "
                         "with the ramp. Must be < 1: a sliding joint creeps at the goal "
                         "rate and no faster, so a higher threshold is never crossed. "
                         "This is what sets the (symmetric) overshoot -- halve it to size "
                         "it. See ramp_breakaway.")
    ap.add_argument("--onset-ticks", type=int, default=3,
                    help="consecutive samples over the threshold before onset is believed. "
                         "Debounce only; the torque reported is from the FIRST of them.")
    ap.add_argument("--max-travel", type=float, default=0.05,
                    help="rad past the start pose at which a ramp gives up")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="scale on torque.joint_impedance.kp for the ramp servo")
    ap.add_argument("--reposition-qdot", type=float, default=0.3,
                    help="rad/s the goal is walked back at between ramps")
    # No --keep-assist. Measuring "with the assist live" is not possible from this
    # script and never was: pylibfranka_control applies _friction_feedforward ONLY in
    # MODE_OSC, and every mode here drives MODE_JOINT. Pushing gains via set_tuning
    # and re-measuring therefore changes nothing and reads as noise. To score a
    # directional gain, use scripts/osc_check/check_osc_axes.py --both-signs, which
    # goes through OSC.
    ap.add_argument("--directional", action="store_true",
                    help="report per-direction friction (F+ / F-) and the asymmetry, and "
                         "print a ready-to-paste tuning.friction_kc_joint_{pos,neg} pair")
    ap.add_argument("--sweep-joint", type=int, default=None,
                    help="with --directional: repeat the whole measurement at several "
                         "poses, offsetting THIS joint by each of --sweep-offsets. The "
                         "asymmetry of real friction is a property of the joint and must "
                         "not move; a torque bias hiding under breakaway does move, and "
                         "is invisible to the holding torque. Use --sweep-joint 0 to test "
                         "joint 1, whose bias cannot be gravity (vertical axis) but can "
                         "be cable-bundle torque.")
    ap.add_argument("--sweep-offsets", type=float, nargs="+", default=[-0.6, 0.6],
                    help="rad, relative to the start pose")
    ap.add_argument("--max-asym-ratio", type=float, default=0.4,
                    help="cap on |asym|/F sym when it becomes a directional gain pair, so "
                         "the gains stay in [1-cap, 1+cap]. Guards the one failure this "
                         "has actually produced on hardware: a ratio near 1 zeroes one "
                         "direction's assist and measured worse than symmetric.")
    ap.add_argument("--gravity-flip-joint", type=int, default=None,
                    help="with --directional: repeat every measurement at a second pose "
                         "with this joint mirrored about 0, so its gravity torque changes "
                         "sign. This is the ONLY thing that separates a real directional "
                         "friction from a constant model bias -- without it the reported "
                         "asymmetry is an upper bound, not an estimate. Joint 4 (index 3) "
                         "is usually the effective one.")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.method == "torque":
        stale = float(fc.control("torque.loop.stale_goal_timeout_s"))
        if args.dwell_s >= stale:
            ap.error(f"--dwell-s {args.dwell_s} must be under "
                     f"torque.loop.stale_goal_timeout_s ({stale}): the torque goal is "
                     "pushed once per step, and that timeout is the only thing that "
                     "stops the torque if this process dies mid-step")
        if args.tau_max > min(fc.control("torque.limits.joint_torque_nm")):
            ap.error(f"--tau-max {args.tau_max} exceeds the smallest joint torque clamp "
                     f"({min(fc.control('torque.limits.joint_torque_nm'))} Nm); the "
                     "staircase would saturate instead of measuring")

    if args.method == "ramp" and not 0.0 < args.onset_frac < 1.0:
        ap.error("--onset-frac must be in (0, 1); at 1 or above a sliding joint creeps "
                 "at the goal rate and the threshold is never crossed")

    travel = args.max_travel if args.method == "ramp" else args.span
    if not args.yes:
        extra = ""
        if args.sweep_joint is not None:
            extra = (f", AND swings joint {args.sweep_joint + 1} by up to "
                     f"{np.degrees(max(abs(o) for o in args.sweep_offsets)):.0f} deg "
                     "between passes")
        resp = input(f"This MOVES the arm ({np.degrees(travel):.0f} deg per joint, "
                     f"returned each time{extra}). Workspace clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    mgr = MultiRobotWrapper()
    mgr.add_robot(ARM, args.server_ip, args.robot_ip, args.port)
    # Sessions outlive their clients, so zero it explicitly rather than inheriting
    # whatever an earlier script left. Belt and braces: MODE_JOINT ignores the assist
    # anyway (see the --keep-assist note above), but MODE_TORQUE must measure the raw
    # plant and this is the one line that guarantees it whatever the loop does.
    mgr.set_tuning_all(friction_kc=0.0)
    # Preflight before anything moves. MODE_TORQUE needs a server that has it AND a
    # matching shm goal block, and neither is inferable from a successful connect.
    if args.method == "torque" and not mgr.supports_torque_mode(ARM):
        mgr.shutdown()
        raise SystemExit(
            "This NUC's server predates MODE_TORQUE, so --method torque cannot run.\n"
            "  bash scripts/deploy_nuc_server.sh <mario|luigi>\n"
            "The shm goal block grew from 56 to 64 slots for this, so the server and\n"
            "the control child have to be redeployed together -- the deploy script\n"
            "restarts both. Nothing has moved.")
    def measure(anchor, joint):
        if args.method == "torque":
            return measure_joint_torque(mgr, anchor, joint, args)
        if args.method == "ramp":
            return measure_joint_ramp(mgr, anchor, joint, args)
        return measure_joint(mgr, anchor, joint, args.speeds, args.span, args.fps,
                             args.repeats, args.kd_scale)

    def reposition(anchor, joint, target):
        if args.method == "ramp":
            ramp_to(mgr, anchor, joint, target, args)
        else:
            goto(mgr, anchor, joint, target, max(args.speeds), kd_scale=args.kd_scale)

    try:
        q_home = hold(mgr, np.asarray(mgr.current_kinematic_state(ARM)[0],
                                      dtype=np.float64), 0.5)
        kind = {"ramp": "breakaway, via impedance ramp",
                "torque": "breakaway, via open-loop torque",
                "velocity": "kinetic"}[args.method]
        print(f"start q = {np.round(q_home, 4)}   ({kind} friction)\n")
        header = (f"{'joint':<7}{'friction Nm':>13}{'spread':>9}{'bias Nm':>10}"
                  f"{'track':>8}   detail")
        print(header)
        print("-" * len(header))
        results = {}
        for j in args.joints:
            r = measure(q_home, j)
            results[j] = r
            print(f"{j + 1:<7}{r['coulomb']:>13.3f}{r['spread']:>9.3f}{r['bias']:>10.3f}"
                  f"{r['track']:>8.2f}   {r['detail']}")
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
            reposition(q_home, gj, q_mirror[gj])
            q_m = hold(mgr, q_mirror, 0.5)
            for j in args.joints:
                mirrored[j] = measure(q_m, j)
                hold(mgr, q_m, 0.4)
            reposition(q_m, gj, q_home[gj])
            hold(mgr, q_home, 0.5)

        # Pose sweep. Real friction asymmetry is a property of the joint, so it must
        # not move with the pose; a torque bias under breakaway does move, and the
        # holding torque cannot see it. This is what the flip cannot do for joint 1.
        swept = {}
        if args.directional and args.sweep_joint is not None:
            sj = args.sweep_joint
            for off in args.sweep_offsets:
                target = q_home[sj] + off
                print(f"\nsweep pass: joint {sj + 1} at {target:+.3f} rad ({off:+.2f})")
                reposition(q_home, sj, target)
                q_s = q_home.copy()
                q_s[sj] = target
                q_a = hold(mgr, q_s, 0.5)
                swept[off] = {}
                for j in args.joints:
                    swept[off][j] = measure(q_a, j)
                    hold(mgr, q_a, 0.4)
                reposition(q_a, sj, q_home[sj])
                hold(mgr, q_home, 0.5)

        coulomb = np.array([results[j]["coulomb"] if j in results else np.nan
                            for j in range(NUM_JOINTS)])
        # Untracked joints did not move as commanded, so their spread says
        # nothing about the measurement noise on the ones that did.
        tracked = [j for j in results if results[j]["track"] >= 0.5]
        worst = max((results[j]["spread"] / max(results[j]["coulomb"], 1e-6)
                     for j in tracked), default=float("nan"))
        print("\npaste into config/control.yaml `torque.friction:` "
              "(then redeploy the NUC):")
        print("  coulomb_nm: [" + ", ".join(f"{c:.2f}" for c in coulomb) + "]")
        print(f"worst relative spread over tracked joints {worst:.2f} -> keep "
              f"friction_kc near {max(0.0, 1.0 - worst):.1f}")
        if args.method in ("ramp", "torque"):
            print("track is the fraction of measurement pairs that reached onset; 0 means "
                  "the joint never broke away below --tau-max.")
            swamped = [j + 1 for j in tracked
                       if not asymmetry_usable(results[j]["coulomb"], results[j]["bias"])]
            if swamped:
                print(f"joints {swamped} were swamped by this pose's gravity residual "
                      "(see the holding torque\nand the INVALID rows): their F sym is "
                      "degraded and their asymmetry is unrecoverable.")
        else:
            print("track is reached/commanded speed; below ~0.5 the joint barely "
                  "moved and that row is not a measurement.")

        curves = [(j, r["curves"]) for j in sorted(results)
                  for r in [results[j]] if r.get("curves")]
        if curves:
            print("\n" + "=" * 78)
            print("KINETIC CURVE  (tau above breakaway -> steady velocity)")
            print("=" * 78)
            print("A curve that RISES with tau has a viscous term worth modelling; one "
                  "that is flat\nis pure Coulomb, and one that falls is Stribeck -- for "
                  "which a line fit gives a\nNEGATIVE viscous slope, i.e. negative "
                  "damping as a compensator. Do not fit one.")
            for j, cs in curves:
                for direction, pts in cs:
                    body = "  ".join(f"{t:.2f}Nm:{v:+.3f}" for t, v in pts)
                    print(f"  joint {j + 1} {'+' if direction > 0 else '-'}: {body}")

        if args.directional:
            _report_directional(results, mirrored, tracked, args)
            if swept:
                _report_pose_sweep(results, swept, tracked, args)
    finally:
        mgr.stop_all_motion()
        mgr.shutdown()


if __name__ == "__main__":
    main()
