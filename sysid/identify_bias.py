#!/usr/bin/env python3
"""Identify the standing joint torque the OSC cannot reject, and fit the cheapest
model the noise floor supports.

    python sysid/identify_bias.py --selftest        # no hardware
    python sysid/identify_bias.py --yes
    python sysid/identify_bias.py --yes --grid wide

robosuite's OSC is a pure PD law: F = kp*e + kd*edot, with no integral term anywhere
in osc.py. A PD law has a non-zero steady-state error against a static disturbance by
construction. In sim that costs nothing because there is no disturbance. On this arm
there is one, and because EE_DELTA re-anchors goal_pos on the MEASURED pose every
policy step, the standing error never accumulates into a restoring force -- it just
moves the goal. A bounded offset becomes a constant drift velocity.

The franky velocity stack this replaced did not have that problem, and not because it
was better posed: it commanded a joint VELOCITY and Franka's firmware servo closed an
inner loop with integral action, absorbing gravity error, cable torque and friction
before they ever reached the task-space law. Taking over torque control deleted that
servo. This script measures what it was silently doing, so it can be fed forward.

WHAT IS MEASURED
tau_ext_hat_filtered at rest, approached from BOTH directions and half-summed
(identify_payload.measure). Friction flips sign with approach direction and the bias
does not, so the half-sum is the bias and the half-difference is the friction band.
A single approach cannot be used: breakaway on joints 1-4 is 0.5-1.0 Nm, the same size
as the bias being looked for.

WHY NOT identify_payload
That script fits the same data to a single flange payload. It is the right model when
the error IS a payload and the wrong one otherwise -- with the payload nulled it
returned -0.067 kg and a COM 0.71 m behind the flange, i.e. it was absorbing joint 4's
offset into a nonsense geometry. This fits the offset directly and only escalates to a
richer model when the residual says the data justifies one.

MODEL SELECTION IS THE POINT
Three nested models are fitted and compared against the friction band:

    constant   tau_bias = b                      (7 params)
    payload    tau_bias = Y_payload(q) phi + b   (11)
    gravity    tau_bias = Y_grav(q) phi + b      (7 + 3 per swept joint)

A model is only justified if it reduces the residual BELOW the friction band. Fitting
past that point fits stiction, which does not reproduce and will not generalise --
that is how a single-pose directional-friction fit ends up fighting the arm elsewhere.
The script prints which model it recommends and the vector to paste into
config/control.yaml `torque.bias.joint_nm`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import franka_config as fc
from identify_payload import (NUM_JOINTS, G, measure, pose_of, ramp_to, regressor)

OUT_ROOT = Path("~/sysid/outputs").expanduser()


# --------------------------------------------------------------------- poses

def pose_grid(q_home, kind: str) -> list:
    """Configurations to measure at.

    `star` is identify_payload's 7-point spread -- enough to separate a payload, and
    enough to fit a constant. `wide` adds shoulder (q2) and a second elbow ring, which
    is what a gravity-model error needs to be distinguishable from a constant: q2 and
    q4 are the joints whose gravity torque actually changes.
    """
    q_home = np.asarray(q_home, dtype=np.float64)
    out = []

    def add(idx, delta, lo, hi):
        q = q_home.copy()
        q[idx] = float(np.clip(q[idx] + delta, lo, hi))
        out.append(q)

    add(0, 0.0, -2.8, 2.8)                                  # the anchor itself
    for d4 in (0.35, -0.35):
        add(3, d4, -3.0, -0.1)
    for d6 in (0.4, -0.4):
        add(5, d6, 0.05, 3.7)
    for d1 in (0.5, -0.5):
        add(0, d1, -2.8, 2.8)
    if kind == "wide":
        for d2 in (0.3, -0.3):
            add(1, d2, -1.7, 1.7)
        for d4 in (0.7, -0.7):
            add(3, d4, -3.0, -0.1)
        for d1 in (1.0, -1.0):
            add(0, d1, -2.8, 2.8)
    return out


# --------------------------------------------------------------------- fits

def _lstsq(A, y):
    phi, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ phi
    return phi, float(np.sqrt(np.mean(resid ** 2)))


def _design(poses, model: str) -> np.ndarray:
    """Stacked design matrix. Only two models, and both are physically meaningful.

    A per-link gravity regressor is the obvious third, and it is deliberately absent:
    it needs per-link FK, franka_fk exposes the EE frame only, and a regressor built
    from the EE Jacobian for every link is simply wrong. A wrong regressor does not
    fail loudly -- it fits, reports a small residual, and hides the error in its
    parameters. If the two below cannot get under the friction band, that is the
    finding, and the answer is a real regressor rather than a plausible-looking one.
    """
    if model == "constant":
        return np.vstack([np.eye(NUM_JOINTS) for _ in poses])
    if model == "payload":
        return np.vstack([np.hstack([regressor(q), np.eye(NUM_JOINTS)]) for q in poses])
    raise ValueError(model)


def _cross_validate(poses, bias, model: str) -> float:
    """Leave-one-pose-out RMS. This, not the in-sample residual, decides.

    In-sample residual always falls as parameters are added -- the selftest picked a
    28-parameter model over a 7-parameter one on data that was constant by
    construction. What matters is whether the model predicts a configuration it was
    not fitted at, because that is exactly how every pose-fitted trim so far has
    failed: right where it was measured, wrong elsewhere.
    """
    if len(poses) < 3:
        return float("nan")
    errs = []
    for i in range(len(poses)):
        tr = [j for j in range(len(poses)) if j != i]
        A = _design([poses[j] for j in tr], model)
        y = np.concatenate([np.asarray(bias[j], dtype=np.float64) for j in tr])
        phi, _ = _lstsq(A, y)
        pred = _design([poses[i]], model) @ phi
        errs.append(np.asarray(bias[i], dtype=np.float64) - pred)
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def fit_models(poses, bias) -> dict:
    """Fit each model on all poses, and score it by leave-one-pose-out."""
    y = np.concatenate([np.asarray(b, dtype=np.float64) for b in bias])
    out = {}
    for model in ("constant", "payload"):
        A = _design(poses, model)
        if A.shape[0] < A.shape[1]:
            out[model] = dict(rms=float("nan"), cv=float("nan"),
                              n_params=int(A.shape[1]),
                              bias_nm=[float("nan")] * NUM_JOINTS,
                              note=f"under-determined ({A.shape[0]}x{A.shape[1]}); "
                                   f"use --grid wide")
            continue
        phi, r = _lstsq(A, y)
        d = dict(rms=r, cv=_cross_validate(poses, bias, model),
                 n_params=int(A.shape[1]), bias_nm=phi[-NUM_JOINTS:].tolist())
        if model == "payload":
            d["mass_kg"] = float(phi[0])
        out[model] = d
    return out


def report(poses, bias, band, models) -> str:
    b = np.asarray(bias)
    floor = float(np.median(np.abs(np.asarray(band))))
    print("\nstanding torque per joint (half-sum over both approaches), Nm\n")
    print(f"{'pose':<6}" + "".join(f"{'j'+str(i+1):>8}" for i in range(NUM_JOINTS)))
    print("-" * 62)
    for i, row in enumerate(b):
        print(f"{i+1:<6}" + "".join(f"{v:>8.2f}" for v in row))
    print("-" * 62)
    print(f"{'mean':<6}" + "".join(f"{v:>8.2f}" for v in b.mean(axis=0)))
    print(f"{'sd':<6}" + "".join(f"{v:>8.2f}" for v in b.std(axis=0)))
    print(f"\nfriction band (half-difference), median |.| = {floor:.3f} Nm")
    print("This is the noise floor. A model that does not get the residual below it is")
    print("fitting stiction, which does not reproduce.\n")

    print(f"{'model':<12}{'params':>8}{'in-sample':>11}{'leave-1-out':>13}   verdict")
    print("-" * 68)
    best = "constant"
    for name in ("constant", "payload"):
        m = models[name]
        if not np.isfinite(m["rms"]):
            print(f"{name:<12}{m['n_params']:>8}{'-':>11}{'-':>13}   {m.get('note','')[:24]}")
            continue
        print(f"{name:<12}{m['n_params']:>8}{m['rms']:>11.3f}{m['cv']:>13.3f}   "
              f"{'generalises below the band' if m['cv'] < floor else 'ABOVE the band'}")
        # Judged on leave-one-out, and it has to be a real margin, not a rounding win.
        if np.isfinite(m["cv"]) and m["cv"] < 0.9 * models[best]["cv"]:
            best = name
    print(f"\nrecommended model: {best}")

    if models[best]["cv"] >= floor:
        print("\n  WARNING: nothing generalises below the friction band. The bias is")
        print("  pose-dependent in a way neither model captures, so a constant fed")
        print("  forward will be right near the anchor and wrong away from it. Collect")
        print("  --grid wide before trusting this, and check whether the residual")
        print("  tracks q1 (cable / base tilt) or q2/q4 (gravity model).")

    # NEGATED. tau_ext_hat_filtered is the torque the environment applies TO the arm,
    # so an unmodelled load pulling joint 4 negative reads negative here and the loop
    # must add positive to stand against it. torque.bias.joint_nm is defined as what
    # the loop ADDS, so it is -tau_ext. Getting this backwards doubles the drift
    # instead of cancelling it, which is the first thing --sag will tell you.
    v = -np.asarray(models[best]["bias_nm"])
    print("\npaste into config/control.yaml `torque:` (then redeploy the NUC).")
    print("Sign is already flipped: this is what the loop ADDS, = -tau_ext.")
    print("  bias:")
    print("    joint_nm: [" + ", ".join(f"{x:.3f}" for x in v) + "]")
    if best != "constant":
        print(f"\n  NOTE: '{best}' is pose-dependent and only its CONSTANT part is")
        print("  shown. If it won, the pose-dependent part is real and belongs in the")
        print("  loop -- a 7-vector does not capture it. Prefer fixing the declared")
        print("  payload so that part goes away.")
    return best


# ---------------------------------------------------------------- selftest

def selftest() -> bool:
    """Recover a known constant through the real fit path, at a realistic noise floor."""
    q_home = np.asarray(fc.home_q("home_pose", key="r"), dtype=np.float64)
    poses = pose_grid(q_home, "star")
    true_b = np.array([0.28, 0.56, 0.40, -0.98, 0.03, -0.20, 0.20])
    rng = np.random.default_rng(0)
    bias = [true_b + rng.normal(0.0, 0.05, NUM_JOINTS) for _ in poses]
    band = [np.full(NUM_JOINTS, 0.35) for _ in poses]
    print("SELFTEST -- known constant bias through the real fit path\n")
    models = fit_models(poses, bias)
    best = report(poses, bias, band, models)
    got = np.asarray(models["constant"]["bias_nm"])
    ok = np.max(np.abs(got - true_b)) < 0.05 and best == "constant"
    print(f"\n  true      {np.round(true_b, 3)}")
    print(f"  recovered {np.round(got, 3)}")
    print(f"\n  {'PASS' if ok else 'FAIL'} -- constant recovered and correctly preferred")
    return ok


# ---------------------------------------------------------------- hardware

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pose", default="home_pose")
    ap.add_argument("--grid", choices=("star", "wide"), default="star")
    ap.add_argument("--profile", default="single_arm_franka")
    ap.add_argument("--arm", default="r", help="key prefix in --profile, not a side")
    ap.add_argument("--dwell-s", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    q_home = np.asarray(fc.home_q(args.pose, key=args.arm), dtype=np.float64)
    poses = pose_grid(q_home, args.grid)
    if not args.yes:
        resp = input(f"This MOVES the arm through {len(poses)} poses, each approached "
                     f"from both sides. Workspace clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    spec = fc.profile(args.profile)
    if args.arm not in spec.arms:
        raise SystemExit(f"profile '{args.profile}' has no key '{args.arm}'")
    arm_spec = fc.arm(spec.arms[args.arm])
    print(f"profile {args.profile}: key '{args.arm}' -> {arm_spec.name} arm "
          f"({arm_spec.nuc_host})")

    kc = float(fc.control("tuning.friction_kc"))
    if kc != 0.0:
        print(f"\n  NOTE: friction_kc is {kc}. The assist is a torque the loop adds on "
              f"top of\n  the control law, and MODE_JOINT does not apply it, so this "
              f"measurement is\n  unaffected -- but the value fed forward must be "
              f"identified independently of\n  whatever the assist is set to.\n")

    from lerobot_robot_bimanual_franka.franka_process import MultiRobotWrapper
    mgr = MultiRobotWrapper()
    mgr.add_robot(args.arm, arm_spec.server_ip, arm_spec.robot_ip, arm_spec.rpyc_port,
                  use_ee_delta=False)
    bias, band = [], []
    try:
        for i, q in enumerate(poses):
            ramp_to(mgr, args.arm, q)
            s, d = measure(mgr, args.arm, q, dwell_s=args.dwell_s)
            bias.append(s)
            band.append(d)
            print(f"  pose {i+1}/{len(poses)}  q=[{' '.join(f'{v:+.2f}' for v in q)}]"
                  f"  bias {np.round(s, 2)}")
        ramp_to(mgr, args.arm, q_home)
    finally:
        mgr.shutdown()

    models = fit_models(poses, bias)
    best = report(poses, bias, band, models)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out).expanduser() if args.out else OUT_ROOT / f"{stamp}_bias"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bias.json").write_text(json.dumps(
        dict(anchor_q=q_home.tolist(), grid=args.grid,
             poses=[p.tolist() for p in poses],
             bias_nm=[np.asarray(b).tolist() for b in bias],
             band_nm=[np.asarray(b).tolist() for b in band],
             models=models, recommended=best), indent=2))
    print(f"\nwrote {out}/bias.json")


if __name__ == "__main__":
    main()
