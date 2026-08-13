#!/usr/bin/env python3
"""Identify the uncompensated end-effector payload from libfranka's own residual.

    python sysid/identify_payload.py --selftest          # no hardware
    python sysid/identify_payload.py --yes

You do not have to weigh the tool. libfranka publishes `tau_ext_hat_filtered`, its
estimate of the joint torque it cannot account for, and an under-declared payload
lands there as a gravity signature. Holding still at several poses and fitting

    tau_ext(q) = [ Jv^T g | -Jw^T skew(g) R_F ] . [ m ; m*c ]  +  b

recovers the mass m and its COM offset c in the flange frame, plus a per-joint
constant b that absorbs friction and cable torque.

What separates the two is POSE DEPENDENCE, not magnitude: the payload term moves
with the elbow (q4 changes how the tool hangs), while b does not. A q1 sweep is the
control -- rotating about the gravity vector leaves every joint's gravity torque
invariant, so anything that moves with q1 is cable, not payload.

The fit is only as good as the pose spread. --selftest recovers a synthetic payload
through the same code path and is the check that the frame and sign conventions are
right; run it after touching the regressor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import franka_config as fc
from lerobot_robot_bimanual_franka.franka_fk import franka_fk
from lerobot_robot_bimanual_franka.franka_jacobian import zero_jacobian

NUM_JOINTS = 7
G = np.array([0.0, 0.0, -9.81])       # base frame; the FR3 bases are level, yaw only
OUT_ROOT = Path("~/sysid/outputs").expanduser()


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def pose_of(q):
    p, quat = franka_fk(np.asarray(q))
    return np.asarray(p).reshape(-1), Rotation.from_quat(np.asarray(quat).reshape(-1)).as_matrix()


def regressor(q) -> np.ndarray:
    """(7, 4) mapping [m, m*cx, m*cy, m*cz] to joint torque under gravity."""
    p, R = pose_of(q)
    J = np.asarray(zero_jacobian(np.asarray(q), ee_pos_base=p))
    Jv, Jw = J[:3], J[3:]
    return np.column_stack([Jv.T @ G, -(Jw.T @ _skew(G) @ R)])


def fit(poses, taus, with_offset=True) -> dict:
    """Least squares for [m, m*c] with a per-joint constant.

    The offset is fitted, not assumed zero: friction and cable torque sit in tau_ext
    too, and leaving them out pushes them into the mass.
    """
    n = len(poses)
    blocks, ys = [], []
    for q, t in zip(poses, taus):
        A = regressor(q)
        if with_offset:
            A = np.hstack([A, np.eye(NUM_JOINTS)])
        blocks.append(A)
        ys.append(np.asarray(t, dtype=np.float64))
    A = np.vstack(blocks)
    y = np.concatenate(ys)
    phi, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ phi
    ss = float(np.sum((y - y.mean()) ** 2))
    m = float(phi[0])
    com = (phi[1:4] / m).tolist() if abs(m) > 1e-6 else [float("nan")] * 3
    return dict(mass_kg=m, com_flange_m=com,
                offset_nm=(phi[4:11].tolist() if with_offset else [0.0] * NUM_JOINTS),
                rms_resid_nm=float(np.sqrt(np.mean(resid ** 2))),
                r2=float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan"),
                cond=float(np.linalg.cond(A)), n_poses=n)


def default_poses(q_home) -> list:
    """Elbow and wrist spread identifies the payload; q1 spread separates cable torque."""
    out = []
    for d4 in (0.0, 0.35, -0.35):
        q = np.array(q_home, dtype=np.float64); q[3] = np.clip(q[3] + d4, -3.0, -0.1)
        out.append(q)
    for d6 in (0.4, -0.4):
        q = np.array(q_home, dtype=np.float64); q[5] = np.clip(q[5] + d6, 0.05, 3.7)
        out.append(q)
    for d1 in (0.5, -0.5):
        q = np.array(q_home, dtype=np.float64); q[0] = q[0] + d1
        out.append(q)
    return out


def selftest():
    q_home = np.asarray(fc.home_q("home_pose", key="r"), dtype=np.float64)
    poses = default_poses(q_home)
    true_m, true_c = 1.05, np.array([0.01, -0.004, 0.06])
    true_b = np.array([0.20, -0.35, 0.15, -0.10, 0.05, -0.08, 0.03])
    rng = np.random.default_rng(0)
    taus = [regressor(q) @ np.concatenate([[true_m], true_m * true_c]) + true_b
            + rng.normal(0.0, 0.02, NUM_JOINTS) for q in poses]
    r = fit(poses, taus)
    print("SELFTEST -- synthetic payload through the real code path\n")
    print(f"  mass   true {true_m:.3f} kg      fitted {r['mass_kg']:.3f} kg")
    print(f"  com    true {np.round(true_c,4)}   fitted {np.round(r['com_flange_m'],4)}")
    print(f"  offset true {np.round(true_b,3)}")
    print(f"         fit  {np.round(r['offset_nm'],3)}")
    print(f"  rms residual {r['rms_resid_nm']:.4f} Nm   r2 {r['r2']:.4f}   cond {r['cond']:.0f}")
    ok = abs(r["mass_kg"] - true_m) < 0.05 and np.max(np.abs(np.asarray(r["com_flange_m"]) - true_c)) < 0.02
    print(f"\n  {'PASS' if ok else 'FAIL'} -- frame and sign conventions {'consistent' if ok else 'WRONG'}")
    return ok


# ------------------------------------------------------------------- hardware

def ramp_to(mgr, arm, target, cfg_rate=0.4, dt=0.04):
    q = np.asarray(mgr.current_kinematic_state(arm)[0], dtype=np.float64)
    lead = q.copy()
    step = cfg_rate * dt
    for _ in range(int(2.0 * np.pi / step) + 20):
        if np.max(np.abs(target - lead)) <= 1e-3:
            break
        lead = lead + np.clip(target - lead, -step, step)
        mgr.move_joint_goal_batch({arm: (lead, 1.0, 1.0)})
        time.sleep(dt)
    for _ in range(30):
        mgr.move_joint_goal_batch({arm: (target, 1.0, 1.0)})
        time.sleep(dt)


def _hold_read(mgr, arm, q_target, dwell_s, fps) -> np.ndarray:
    rows = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dwell_s:
        mgr.move_joint_goal_batch({arm: (q_target, 1.0, 1.0)})
        rows.append(np.asarray(mgr.torque_snapshot(arm)[2], dtype=np.float64))
        time.sleep(1.0 / fps)
    # Median, not mean: tau_ext_hat_filtered still carries the occasional spike.
    return np.median(np.asarray(rows), axis=0)


def measure(mgr, arm, q_target, dwell_s=1.5, fps=30.0, approach=0.10) -> tuple:
    """tau_ext at q_target, approached from BOTH directions and averaged.

    A single approach measures gravity plus wherever friction caught the arm, and on
    joints 1-4 breakaway is 0.5-1.0 Nm -- the same size as the payload signature being
    looked for. Approaching from each side flips the hysteresis and leaves gravity, the
    same half-sum/half-difference split measure_joint_friction uses.

    The half-difference is returned too: it is the friction band, and it bounds how much
    of the fit can be trusted.
    """
    reads = []
    for sign in (+1.0, -1.0):
        ramp_to(mgr, arm, np.asarray(q_target) + sign * approach)
        ramp_to(mgr, arm, q_target)
        reads.append(_hold_read(mgr, arm, q_target, dwell_s, fps))
    return 0.5 * (reads[0] + reads[1]), 0.5 * (reads[0] - reads[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pose", default="home_pose")
    ap.add_argument("--profile", default="single_arm_franka")
    ap.add_argument("--arm", default="r")
    ap.add_argument("--dwell-s", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    from lerobot_robot_bimanual_franka.franka_process import MultiRobotWrapper
    q_home = np.asarray(fc.home_q(args.pose, key=args.arm), dtype=np.float64)
    poses = default_poses(q_home)
    if not args.yes:
        resp = input(f"This MOVES the arm through {len(poses)} poses "
                     f"(elbow +/-0.35, wrist +/-0.4, base +/-0.5 rad). Clear? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    spec = fc.profile(args.profile)
    arm_spec = fc.arm(spec.arms[args.arm])
    print(f"profile {args.profile}: key '{args.arm}' -> {arm_spec.name} arm "
          f"({arm_spec.nuc_host})")
    mgr = MultiRobotWrapper()
    mgr.add_robot(args.arm, arm_spec.server_ip, arm_spec.robot_ip, arm_spec.rpyc_port)
    # The assist would ride on top of the residual being measured.
    mgr.set_tuning_all(friction_kc=0.0)

    taus, bands = [], []
    try:
        for i, q in enumerate(poses):
            t, band = measure(mgr, args.arm, q, args.dwell_s)
            taus.append(t)
            bands.append(band)
            print(f"  pose {i+1}/{len(poses)}  q4={q[3]:+.3f} q6={q[5]:+.3f} q1={q[0]:+.3f}"
                  f"   tau_ext {np.round(t, 2)}   +/-{np.round(np.abs(band), 2)}")
        ramp_to(mgr, args.arm, q_home)
    finally:
        mgr.shutdown()

    # q1 rotates about gravity, so a correct gravity model gives identical tau_ext at
    # +q1 and -q1. Whatever differs is not gravity, and it caps what this fit can claim.
    qs = [i for i, q in enumerate(poses) if abs(q[0] - q_home[0]) > 1e-6]
    if len(qs) == 2:
        inv = np.abs(taus[qs[0]] - taus[qs[1]])
        print(f"\n  q1-invariance check (should be ~0): {np.round(inv, 3)} Nm"
              f"   worst {inv.max():.3f}")
    print(f"  friction band (half-difference): worst {np.max(np.abs(bands)):.3f} Nm")

    r = fit(poses, taus)
    print(f"\nidentified payload (flange frame):")
    print(f"  mass          {r['mass_kg']:.3f} kg")
    print(f"  COM offset    {np.round(r['com_flange_m'], 4)} m")
    print(f"  joint offset  {np.round(r['offset_nm'], 3)} Nm   (friction + cable)")
    print(f"  rms residual  {r['rms_resid_nm']:.3f} Nm   r2 {r['r2']:.3f}  cond {r['cond']:.0f}")
    print("\nCompare against what Desk has configured. The difference is the force that")
    print("makes the arm sag under EE_DELTA, which re-anchors its goal and cannot correct it.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out).expanduser() if args.out else OUT_ROOT / f"{stamp}_payload"
    out.mkdir(parents=True, exist_ok=True)
    (out / "payload.json").write_text(json.dumps(
        dict(poses=[q.tolist() for q in poses], tau_ext=[t.tolist() for t in taus],
             **r), indent=2))
    print(f"\nwrote {out}/payload.json")


if __name__ == "__main__":
    main()
