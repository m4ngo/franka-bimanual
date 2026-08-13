#!/usr/bin/env python3
"""Tune the real arm to match a sim reference trajectory, open-loop.

    python sysid/tune.py sysid/data.hdf5 --yes                       # score as-configured
    python sysid/tune.py sysid/data.hdf5 --sweep friction_kc=0.5,1,1.5 --yes
    python sysid/tune.py sysid/data.hdf5 --score-only ~/sysid/outputs/<run>/

Objective is the PER-STEP task response ratio, real/sim. Accumulated endpoint error is
not usable here: once the plants diverge the same deltas resolve to different goals, so
it measures horizon rather than plant, and on a position circle it exceeds the
frozen-arm floor. Get each step right and the trajectory follows.

Task and nullspace are scored separately because the OSC controls 6 DOF at kp and the
7th at nullspace_kp, 15x weaker -- different plants. A nullspace ratio near 1 with a low
task ratio rules out a global torque deficit, and makes per-joint ratios misleading: a
nullspace-heavy joint reads high while every task direction reads low.

Clear the workspace; this moves the arm continuously.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot_robot_bimanual_franka import ControlMode, SingleArmFranka, SingleArmFrankaConfig
from lerobot_robot_bimanual_franka.franka_jacobian import zero_jacobian
from lerobot_robot_bimanual_franka.osc_torque_controller import (
    DELTA_POS_MAX, DELTA_ROT_MAX, JOINT_TORQUE_LIMITS,
)

ARM = "r"
NJ = 7
# The loop's clamp, unscaled: pylibfranka_control applies no safety factor.
_TAU_LIMIT = np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64)
_VEC = ("kp_pos_scale", "kp_ori_scale", "kd_pos_scale", "kd_ori_scale")
_SCALAR = ("friction_kc", "ee_translation_fudge", "ee_rotation_fudge")
_ATTR = {"kp_pos_scale": "_kp_pos_scale", "kp_ori_scale": "_kp_ori_scale",
         "kd_pos_scale": "_kd_pos_scale", "kd_ori_scale": "_kd_ori_scale",
         "ee_translation_fudge": "_trans_fudge", "ee_rotation_fudge": "_rot_fudge"}


def rig_config() -> SingleArmFrankaConfig:
    # Connection fields and every tuning knob come from config/ via default_factory.
    return SingleArmFrankaConfig(control_mode=ControlMode.EE_DELTA, cameras={},
                                 depth=False, depth_cam={})


def load_ref(path: str, index: int) -> dict:
    with h5py.File(Path(path).expanduser(), "r") as f:
        top = f[sorted(f.keys())[0]]
        name = sorted(top.keys())[index]
        g = top[name]
        ref = dict(name=name, action=g["action"][:].astype(np.float64),
                   qpos=g["qpos"][:].astype(np.float64),
                   eef_pos=g["eef_pos"][:].astype(np.float64))
        cfg = {}
        raw = f.attrs.get("controller_cfg")
        if raw is not None:
            try:
                cfg = json.loads(raw)
            except (TypeError, ValueError):
                cfg = {}
    # Invert the exponential remap so real runs the gains sim was generated with.
    ref["kp_action"] = float(np.log10(float(cfg.get("kp", 150.0)) / 150.0))
    ref["kd_action"] = float(np.log10(float(cfg.get("damping_ratio", 1.0))))
    return ref


# ---------------------------------------------------------------------- score

def _decompose(q, p, n):
    """Total per-step joint travel split into task (6 dof) and nullspace (1 dof).

    Takes an absolute step count, not a fraction: derived per-array it silently
    scored the two sides over different horizons whenever the real run was cut
    short. On the one recorded --score-only run that reported TASK 0.15 against a
    true 0.45, because 50 real steps were being compared with 100 sim ones.
    """
    d = np.diff(q[:n], axis=0)
    t = ns = 0.0
    for i in range(len(d)):
        V = np.linalg.svd(zero_jacobian(q[i], ee_pos_base=p[i]))[2]
        t += float(np.linalg.norm(V[:6] @ d[i]))
        ns += float(np.linalg.norm(V[6:] @ d[i]))
    return t, ns


def score(ref, real_q, real_p, frac=0.5, dead=2e-4, min_travel=1e-2) -> dict:
    # ONE window, taken from the shorter run, so a truncated real episode shortens
    # both sides instead of only its own. _per_joint already did this.
    n = max(2, int(min(len(ref["qpos"]), len(real_q)) * frac))
    ts, ns_s = _decompose(ref["qpos"], ref["eef_pos"], n)
    tr, ns_r = _decompose(real_q, real_p, n)
    # A ratio needs sim to have moved. A single-axis sine leaves the nullspace almost
    # untouched, and dividing by that gave 23605.
    ratio = lambda r, s: r / s if s > min_travel else float("nan")
    return dict(task=ratio(tr, ts), nullspace=ratio(ns_r, ns_s),
                per_joint=_per_joint(ref["qpos"], real_q, frac, dead))


def _per_joint(sim_q, real_q, frac, dead) -> dict:
    n = max(2, int(min(len(sim_q), len(real_q)) * frac))
    ds, dr = np.diff(sim_q[:n], axis=0), np.diff(real_q[:n], axis=0)
    per = {}
    for j in range(NJ):
        for key, m in (("pos", ds[:, j] > dead), ("neg", ds[:, j] < -dead)):
            # Ratio of total travel, not mean of per-step ratios: the latter is
            # dominated by the smallest sim steps, which are the noisiest.
            s = float(np.abs(ds[m, j]).sum()) if m.sum() >= 3 else 0.0
            per[(j, key)] = float(np.abs(dr[m, j]).sum() / s) if s > 1e-3 else float("nan")
    return per


def report(s: dict, verbose=False) -> None:
    ns = "  n/a" if np.isnan(s["nullspace"]) else f"{s['nullspace']:.2f}"
    print(f"  TASK {s['task']:.2f}   NULLSPACE {ns}"
          + ("   (n/a = sim barely used the nullspace here)" if ns == "  n/a" else ""))
    if not verbose:
        return
    for j in range(NJ):
        p, n = s["per_joint"][(j, "pos")], s["per_joint"][(j, "neg")]
        if np.isfinite(p) or np.isfinite(n):
            print(f"    j{j+1}  +{p:6.2f}  -{n:6.2f}")


# ------------------------------------------------------------------- hardware

def home_verified(robot, q_target, tol=0.005, attempts=3) -> bool:
    # home() returning True is not enough: a wrong start pose measures a different
    # inertia, and that spread has previously swamped the signal being measured.
    for k in range(attempts):
        robot.home(home_q_left=None, home_q_right=q_target, max_time_s=20.0,
                   tol_rad=tol, fps=30)
        q = np.asarray(robot.robot_manager.current_kinematic_state(ARM)[0], dtype=np.float64)
        if float(np.max(np.abs(q - q_target))) < tol:
            return True
    return False


def apply_knobs(robot, vals: dict) -> None:
    # send_action reads these per step, so no reconnect between trials.
    for k, v in vals.items():
        if k == "friction_kc":
            robot.robot_manager.set_tuning_all(friction_kc=float(v))
        elif k in _VEC:
            setattr(robot, _ATTR[k], np.asarray(v, dtype=np.float64))
        else:
            setattr(robot, _ATTR[k], float(v))


def replay(robot, ref, fps):
    a = ref["action"]
    period, sat = 1.0 / fps, 0
    q = np.zeros((len(a), NJ)); p = np.zeros((len(a), 3))
    f0 = robot.robot_manager.recovery_counts().get(ARM, 0)
    for i in range(len(a)):
        t0 = time.perf_counter()
        kin = robot.robot_manager.current_kinematic_state_batch([ARM])
        robot._cached_kin_state = kin
        q[i], _, _, p[i], _, _ = kin[ARM]
        tau = np.asarray(robot.robot_manager.torque_snapshot(ARM)[0], dtype=np.float64)
        sat += bool(np.any(np.abs(tau) >= 0.99 * _TAU_LIMIT))
        dp = a[i][0:3] * DELTA_POS_MAX
        dq = Rotation.from_rotvec(a[i][3:6] * DELTA_ROT_MAX).as_quat()
        robot.send_action({"r_x": float(dp[0]), "r_y": float(dp[1]), "r_z": float(dp[2]),
                           "r_qx": float(dq[0]), "r_qy": float(dq[1]), "r_qz": float(dq[2]),
                           "r_qw": float(dq[3]), "r_gripper": 0.0,
                           "kp": ref["kp_action"], "kd": ref["kd_action"]})
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
    faults = robot.robot_manager.recovery_counts().get(ARM, 0) - f0
    return q, p, int(faults), sat / max(len(a), 1)


def trial(robot, ref, vals, fps, max_sat) -> dict | None:
    if not home_verified(robot, ref["qpos"][0]):
        print("      HOMING FAILED - skipped")
        return None
    apply_knobs(robot, vals)
    q, p, faults, sat = replay(robot, ref, fps)
    s = score(ref, q, p)
    # Past the clamp the arm is under maximum force, not under the control law; and a
    # trial that FAULTED measures reflex recovery, not the law either. Both disqualify.
    s.update(config=vals, faults=faults, sat=sat, valid=sat <= max_sat and faults == 0)
    return s


# ----------------------------------------------------------------------- spec

def parse_sweep(spec: str):
    lhs, _, rhs = spec.partition("=")
    idx = None
    if "[" in lhs:
        lhs, _, tail = lhs.partition("[")
        idx = int(tail.rstrip("]"))
    lhs = lhs.strip()
    if lhs not in _VEC + _SCALAR:
        raise SystemExit(f"unknown knob {lhs!r}; choose from {_VEC + _SCALAR}")
    return lhs, idx, [float(v) for v in rhs.split(",")]


def with_value(base: dict, knob, idx, v) -> dict:
    out = dict(base)
    if knob in _VEC:
        vec = np.array(base[knob], dtype=np.float64)
        if idx is None:
            vec[:] = v
        else:
            vec[idx] = v
        out[knob] = vec
    else:
        out[knob] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--max-saturation", type=float, default=0.05)
    ap.add_argument("--score-only", metavar="RUN_DIR",
                    help="score an existing sysid.py output directory; no hardware")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        with h5py.File(Path(args.ref).expanduser(), "r") as f:
            names = set(f[sorted(f.keys())[0]].keys())
        for path in sorted(glob.glob(str(Path(args.score_only).expanduser() / "*.hdf5"))):
            stem = Path(path).stem
            name = stem.split("_record_", 1)[1] if "_record_" in stem else stem
            if name not in names:
                continue
            ref = load_ref(args.ref, sorted(names).index(name))
            with h5py.File(path, "r") as rf:
                g = rf["data/episode_0"] if "data/episode_0" in rf else rf
                q, p = np.asarray(g["qpos"]), np.asarray(g["eef_pos"])
            print(f"\n{name}")
            report(score(ref, q, p), verbose=True)
        return

    ref = load_ref(args.ref, args.traj)
    sweeps = [parse_sweep(s) for s in args.sweep]
    n_trials = max(1, sum(len(v) for _, _, v in sweeps) * args.passes)
    print(f"reference {ref['name']} ({len(ref['action'])} steps at {args.fps} Hz)")
    print(f"{n_trials} trial(s), ~{n_trials * (len(ref['action']) / args.fps + 12) / 60:.0f}"
          " min including homing. CLEAR THE WORKSPACE.")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        return

    c = rig_config()
    base = {k: (np.array(getattr(c, k), dtype=np.float64) if k in _VEC
                else float(getattr(c, k))) for k, _, _ in sweeps} if sweeps else {}
    robot = SingleArmFranka(rig_config())
    robot.connect()
    best, best_s = dict(base), None
    try:
        if not sweeps:
            s = trial(robot, ref, {}, args.fps, args.max_saturation)
            if s:
                report(s, verbose=True)
            return
        for p in range(args.passes):
            for knob, idx, vals in sweeps:
                print(f"\npass {p+1}: {knob}{'' if idx is None else f'[{idx}]'}")
                for v in vals:
                    cand = with_value(best, knob, idx, v)
                    s = trial(robot, ref, cand, args.fps, args.max_saturation)
                    if not s:
                        continue
                    flag = "" if s["valid"] else "  [SATURATED, excluded]"
                    print(f"    {knob}={v}: task {s['task']:.2f} nullspace "
                          f"{s['nullspace']:.2f} faults {s['faults']} "
                          f"sat {100*s['sat']:.0f}%{flag}")
                    # Closest to 1.0 wins, not largest: overshoot is as wrong as lag.
                    if s["valid"] and (best_s is None
                                       or abs(s["task"] - 1) < abs(best_s["task"] - 1)):
                        best, best_s = cand, s
        if best_s:
            print("\nbest:")
            print("  " + "  ".join(f"{k}={np.round(v, 3)}" for k, v in best.items()))
            report(best_s, verbose=True)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
