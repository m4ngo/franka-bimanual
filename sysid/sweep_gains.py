#!/usr/bin/env python3
"""Search the OSC gain scales for the best real-vs-sim match on ONE trajectory.

    python sysid/sweep_gains.py --yes
    python sysid/sweep_gains.py sysid/data.hdf5 --traj 0 --yes
    python sysid/sweep_gains.py --sweep 'kp_ori_scale[2]=1,3,6,10' --sweep friction_kc=0,0.5,0.9 --yes

LOWER total_err IS BETTER. It is a normalised tracking error, not a fit score:
0 is a perfect match. The "frozen arm" floor -- what an arm that never moved
would score -- depends on the reference, so the script measures it for the file
you loaded and prints it before starting. Any trial scoring near or above that
floor did not track; a trial ABOVE it moved in the wrong direction.

Replays one sim trajectory over and over, each pass with a different gain
configuration, and scores the EE motion against the sim reference. No modelling
-- just search. Complements scripts/sweep_sim_match.py, which searches the
delta fudges and friction_kc; this one searches the gains those were papering
over.

Knobs are set LIVE on the connected robot between trials (send_action reads
them per step), so the whole sweep runs on one connection and one arming.

Two things this refuses to do
-----------------------------
- Score a trial that started from the wrong pose. Homing is verified, not
  trusted: the arm's inertia depends on configuration, so a trial that began
  0.02 rad off measures a different plant.
- Score a trial that SATURATED THE TORQUE CLAMP. Past the clamp the arm is
  under maximum force, not under the OSC law, so the number would describe the
  clamp rather than the gains -- and that is exactly the regime that trips
  power_limit_violation / joint_velocity_violation on libfranka's side. Trials
  over --max-saturation are reported and excluded from the ranking.

Scoring is START-RELATIVE (cancels the constant frame offset between the sim
recorder and the robot) and PER AXIS, because the axes do not fail together:
lambda_ori on this arm is 0.028/0.031/0.0019 kg m^2, so at sim gains the yaw
moment is ~0.14 Nm against joint 7's 0.41 Nm breakaway and yaw alone cannot
start. A scalar score hides that; rot_rms_axis does not.

Clear the workspace -- this moves the arm continuously for the whole sweep.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot_robot_bimanual_franka import ControlMode, SingleArmFranka, SingleArmFrankaConfig
from lerobot_robot_bimanual_franka.osc_torque_controller import (
    DELTA_POS_MAX, DELTA_ROT_MAX, JOINT_TORQUE_LIMITS,
)

ARM = "r"
RIG = dict(r_server_ip="192.168.3.10", r_robot_ip="192.168.201.10",
           r_gripper_ip="192.168.201.10", r_port=18812)

# Mirrors pylibfranka_control._TAU_SAFETY_FACTOR; the clamp the loop applies.
_TAU_SAFETY_FACTOR = 0.8
_TAU_LIMIT = np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64) * _TAU_SAFETY_FACTOR

# Knobs this can sweep, and how to apply them to a live robot.
_VECTOR_KNOBS = ("kp_pos_scale", "kp_ori_scale", "kd_pos_scale", "kd_ori_scale")
_SCALAR_KNOBS = ("friction_kc", "ee_translation_fudge", "ee_rotation_fudge")
_ATTR = {"kp_pos_scale": "_kp_pos_scale", "kp_ori_scale": "_kp_ori_scale",
         "kd_pos_scale": "_kd_pos_scale", "kd_ori_scale": "_kd_ori_scale",
         "ee_translation_fudge": "_trans_fudge", "ee_rotation_fudge": "_rot_fudge"}

# Default search: the axes that actually fail on this arm. yaw first, because
# nothing else can be judged from a pose the arm never reached.
_DEFAULT_SWEEPS = [
    "friction_kc=0.5,0.9,1.5,2.0",
    "kp_ori_scale[2]=1,2,4,8",
    "kd_ori_scale[2]=1,1.5,2",
    "kp_pos_scale=1,1.5,2",
]


# ----------------------------------------------------------------- reference

def load_reference(path: str, index: int) -> dict:
    """One trajectory, plus the gain actions that reproduce the sim's own gains."""
    p = Path(path).expanduser()
    with h5py.File(p, "r") as f:
        group = f[sorted(f.keys())[0]]
        name = sorted(group.keys())[index]
        g = group[name]
        ref = dict(
            name=name,
            action=g["action"][:].astype(np.float64),
            eef_pos=g["eef_pos"][:].astype(np.float64),
            eef_quat=g["eef_quat"][:].astype(np.float64),
            # Home from the trajectory's own first qpos: not every reference
            # file carries an init_qpos attr, but every one carries qpos.
            init_qpos=g["qpos"][0].astype(np.float64) if "qpos" in g else None,
        )
        # Older files store the bare class name here, newer ones a JSON blob.
        cfg = {}
        raw = f.attrs.get("controller_cfg")
        if raw is not None:
            try:
                cfg = json.loads(raw)
            except (TypeError, ValueError):
                cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
    if ref["init_qpos"] is None:
        raise SystemExit(f"{p}:{name} has no qpos; cannot home to the reference start pose")
    # Invert the exponential remap so the real run uses the gains the sim was
    # generated with rather than whatever is typed. Absent config -> sim defaults.
    kp_sim = float(cfg.get("kp", 150.0))
    damp_sim = float(cfg.get("damping_ratio", 1.0))
    ref["kp_action"] = float(np.log10(kp_sim / 150.0))
    ref["kd_action"] = float(np.log10(damp_sim / 1.0))
    return ref


# --------------------------------------------------------------------- score

def _relative(pos: np.ndarray, quat: np.ndarray):
    """Motion relative to the first sample; cancels constant frame offsets."""
    rot = Rotation.from_quat(quat)
    return pos - pos[0], (rot[0].inv() * rot).as_rotvec()


def score(real_pos, real_quat, ref_pos, ref_quat) -> dict:
    n = min(len(real_pos), len(ref_pos))
    dp_r, rv_r = _relative(real_pos[:n], real_quat[:n])
    dp_s, rv_s = _relative(ref_pos[:n], ref_quat[:n])
    pos_e, rot_e = dp_r - dp_s, rv_r - rv_s

    def amp(a, b):
        pb = float(np.ptp(b))
        return float(np.ptp(a) / pb) if pb > 1e-9 else float("nan")

    return dict(
        pos_rms_m=float(np.sqrt((np.linalg.norm(pos_e, axis=1) ** 2).mean())),
        rot_rms_rad=float(np.sqrt((np.linalg.norm(rot_e, axis=1) ** 2).mean())),
        # Weighted the way the action space weights the two channels.
        total_err=float(np.sqrt((np.linalg.norm(pos_e, axis=1) ** 2).mean()) / DELTA_POS_MAX
                        + np.sqrt((np.linalg.norm(rot_e, axis=1) ** 2).mean()) / DELTA_ROT_MAX),
        pos_rms_axis=[float(np.sqrt((pos_e[:, i] ** 2).mean())) for i in range(3)],
        rot_rms_axis=[float(np.sqrt((rot_e[:, i] ** 2).mean())) for i in range(3)],
        # Right shape wrong size (a gain fixes it) vs wrong shape (it cannot).
        rot_amp_ratio=[amp(rv_r[:, i], rv_s[:, i]) for i in range(3)],
        pos_amp_ratio=[amp(dp_r[:, i], dp_s[:, i]) for i in range(3)],
    )


# ------------------------------------------------------------------ hardware

def home_verified(robot, target_q, tol_rad=0.005, attempts=3, max_time_s=20.0) -> bool:
    """Home and CHECK. home() returning True is not enough: a trial started from
    the wrong configuration measures a different inertia, and that spread has
    previously swamped the entire signal a sweep was trying to measure."""
    for k in range(attempts):
        robot.home(home_q_left=None, home_q_right=target_q,
                   max_time_s=max_time_s, tol_rad=tol_rad, fps=30)
        q = np.asarray(robot.robot_manager.current_kinematic_state(ARM)[0], dtype=np.float64)
        err = float(np.max(np.abs(q - target_q)))
        if err < tol_rad:
            return True
        print(f"      homing attempt {k + 1}/{attempts}: max joint error {err:.4f} rad")
    return False


def apply_knobs(robot, cfgvals: dict) -> None:
    """Set the gain knobs on the live robot. send_action reads these per step,
    so no reconnect is needed between trials."""
    for knob, val in cfgvals.items():
        if knob == "friction_kc":
            robot.robot_manager.set_tuning_all(friction_kc=float(val))
        elif knob in _VECTOR_KNOBS:
            setattr(robot, _ATTR[knob], np.asarray(val, dtype=np.float64))
        else:
            setattr(robot, _ATTR[knob], float(val))


def replay(robot, ref, fps) -> tuple[np.ndarray, np.ndarray, int, float]:
    """One pass of the reference action sequence.

    Returns (pos, quat, faults, saturated_fraction). Saturation is measured from
    the torque the loop actually wrote (post clamp and rate limit), which the
    state bundle already carries -- no extra round-trip.
    """
    action_all = ref["action"]
    n = len(action_all)
    period = 1.0 / fps
    pos, quat = np.zeros((n, 3)), np.zeros((n, 4))
    faults0 = robot.robot_manager.recovery_counts().get(ARM, 0)
    sat = 0

    for step in range(n):
        t0 = time.perf_counter()
        kin = robot.robot_manager.current_kinematic_state_batch([ARM])
        robot._cached_kin_state = kin
        _, _, _, ee_pos, ee_quat, _ = kin[ARM]
        pos[step], quat[step] = ee_pos, ee_quat

        tau_cmd = np.asarray(robot.robot_manager.torque_snapshot(ARM)[0], dtype=np.float64)
        if np.any(np.abs(tau_cmd) >= 0.99 * _TAU_LIMIT):
            sat += 1

        dpos = action_all[step][0:3] * DELTA_POS_MAX
        dq = Rotation.from_rotvec(action_all[step][3:6] * DELTA_ROT_MAX).as_quat()
        robot.send_action({
            "r_x": float(dpos[0]), "r_y": float(dpos[1]), "r_z": float(dpos[2]),
            "r_qx": float(dq[0]), "r_qy": float(dq[1]), "r_qz": float(dq[2]), "r_qw": float(dq[3]),
            "r_gripper": 0.0, "kp": ref["kp_action"], "kd": ref["kd_action"],
        })
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)

    faults = robot.robot_manager.recovery_counts().get(ARM, 0) - faults0
    return pos, quat, faults, sat / max(n, 1)


def run_trial(robot, ref, cfgvals: dict, fps: float, max_sat: float) -> dict | None:
    if not home_verified(robot, ref["init_qpos"]):
        print("      HOMING FAILED - skipping trial")
        return None
    apply_knobs(robot, cfgvals)
    pos, quat, faults, sat = replay(robot, ref, fps)
    r = score(pos, quat, ref["eef_pos"], ref["eef_quat"])
    r.update(config={k: (list(np.atleast_1d(v)) if k in _VECTOR_KNOBS else float(v))
                     for k, v in cfgvals.items()},
             faults=int(faults), saturated=float(sat))
    # Past the clamp the arm is under maximum force, not under the control law.
    r["valid"] = bool(sat <= max_sat)
    r["_traj"] = (pos, quat)
    return r


# ---------------------------------------------------------------------- spec

def parse_sweep(spec: str) -> tuple[str, int | None, list[float]]:
    """'kp_ori_scale[2]=1,2,4' -> ('kp_ori_scale', 2, [1.0, 2.0, 4.0]).

    The index form matters: the rotation axes do not fail together, so sweeping
    all three at once cannot separate a yaw that will not start from a roll that
    is already saturating.
    """
    if "=" not in spec:
        raise SystemExit(f"--sweep needs NAME=v1,v2,...  got {spec!r}")
    lhs, rhs = spec.split("=", 1)
    lhs, idx = lhs.strip(), None
    if lhs.endswith("]") and "[" in lhs:
        lhs, i = lhs[:-1].split("[", 1)
        idx = int(i)
    if lhs not in _VECTOR_KNOBS + _SCALAR_KNOBS:
        raise SystemExit(f"unknown knob {lhs!r}; choose from {_VECTOR_KNOBS + _SCALAR_KNOBS}")
    if idx is not None and lhs not in _VECTOR_KNOBS:
        raise SystemExit(f"{lhs} is scalar; drop the [{idx}]")
    return lhs, idx, [float(v) for v in rhs.split(",") if v.strip()]


def with_value(base: dict, knob: str, idx: int | None, value: float) -> dict:
    out = {k: (np.array(v, dtype=np.float64) if isinstance(v, np.ndarray) else v)
           for k, v in base.items()}
    if idx is None:
        out[knob] = np.full(3, value) if knob in _VECTOR_KNOBS else value
    else:
        vec = np.array(out[knob], dtype=np.float64)
        vec[idx] = value
        out[knob] = vec
    return out


def fmt(r: dict) -> str:
    cfg = " ".join(f"{k}={np.round(v, 2)}" for k, v in r["config"].items())
    flag = "" if r["valid"] else "  [SATURATED - excluded]"
    return (f"{cfg}\n        ERR {r['total_err']:7.3f}  pos {1000 * r['pos_rms_m']:6.1f}mm  "
            f"rot {np.degrees(r['rot_rms_rad']):6.2f}deg  "
            f"| rot rms/axis {np.round(np.degrees(r['rot_rms_axis']), 2)}deg  "
            f"amp {np.round(r['rot_amp_ratio'], 2)}  "
            f"| faults {r['faults']} sat {100 * r['saturated']:.0f}%{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?", default="sysid/data.hdf5",
                    help="reference HDF5 (default: sysid/data.hdf5)")
    ap.add_argument("--traj", type=int, default=0, help="trajectory index (default: 0, the first)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--sweep", action="append", default=None,
                    help="NAME=v1,v2,... or NAME[i]=v1,v2,... Repeatable. "
                         f"Default: {_DEFAULT_SWEEPS}")
    ap.add_argument("--mode", choices=("coord", "grid"), default="coord",
                    help="coord: one knob at a time, keeping the best (default). "
                         "grid: full product -- trials multiply fast on hardware.")
    ap.add_argument("--passes", type=int, default=1, help="coordinate-descent passes")
    ap.add_argument("--repeats", type=int, default=1,
                    help="trials per config, ranked by MEDIAN. Raise this before "
                         "trusting a small difference between configs.")
    ap.add_argument("--max-saturation", type=float, default=0.05,
                    help="reject a trial with more than this fraction of steps at the "
                         "torque clamp (default 0.05); such a trial measures the clamp")
    ap.add_argument("--out", default="~/sysid/outputs/sweep_gains.json")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    sweeps = [parse_sweep(s) for s in (args.sweep or _DEFAULT_SWEEPS)]
    ref = load_reference(args.ref, args.traj)

    n_cfg = (int(np.prod([len(v) for _, _, v in sweeps])) if args.mode == "grid"
             else sum(len(v) for _, _, v in sweeps) * args.passes)
    # What an arm that never moved would score on THIS reference: the number a
    # trial has to beat before it has demonstrated anything at all.
    n_ref = len(ref["eef_pos"])
    floor = score(np.repeat(ref["eef_pos"][:1], n_ref, 0),
                  np.repeat(ref["eef_quat"][:1], n_ref, 0),
                  ref["eef_pos"], ref["eef_quat"])["total_err"]

    print(f"reference {args.ref}:{ref['name']}  ({len(ref['action'])} steps at {args.fps} Hz)")
    print(f"sim gains: kp_action={ref['kp_action']:.3f} kd_action={ref['kd_action']:.3f}")
    print(f"frozen-arm floor for this reference: total_err {floor:.2f} "
          "(a trial at or above this did not track)")
    print(f"{n_cfg} configs x {args.repeats} repeats, "
          f"~{n_cfg * args.repeats * (len(ref['action']) / args.fps + 12) / 60:.0f} min "
          "including homing. CLEAR THE WORKSPACE.")
    for knob, idx, vals in sweeps:
        print(f"  {knob}{'' if idx is None else f'[{idx}]'} = {vals}")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        return

    # Start from the config's own values so the sweep is a delta from what flies.
    c = SingleArmFrankaConfig
    base: dict = {}
    for knob, _, _ in sweeps:
        if knob in _VECTOR_KNOBS:
            base[knob] = np.array(getattr(c, knob), dtype=np.float64)
        else:
            base[knob] = float(getattr(c, knob))

    robot = SingleArmFranka(SingleArmFrankaConfig(
        **RIG, control_mode=ControlMode.EE_DELTA, cameras={}, depth=False, depth_cam={}))
    robot.connect()
    results: list[dict] = []
    best = dict(base)
    try:
        if args.mode == "grid":
            for combo in itertools.product(*[[(k, i, v) for v in vals] for k, i, vals in sweeps]):
                cand = dict(base)
                for k, i, v in combo:
                    cand = with_value(cand, k, i, v)
                r = run_repeats(robot, ref, cand, args.fps, args.repeats, args.max_saturation)
                if r:
                    results.append(r)
                    print("    " + fmt(r))
        else:
            for p in range(args.passes):
                for knob, idx, vals in sweeps:
                    label = f"{knob}{'' if idx is None else f'[{idx}]'}"
                    print(f"\n  pass {p + 1}: sweeping {label}")
                    trials = []
                    for v in vals:
                        cand = with_value(best, knob, idx, v)
                        r = run_repeats(robot, ref, cand, args.fps, args.repeats,
                                        args.max_saturation)
                        if r:
                            trials.append(r)
                            results.append(r)
                            print("    " + fmt(r))
                    usable = [t for t in trials if t["valid"]]
                    if not usable:
                        print(f"    -> every {label} trial saturated; keeping "
                              f"{np.round(best[knob], 2)}")
                        continue
                    win = min(usable, key=lambda t: t["total_err"])
                    best = with_value(best, knob, idx,
                                      win["config"][knob][idx] if idx is not None
                                      else (win["config"][knob][0] if knob in _VECTOR_KNOBS
                                            else win["config"][knob]))
                    print(f"    -> {label} = {np.round(best[knob], 3)}  "
                          f"(err {win['total_err']:.3f})")
    finally:
        try:
            home_verified(robot, ref["init_qpos"])
        finally:
            robot.disconnect()

    usable = [r for r in results if r["valid"]]
    if not usable:
        print("\nno usable trials (every one saturated the clamp, or homing failed). "
              "Lower the gains before sweeping.")
    else:
        usable.sort(key=lambda r: r["total_err"])
        print("\n=== best 5 (LOWEST error) ===")
        for r in usable[:5]:
            print("  " + fmt(r))
        rejected = len(results) - len(usable)
        if rejected:
            print(f"\n{rejected} trial(s) excluded for saturating the torque clamp.")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    best_traj = usable[0].pop("_traj", None) if usable else None
    for r in results:
        r.pop("_traj", None)
    out.write_text(json.dumps({
        "reference": str(args.ref), "traj": ref["name"],
        "kp_action": ref["kp_action"], "kd_action": ref["kd_action"],
        "max_saturation": args.max_saturation, "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    if best_traj is not None:
        npz = out.with_suffix(".npz")
        np.savez(npz, real_pos=best_traj[0], real_quat=best_traj[1],
                 sim_pos=ref["eef_pos"], sim_quat=ref["eef_quat"], action=ref["action"])
        print(f"wrote {npz}  (best trial vs sim, for plotting)")


def run_repeats(robot, ref, cfgvals, fps, repeats, max_sat) -> dict | None:
    """Median of `repeats` trials; the spread is kept so noise stays visible."""
    got = [t for t in (run_trial(robot, ref, cfgvals, fps, max_sat) for _ in range(repeats))
           if t is not None]
    if not got:
        return None
    got.sort(key=lambda r: r["total_err"])
    med = got[len(got) // 2]
    med["n_trials"] = len(got)
    med["err_spread"] = got[-1]["total_err"] - got[0]["total_err"]
    return med


if __name__ == "__main__":
    main()
