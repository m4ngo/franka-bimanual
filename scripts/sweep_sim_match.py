#!/usr/bin/env python3
"""Empirically tune friction_kc and the delta fudges until the real arm's motion
matches a sim reference trajectory.

LOWER total_err IS BETTER -- it is a normalised error, not a fit score. 0 is a
perfect match; a frozen arm on this reference scores ~2.8.

Replays ONE sim trajectory (default: the first in the file) over and over, each
time with a different (friction_kc, translation_fudge, rotation_fudge), and
scores the resulting EE motion against sim. No modelling -- just search.

Scoring compares motion RELATIVE TO THE START POSE, which cancels the constant
frame offsets between the sim recorder and the robot (the sim file's
eef_goal_quat sits 90 deg about Z from its own eef_quat). Position error is
normalised by 0.05 m and rotation error by 0.5 rad -- osc_pose.json's
output_max -- so the two are weighted the way the action space weights them.

Gains come from the reference file's own controller_cfg, so the real run always
uses the kp/damping the sim was generated with rather than whatever was typed.

    python scripts/sweep_sim_match.py --yes
    python scripts/sweep_sim_match.py --ref ~/sysid/<run>/data.hdf5
    python scripts/sweep_sim_match.py --mode grid --yes      # full grid, slower

Every trial re-homes to the reference's own init_qpos and verifies it got there;
a trial whose homing fails is skipped rather than silently scored.
Clear the workspace -- this moves the arm continuously for the whole sweep.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_bimanual_franka import ControlMode, SingleArmFranka, SingleArmFrankaConfig
from lerobot_robot_bimanual_franka.osc_torque_controller import DELTA_POS_MAX, DELTA_ROT_MAX

ARM = "r"

# uncouple_pos_ori follows the robot config so the sweep measures the controller
# we actually fly, not sim's setting. False raises commanded torque a lot; that is
# bounded by LAMBDA_DLS_MU now, but a high ori_scale on top can still saturate the
# clamp, which is maximum-force motion -- watch clamp_trips in the server log.
RIG = dict(r_server_ip="192.168.3.10", r_robot_ip="192.168.201.10",
           r_gripper_ip="192.168.201.10", r_port=18812)


# ----------------------------------------------------------------- reference

def load_reference(path: str, index: int) -> dict:
    with h5py.File(Path(path).expanduser(), "r") as f:
        cfg = json.loads(f.attrs["controller_cfg"])
        names = sorted(f["data"].keys())
        name = names[index]
        g = f[f"data/{name}"]
        ref = dict(name=name, action=g["action"][:].astype(np.float64),
                   eef_pos=g["eef_pos"][:].astype(np.float64),
                   eef_quat=g["eef_quat"][:].astype(np.float64),
                   init_qpos=np.array(json.loads(str(f.attrs["init_qpos"])), dtype=np.float64),
                   control_freq=float(f.attrs.get("control_freq", 20)))
    # The sim ran impedance_mode="fixed", so its gains are constants. Invert the
    # exponential remap to get the action values that reproduce them on our side.
    ref["kp_action"] = float(np.log10(cfg["kp"] / 150.0))
    ref["kd_action"] = float(np.log10(cfg["damping_ratio"] / 1.0))
    ref["cfg"] = cfg
    return ref


def relative_motion(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """Motion relative to the first sample; cancels constant frame offsets."""
    rot = Rotation.from_quat(quat)
    return pos - pos[0], rot[0].inv() * rot


def score(real_pos, real_quat, ref_pos, ref_quat) -> dict:
    """Per-step trajectory error over the WHOLE episode, not endpoint error.

    Both series are 200 samples at the same 20 Hz, so they align by index. The
    headline numbers are RMS over every step; endpoint-only agreement is
    meaningless here because a sine sweep returns near its start, so an arm that
    never moved would score perfectly on final pose alone.

    Reported alongside:
      rms/max/mean  - per-step error, the thing being minimised
      per-axis rms  - which axis is wrong, not just that something is
      amp_ratio     - real peak-to-peak / sim peak-to-peak. Separates "right
                      shape, wrong size" (a fudge can fix it) from "wrong
                      shape" (it cannot).
      corr          - correlation of the two signals over time. High corr with
                      low amp_ratio is exactly the case a gain fudge repairs.
    """
    n = min(len(real_pos), len(ref_pos))
    dp_r, dr_r = relative_motion(real_pos[:n], real_quat[:n])
    dp_s, dr_s = relative_motion(ref_pos[:n], ref_quat[:n])
    rv_r, rv_s = dr_r.as_rotvec(), dr_s.as_rotvec()

    pos_step = np.linalg.norm(dp_r - dp_s, axis=1)
    rot_step = (dr_s.inv() * dr_r).magnitude()

    def _corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 1e-12 else 0.0

    def _amp(a, b):
        pb = float(np.ptp(b))
        return float(np.ptp(a) / pb) if pb > 1e-9 else float("nan")

    return dict(
        # headline: RMS per-step error across the trajectory
        pos_rms_m=float(np.sqrt((pos_step ** 2).mean())),
        rot_rms_rad=float(np.sqrt((rot_step ** 2).mean())),
        pos_max_m=float(pos_step.max()), rot_max_rad=float(rot_step.max()),
        pos_err_m=float(pos_step.mean()), rot_err_rad=float(rot_step.mean()),
        total_err=float(np.sqrt((pos_step ** 2).mean()) / DELTA_POS_MAX
                    + np.sqrt((rot_step ** 2).mean()) / DELTA_ROT_MAX),
        pos_rms_axis=[float(np.sqrt(((dp_r[:, i] - dp_s[:, i]) ** 2).mean())) for i in range(3)],
        rot_rms_axis=[float(np.sqrt(((rv_r[:, i] - rv_s[:, i]) ** 2).mean())) for i in range(3)],
        pos_amp_ratio=[_amp(dp_r[:, i], dp_s[:, i]) for i in range(3)],
        rot_amp_ratio=[_amp(rv_r[:, i], rv_s[:, i]) for i in range(3)],
        pos_corr=[_corr(dp_r[:, i], dp_s[:, i]) for i in range(3)],
        rot_corr=[_corr(rv_r[:, i], rv_s[:, i]) for i in range(3)],
        rot_p2p_real=float(np.ptp(dr_r.magnitude())), rot_p2p_sim=float(np.ptp(dr_s.magnitude())),
    )


# ----------------------------------------------------------------- hardware

def home_reliably(robot, target_q, tol_rad=0.005, attempts=3, max_time_s=20.0) -> bool:
    """Home and VERIFY. home() returning True is not enough on its own -- it can
    time out and, before the per-joint impedance fix, could leave the wrist
    short. A trial started from the wrong pose is worse than a skipped one.

    tol_rad is tight on purpose: at the old 0.02 rad (1.1 deg/joint) the start
    pose varied enough between trials to change the arm's inertia, and repeats
    of one config spread by 0.40 total_err -- an order of magnitude more than
    the 0.04 that separated the sweep's "best" from a frozen arm. Every knob
    choice that sweep made was noise.
    """
    for k in range(attempts):
        robot.home(home_q_left=None, home_q_right=target_q,
                   max_time_s=max_time_s, tol_rad=tol_rad, fps=30)
        q = np.asarray(robot.robot_manager.current_kinematic_state(ARM)[0], dtype=np.float64)
        err = float(np.max(np.abs(q - target_q)))
        if err < tol_rad:
            return True
        print(f"      homing attempt {k + 1}/{attempts}: max joint error {err:.4f} rad, retrying")
    return False


def replay(robot, ref, kp_action, kd_action, fps) -> tuple[np.ndarray, np.ndarray, int]:
    """One pass of the reference action sequence; returns the EE trajectory."""
    action_all = ref["action"]
    n = len(action_all)
    period = 1.0 / fps
    pos, quat = np.zeros((n, 3)), np.zeros((n, 4))
    faults0 = robot.robot_manager.recovery_counts().get(ARM, 0)

    for step in range(n):
        t0 = time.perf_counter()
        kin = robot.robot_manager.current_kinematic_state_batch([ARM])
        robot._cached_kin_state = kin
        _, _, _, ee_pos, ee_quat, _ = kin[ARM]
        pos[step], quat[step] = ee_pos, ee_quat

        dpos = action_all[step][0:3] * DELTA_POS_MAX
        dq = Rotation.from_rotvec(action_all[step][3:6] * DELTA_ROT_MAX).as_quat()
        robot.send_action({
            "r_x": float(dpos[0]), "r_y": float(dpos[1]), "r_z": float(dpos[2]),
            "r_qx": float(dq[0]), "r_qy": float(dq[1]), "r_qz": float(dq[2]), "r_qw": float(dq[3]),
            "r_gripper": 0.0, "kp": kp_action, "kd": kd_action,
        })
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)

    faults = robot.robot_manager.recovery_counts().get(ARM, 0) - faults0
    return pos, quat, faults


def run_repeats(robot, ref, kc, tf, rf, oscale, fps, repeats) -> dict | None:
    """Median of `repeats` trials, with the spread kept so noise stays visible."""
    got = []
    for _ in range(repeats):
        r = run_trial(robot, ref, kc, tf, rf, oscale, fps)
        if r:
            got.append(r)
    if not got:
        return None
    got.sort(key=lambda r: r["total_err"])
    med = got[len(got) // 2]
    med["n_trials"] = len(got)
    med["err_spread"] = got[-1]["total_err"] - got[0]["total_err"]
    med["_traj"] = got[len(got) // 2].get("_traj")
    return med


def run_trial(robot, ref, kc, tf, rf, oscale, fps,
              uncouple=SingleArmFrankaConfig.uncouple_pos_ori) -> dict | None:
    if not home_reliably(robot, ref["init_qpos"]):
        print("      HOMING FAILED - skipping trial")
        return None
    robot.robot_manager.set_tuning_all(friction_kc=kc, uncouple_pos_ori=bool(uncouple))
    robot._trans_fudge, robot._rot_fudge = float(tf), float(rf)
    robot._kp_ori_scale = np.full(3, float(oscale))
    pos, quat, faults = replay(robot, ref, ref["kp_action"], ref["kd_action"], fps)
    r = score(pos, quat, ref["eef_pos"], ref["eef_quat"])
    r.update(friction_kc=kc, trans_fudge=tf, rot_fudge=rf, ori_scale=oscale, faults=faults)
    r["_traj"] = (pos, quat)      # stripped before JSON; kept for the npz dump
    return r


def fmt(r: dict) -> str:
    dom = int(np.argmax(np.abs(r["rot_amp_ratio"])))
    return (f"kc={r['friction_kc']:<4.2f} tf={r['trans_fudge']:<4.2f} rf={r['rot_fudge']:<4.2f} "
            f"os={r['ori_scale']:<5.1f} "
            f"| ERR {r['total_err']:7.3f}  posRMS {1000*r['pos_rms_m']:6.1f}mm  "
            f"rotRMS {np.degrees(r['rot_rms_rad']):5.2f}deg "
            f"| amp r/s {r['rot_amp_ratio'][dom]:5.2f}  corr {r['rot_corr'][dom]:+.2f}"
            f"  faults {r['faults']}"
            + (f"  spread {r['err_spread']:.3f} (n={r['n_trials']})" if "err_spread" in r else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",
                    default="~/sysid/sim_gain_grid_3x3_rotation/kp_actn0.50_damp_actn0.50/data.hdf5")
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--mode", choices=("coord", "grid"), default="coord")
    ap.add_argument("--kc", type=float, nargs="+", default=[0.6, 0.9, 1.2, 1.5])
    ap.add_argument("--tf", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--rf", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--os", dest="oscale", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0],
                    help="kp_ori_scale. THE knob for rotation: rot_fudge scales the error, which "
                         "enters the moment through the sin-based orientation_error and so saturates "
                         "at Lambda_ori*kp (~0.09 Nm here, under breakaway). This scales kp itself.")
    ap.add_argument("--passes", type=int, default=2, help="coordinate-descent passes")
    ap.add_argument("--repeats", type=int, default=3,
                    help="trials per config; configs are ranked by MEDIAN total_err. "
                         "1 is not enough -- repeat spread was 0.40 while the whole "
                         "signal between os=1 configs was 0.04.")
    ap.add_argument("--out", default="~/sysid/outputs/sweep_sim_match.json")
    # uncouple_pos_ori follows the robot config; see the module note.
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    ref = load_reference(args.ref, args.traj_index)
    n_trials = (len(args.kc) * len(args.tf) * len(args.rf) * len(args.oscale) if args.mode == "grid"
                else args.passes * (len(args.kc) + len(args.tf) + len(args.rf) + len(args.oscale)))
    n_trials *= args.repeats
    secs = n_trials * (len(ref["action"]) / args.fps + 12)
    print(f"reference : {args.ref}  traj[{args.traj_index}] = {ref['name']}")
    print(f"            {len(ref['action'])} steps, sim kp={ref['cfg']['kp']:.2f} "
          f"damping={ref['cfg']['damping_ratio']:.4f} "
          f"-> action kp={ref['kp_action']:+.3f} kd={ref['kd_action']:+.3f}")
    print(f"            init_qpos = {np.round(ref['init_qpos'], 4)}")
    print(f"plan      : {args.mode}, ~{n_trials} trials, ~{secs/60:.0f} min of continuous motion\n")

    if not args.yes and input("This MOVES the arm continuously. Workspace clear? [y/N] ").strip().lower() not in ("y", "yes"):
        print("aborted")
        return

    cfg = SingleArmFrankaConfig(**RIG, control_mode=ControlMode.EE_DELTA,
                                cameras={}, depth=False, depth_cam={})
    robot = SingleArmFranka(cfg)
    robot.connect()
    results: list[dict] = []
    try:
        best = dict(friction_kc=args.kc[0], trans_fudge=1.0, rot_fudge=1.0, ori_scale=1.0)
        if args.mode == "grid":
            combos = list(itertools.product(args.kc, args.tf, args.rf, args.oscale))
            for i, (kc, tf, rf, oc) in enumerate(combos):
                print(f"  [{i+1}/{len(combos)}] ", end="", flush=True)
                r = run_repeats(robot, ref, kc, tf, rf, oc, args.fps, args.repeats)
                if r:
                    results.append(r); print(fmt(r))
        else:
            for p in range(args.passes):
                # Order matters: sweep the knob that can actually move the
                # dominant axis first, so later knobs are tuned around a pose
                # that is at least roughly tracking.
                for knob, values in (("ori_scale", args.oscale), ("friction_kc", args.kc),
                                     ("rot_fudge", args.rf), ("trans_fudge", args.tf)):
                    print(f"  pass {p+1} sweeping {knob}")
                    trials = []
                    for v in values:
                        cand = dict(best); cand[knob] = v
                        print("    ", end="", flush=True)
                        r = run_repeats(robot, ref, cand["friction_kc"], cand["trans_fudge"],
                                        cand["rot_fudge"], cand["ori_scale"], args.fps,
                                        args.repeats)
                        if r:
                            trials.append(r); results.append(r); print(fmt(r))
                    if trials:
                        best_t = min(trials, key=lambda r: r["total_err"])
                        best[knob] = best_t[knob]
                        print(f"    -> {knob} = {best[knob]}  (score {best_t['total_err']:.3f})")
    finally:
        try:
            home_reliably(robot, ref["init_qpos"])
        finally:
            robot.disconnect()

    if results:
        results.sort(key=lambda r: r["total_err"])
        print("\n=== best 5 (LOWEST error) ===")
        for r in results[:5]:
            print("  " + fmt(r))
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        best_traj = results[0].pop("_traj", None)
        for r in results:
            r.pop("_traj", None)
        out.write_text(json.dumps({"reference": args.ref, "traj": ref["name"],
                                   "kp_action": ref["kp_action"], "kd_action": ref["kd_action"],
                                   "results": results}, indent=2))
        print(f"\nwrote {out}")
        if best_traj is not None:
            npz = out.with_suffix(".npz")
            np.savez(npz, real_pos=best_traj[0], real_quat=best_traj[1],
                     sim_pos=ref["eef_pos"], sim_quat=ref["eef_quat"],
                     action=ref["action"], best=json.dumps(results[0]))
            print(f"wrote {npz}  (best trial's trajectory vs sim, for plotting)")
    else:
        print("\nno trials completed (homing failed every time?)")


if __name__ == "__main__":
    main()
