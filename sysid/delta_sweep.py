#!/usr/bin/env python3
"""Task-level sim-vs-real check: EE travel per step against command amplitude.

    python sysid/delta_sweep.py --backend real --yes
    PYTHONPATH=franka_config multi-fast/.venv/bin/python \
        sysid/delta_sweep.py --backend sim
    python sysid/delta_sweep.py --compare <sim_dir> <real_dir>

joint_id.py measures the plant one joint at a time; this measures the thing the
policy actually drives. Per axis it holds a CONSTANT normalized EE_DELTA of
magnitude `a` for N policy steps and records how far the EE travelled, sweeping
`a` over the range real task actions occupy.

The point is the SHAPE of travel against amplitude, because it separates the two
failure modes that a single scalar cannot:

    ratio flat and ~1.0   -> matched
    ratio flat and != 1.0 -> a pure gain error; one scalar fixes it
    ratio falls as a->0   -> residual friction, a DEADBAND; no scalar fixes it

Fitting travel(a) = k*(a*output_max - d) recovers d directly, so the comparison
reports each side's deadband in metres rather than a single fudge that is only
correct at whichever amplitude happened to dominate the trajectory it was tuned on.

Both backends run their own shipped config -- that is the question being asked, so
neither side is normalised to the other. Clear the workspace before --backend real.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import franka_config as fc

OUT_ROOT = Path("~/sysid/outputs").expanduser()
AXES = ("x", "y", "z", "rx", "ry", "rz")
POS_MAX = float(fc.control("torque.delta.pos_max_m"))
ROT_MAX = float(fc.control("torque.delta.rot_max_rad"))


@dataclass
class Config:
    amps: tuple = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
    steps: int = 20
    fps: float = float(fc.control_fps())
    settle_s: float = 0.5
    axes: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    sign: float = 1.0
    max_travel_m: float = 0.15   # excursion cap per trial; see sweep()


def scale(axis: int) -> float:
    return POS_MAX if axis < 3 else ROT_MAX


def travel_of(p0, R0, p1, R1, axis: int) -> float:
    """Displacement PROJECTED ON the swept axis: metres for 0-2, radians for 3-5.

    Not the magnitude. EE_DELTA re-anchors its goal on the measured pose every step,
    so on an axis with zero commanded delta the error is zero and the controller
    applies no restoring force -- any residual disturbance drifts freely. A magnitude
    counts that drift as travel: a +x command that sagged 137 mm while advancing 61 mm
    read as 150 mm of "x travel".
    """
    if axis < 3:
        return float((np.asarray(p1) - np.asarray(p0))[axis])
    dR = np.asarray(R1) @ np.asarray(R0).T
    rv = np.zeros(3)
    ang = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    if ang > 1e-9:
        rv = ang / (2.0 * np.sin(ang)) * np.array(
            [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return float(rv[axis - 3])


# ------------------------------------------------------------------ backends

class SimBackend:
    name = "sim"

    def __init__(self):
        import robosuite
        from robosuite.controllers import load_controller_config
        cc = load_controller_config(default_controller="OSC_POSE")
        # Take the law from config/control.yaml, not from whatever osc_pose.json
        # happens to ship. This used to run stock -- uncouple=True, kp_limits
        # [0,300] -- against a real arm configured uncouple=False, so every "sim vs
        # real" verdict below had a CONTROL-LAW difference folded into what the
        # docstring calls a plant comparison.
        cc["uncouple_pos_ori"] = bool(fc.control("torque.osc.uncouple_pos_ori"))
        cc["kp"] = float(fc.control("torque.osc.default_kp"))
        cc["damping_ratio"] = float(fc.control("torque.osc.default_damping_ratio"))
        cc["kp_limits"] = list(fc.control("torque.osc.kp_limits"))
        cc["damping_ratio_limits"] = list(fc.control("torque.osc.damping_ratio_limits"))
        cc["output_max"] = [POS_MAX] * 3 + [ROT_MAX] * 3
        cc["output_min"] = [-POS_MAX] * 3 + [-ROT_MAX] * 3
        # Deliberately NOT synced: impedance_mode stays "fixed". The references in
        # sysid/*.hdf5 were collected under fixed, and at the kp=kd=0 action this
        # sweep sends, fixed and variable are the same numbers (kp=150, ratio=1) --
        # but variable changes robosuite's action LAYOUT to
        # [damping(6), kp(6), delta(6)], and step() below indexes the delta by axis.
        assert cc["impedance_mode"] == "fixed", cc["impedance_mode"]
        self.env = robosuite.make("Lift", robots="Panda", has_renderer=False,
                                  has_offscreen_renderer=False, use_camera_obs=False,
                                  controller_configs=cc, control_freq=int(fc.control_fps()))
        self.env.reset()
        assert self.env.robots[0].controller.control_dim == 6, "action layout moved"

    def reset(self, q0):
        self.env.reset()
        r = self.env.robots[0]
        self.env.sim.data.qpos[r._ref_joint_pos_indexes] = q0
        self.env.sim.data.qvel[r._ref_joint_vel_indexes] = 0.0
        self.env.sim.forward()
        r.controller.update(force=True)
        r.controller.reset_goal()

    def pose(self):
        self.env.sim.forward()
        c = self.env.robots[0].controller
        return np.array(c.ee_pos), np.array(c.ee_ori_mat)

    def q(self):
        r = self.env.robots[0]
        return np.asarray(self.env.sim.data.qpos[r._ref_joint_pos_indexes], dtype=np.float64)

    def step(self, axis, amp):
        a = np.zeros(self.env.action_dim)
        a[axis] = amp
        self.env.step(a)


class RealBackend:
    name = "real"

    def __init__(self, arm_key, profile):
        from lerobot_robot_bimanual_franka import (
            ControlMode, SingleArmFranka, SingleArmFrankaConfig)
        from scipy.spatial.transform import Rotation
        self._Rot = Rotation
        self.key = arm_key
        self.robot = SingleArmFranka(SingleArmFrankaConfig(
            control_mode=ControlMode.EE_DELTA, cameras={}, depth=False, depth_cam={}))
        self.robot.connect()

    def reset(self, q0, tol=0.02):
        # Verified, not fire-and-forget: a wrong start pose measures a different
        # Jacobian. But the tolerance has a FLOOR set by friction, not by patience:
        # the hold runs at joint_impedance.kp, so 0.6 Nm of friction leaves ~0.002 rad
        # of steady-state error on joints 1-4 and ~0.0034 on the wrist. Asking for
        # 0.005 rad cost 3 x 20 s of timeout per trial and then skipped the trial.
        for _ in range(2):
            # 25 s, not 10: once the arm actually tracks, a full-amplitude trial ends
            # a quarter-metre out, and homing that far at homing.max_qdot_rad_s does
            # not fit in 10 s. The timeout was sized when real barely moved.
            self.robot.home(home_q_left=None, home_q_right=q0, max_time_s=25.0,
                            tol_rad=tol, fps=30)
            q = np.asarray(self.robot.robot_manager.current_kinematic_state(self.key)[0],
                           dtype=np.float64)
            err = float(np.max(np.abs(q - q0)))
            if err < tol:
                return True
        print(f"      home converged only to {err:.4f} rad (tol {tol})")
        return False

    def pose(self):
        _, _, _, p, quat, _ = self.robot.robot_manager.current_kinematic_state(self.key)
        return np.asarray(p, dtype=np.float64), self._Rot.from_quat(np.asarray(quat)).as_matrix()

    def q(self):
        return np.asarray(self.robot.robot_manager.current_kinematic_state(self.key)[0],
                          dtype=np.float64)

    def step(self, axis, amp):
        dp = np.zeros(3)
        rv = np.zeros(3)
        if axis < 3:
            dp[axis] = amp * POS_MAX
        else:
            rv[axis - 3] = amp * ROT_MAX
        dq = self._Rot.from_rotvec(rv).as_quat()
        # Metric on this side; osc.py's normalize/scale already happened above.
        self.robot.send_action({
            f"{self.key}_x": float(dp[0]), f"{self.key}_y": float(dp[1]),
            f"{self.key}_z": float(dp[2]), f"{self.key}_qx": float(dq[0]),
            f"{self.key}_qy": float(dq[1]), f"{self.key}_qz": float(dq[2]),
            f"{self.key}_qw": float(dq[3]), f"{self.key}_gripper": 0.0,
            "kp": 0.0, "kd": 0.0})

    def probe(self):
        """(tau_cmd, saturated?) -- is the arm under the control law or under its clamp?

        A saturated joint is not tracking the goal, so travel under saturation measures
        the torque limit, not the plant. Recorded per step because the distinction is
        invisible in the endpoint displacement.

        `saturated` is the REAL hardware clamp (limits.joint_torque_nm). It is NOT the
        sim ctrlrange clip, and reading it as such is what made this instrument lie:
        sat_frac read 0.000 across every axis and amplitude, which is true of the
        hardware clamp and says nothing about sim's +/-12 Nm wrist limit -- the clip
        whose engagement decides the rotation roll-off. That one is counted inside
        run_controller and comes back through sim_clip_ticks().
        """
        tau = np.asarray(self.robot.robot_manager.torque_snapshot(self.key)[0],
                         dtype=np.float64)
        lim = np.asarray(fc.control("torque.limits.joint_torque_nm"), dtype=np.float64)
        return tau, bool(np.any(np.abs(tau) >= 0.99 * lim))

    def sim_clip_ticks(self) -> int:
        return self.robot.robot_manager.sim_clip_ticks(self.key)

    def faults(self):
        return int(self.robot.robot_manager.recovery_counts().get(self.key, 0))

    def close(self):
        self.robot.disconnect()


# ------------------------------------------------------------------- protocol

def sweep(backend, q0, cfg, out=None) -> dict:
    # Caller-owned so a Ctrl-C partway through still leaves every completed trial on
    # disk. Interrupting an odd-looking run is the normal way this gets used, and
    # losing the traces is exactly when they are most wanted.
    out = {} if out is None else out
    period = 1.0 / cfg.fps
    for axis in cfg.axes:
        rows = []
        out[AXES[axis]] = rows
        for amp in cfg.amps:
            if hasattr(backend, "reset") and backend.reset(q0) is False:
                print(f"  axis {AXES[axis]} amp {amp}: HOMING FAILED, skipped")
                continue
            time.sleep(cfg.settle_s) if backend.name == "real" else None
            p0, R0 = backend.pose()
            path, sat, taumax, taken = [], 0, 0.0, 0
            # Monotonic counter on the NUC, so take a difference over the trial rather
            # than a rate per step -- the clip is evaluated at 500 Hz and the sweep
            # steps at 20 Hz, so per-step polling would miss most of it.
            clip0 = backend.sim_clip_ticks() if hasattr(backend, "sim_clip_ticks") else 0
            for _ in range(cfg.steps):
                t0 = time.perf_counter()
                # Cap the excursion. A tracking arm covers 240 mm in 20 full-amplitude
                # steps, which runs it toward full extension where the Jacobian
                # degrades and homing back stops being cheap. Applied to BOTH backends
                # so the per-step averages stay over comparable arcs.
                if np.linalg.norm(np.asarray(backend.pose()[0]) - np.asarray(p0)) \
                        > cfg.max_travel_m:
                    break
                taken += 1
                backend.step(axis, cfg.sign * amp)
                if hasattr(backend, "probe"):
                    tau, s = backend.probe()
                    sat += int(s)
                    taumax = max(taumax, float(np.max(np.abs(tau))))
                path.append(np.round(backend.pose()[0], 6).tolist())
                if backend.name == "real":
                    dt = time.perf_counter() - t0
                    if dt < period:
                        time.sleep(period - dt)
            p1, R1 = backend.pose()
            d = travel_of(p0, R0, p1, R1, axis)
            # Per ACTUAL steps, not cfg.steps: a trial cut short by the travel cap
            # otherwise reads as a lower rate purely because it stopped early.
            taken = max(taken, 1)
            # Steady-state rate from the LAST HALF of the path. The first steps are the
            # servo accelerating from rest, and the travel cap truncates different
            # trials at different points along that ramp -- averaging the whole trial
            # then reads as a lower gain purely because the cap bit earlier (real
            # 12.1 -> 8.1 mm/step at amp 1.0 on identical settings).
            ss = float("nan")
            if axis < 3 and len(path) >= 4:
                P = np.asarray(path)
                half = len(P) // 2
                ss = float((P[-1, axis] - P[half, axis]) / (len(P) - 1 - half))
            rows.append(dict(amp=float(amp), travel=d, per_step=d / taken, steps=taken,
                             commanded_per_step=float(amp * scale(axis)),
                             frac=float(d / taken / (amp * scale(axis))),
                             # Direction matters: a command that moves the EE somewhere
                             # other than the commanded axis is a mapping fault, and the
                             # travel MAGNITUDE alone cannot tell you that happened.
                             steady_per_step=ss,
                             # Step-to-step reversals along the commanded axis. Smooth
                             # tracking never reverses; jitter does, and the endpoint
                             # displacement cannot show it.
                             reversals=int(np.sum(np.diff(np.sign(
                                 np.diff(np.asarray(path)[:, axis]))) != 0))
                             if axis < 3 and len(path) >= 3 else 0,
                             disp=np.round(np.asarray(p1) - np.asarray(p0), 6).tolist(),
                             # Per ACTUAL steps, like per_step above: a trial cut
                             # short by the travel cap otherwise under-reports how
                             # much of it ran against the clamp.
                             sat_frac=sat / taken, tau_max=taumax,
                             # SIM's ctrlrange clip, which is a different question from
                             # sat_frac: whether the reference's own saturation is being
                             # reproduced. Law ticks, so ~25 per sweep step at 500 Hz.
                             sim_clip_ticks=(backend.sim_clip_ticks() - clip0)
                             if hasattr(backend, "sim_clip_ticks") else None,
                             path=path))
            unit = "mm" if axis < 3 else "mrad"
            k = 1e3
            r = rows[-1]
            extra = ""
            if np.isfinite(ss):
                extra = f"  steady {ss*k:7.3f}  rev {r['reversals']:2d}"
            print(f"  {AXES[axis]:>3} amp {amp:4.2f}: {r['per_step']*k:8.3f} {unit}/step "
                  f"of {amp*scale(axis)*k:8.3f} commanded  ({r['frac']:5.1%}){extra}")
    return out


def deadband(rows) -> tuple:
    """Fit travel_per_step = k*(commanded - d). Returns (d, k, r2).

    d is the amplitude at which motion starts -- the deadband, in the axis's units.
    A gain error moves k and leaves d at zero; friction moves d. Reporting both is
    the whole point, since a single fudge cannot represent d at all.
    """
    # Fit only the linear region. Large rotation commands SATURATE (0.5 rad in one
    # 50 ms step is far past what the axis can deliver), and including those points
    # tilts the line until the intercept comes out negative -- a deadband of -57 mrad,
    # which is not a physical reading. Saturation at the top and a deadband at the
    # bottom are different effects; only the second is what a fudge is fighting.
    rows = sorted(rows, key=lambda r: r["commanded_per_step"])
    # Points below the deadband carry no slope information -- they are all ~0 (or
    # negative, when residual drift dominates) and including them drags the intercept
    # toward the origin, hiding the very quantity being measured.
    rows = [r for r in rows if r["per_step"] > 0]
    if len(rows) < 2:
        return float("nan"), float("nan"), float("nan")
    fr = np.array([r["frac"] for r in rows])
    keep = fr >= 0.85 * np.nanmax(fr) if np.isfinite(fr).any() else np.ones(len(rows), bool)
    # A large deadband leaves only two or three points above it. Two still determine a
    # line, and d is what we are after; r2 is simply undefined there and says so.
    if keep.sum() < 2:
        keep = np.zeros(len(rows), bool)
        keep[-min(3, len(rows)):] = True
    x = np.array([r["commanded_per_step"] for r in rows])[keep]
    y = np.array([r["per_step"] for r in rows])[keep]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([x, np.ones(x.size)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    ss = float(np.sum((y - y.mean()) ** 2))
    k = float(c[0])
    return (float(-c[1] / k) if abs(k) > 1e-12 else float("nan"), k,
            float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan"))


def sag(backend, q0, cfg, offset_steps: int = 0) -> dict:
    """Command ZERO delta and measure the drift. The residual-force measurement.

    EE_DELTA re-anchors its goal on the measured pose each step, so a zero command
    leaves zero position error and only damping opposes gravity: any uncompensated
    force shows up as a steady drift, v = F/(Lambda*kd). Nulling that drift calibrates
    the declared payload WITHOUT needing Lambda, and unlike static tau_ext it is not
    buried under a 0.78 Nm friction-hysteresis floor -- the arm is moving throughout,
    so there is no stiction band to settle into.

    MEASURE IT OFF THE ANCHOR TOO (`offset_steps`). reset() parks the arm on q0, which
    is also the OSC's nullspace reference, and the nullspace regulator is identically
    zero there -- so a sag measured at the anchor is blind to every residual that
    depends on being off it. Teleop drift is reported after a move, which is exactly
    the configuration this skips. `offset_steps` drives the arm away along cfg.axes[0]
    first and reports the joint deviation it achieved, so a rate that grows with
    `dev_rad` separates a configuration-dependent residual from a payload error, which
    is present at the anchor as well.
    """
    backend.reset(q0)
    if backend.name == "real":
        time.sleep(cfg.settle_s)
    for _ in range(offset_steps):
        backend.step(cfg.axes[0], cfg.sign)
        if backend.name == "real":
            time.sleep(1.0 / cfg.fps)
    if offset_steps:
        # COMMAND zero deltas to settle; do not sleep. Sleeping sends no goal at all,
        # so the arm keeps driving toward the last full-amplitude goal and then trips
        # the stale-goal timeout, which settle_s happens to equal exactly. p0 was then
        # captured mid-flight and the offset run reported the arm's coasting speed as
        # drift -- 144 mm/s, most of it the offset itself.
        for _ in range(int(np.ceil(cfg.settle_s * cfg.fps))):
            backend.step(0, 0.0)
            if backend.name == "real":
                time.sleep(1.0 / cfg.fps)
    dev = float("nan")
    if hasattr(backend, "q"):
        dev = float(np.max(np.abs(backend.q() - q0)))
    p0, _ = backend.pose()
    path, period = [], 1.0 / cfg.fps
    for _ in range(cfg.steps):
        t0 = time.perf_counter()
        backend.step(0, 0.0)
        path.append(np.round(backend.pose()[0], 6).tolist())
        if backend.name == "real":
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    P = np.asarray(path)
    drift = P[-1] - np.asarray(p0)
    secs = cfg.steps / cfg.fps
    return dict(drift_m=drift.tolist(), drift_rate_m_s=(drift / secs).tolist(),
                sag_rate_mm_s=float(-drift[2] / secs * 1e3), seconds=secs,
                offset_steps=int(offset_steps), dev_rad=dev,
                speed_mm_s=float(np.linalg.norm(drift) / secs * 1e3), path=path)


def compare(sim_dir, real_dir):
    s = json.loads((Path(sim_dir).expanduser() / "delta_sweep.json").read_text())
    r = json.loads((Path(real_dir).expanduser() / "delta_sweep.json").read_text())
    print("\ntravel ratio real/sim, per axis and amplitude\n")
    amps = [row["amp"] for row in next(iter(s["sweep"].values()))]
    head = f"{'axis':<6}" + "".join(f"{a:>8.2f}" for a in amps) + f"{'shape':>28}"
    print(head)
    print("-" * len(head))
    for ax in s["sweep"]:
        if ax not in r["sweep"]:
            continue
        sm = {row["amp"]: row["per_step"] for row in s["sweep"][ax]}
        rm = {row["amp"]: row["per_step"] for row in r["sweep"][ax]}
        ratios = [rm[a] / sm[a] if a in rm and a in sm and sm[a] > 1e-12 else np.nan
                  for a in amps]
        lo = np.nanmean(ratios[:2])
        hi = np.nanmean(ratios[-2:])
        # Judge on the FITTED k and d, not on the ratio shape. The shape test alone
        # calls a collapsed gain a deadband: if the axis barely responds at any
        # amplitude the ratios are all near zero, lo/hi is then a ratio of two noise
        # figures, and it trips the < 0.75 test for free. Measured on this rig -- rz
        # read "DEADBAND" at k_real/k_sim = 0.04 with a fitted d of 2 mrad against a
        # 500 mrad command, i.e. no meaningful deadband at all.
        d_r, k_r, _ = deadband(r["sweep"][ax])
        _, k_s, _ = deadband(s["sweep"][ax])
        span = max(amps) * scale(AXES.index(ax))
        k_ratio = k_r / k_s if abs(k_s) > 1e-12 else float("nan")
        if not np.isfinite(lo) or not np.isfinite(hi):
            verdict = "insufficient data"
        elif np.isfinite(k_ratio) and k_ratio < 0.25:
            verdict = f"GAIN COLLAPSE, k x{k_ratio:.2f}"
        elif np.isfinite(d_r) and span > 0 and d_r / span > 0.05:
            verdict = f"DEADBAND, d={100*d_r/span:.0f}% of full cmd"
        elif abs(hi - 1.0) > 0.15:
            verdict = f"gain error, x{hi:.2f} flat"
        else:
            verdict = "matched"
        print(f"{ax:<6}" + "".join(f"{v:>8.2f}" if np.isfinite(v) else f"{'-':>8}"
                                   for v in ratios) + f"{verdict:>28}")
    print("\ndeadband from travel(commanded) = k*(commanded - d):\n")
    print(f"{'axis':<6}{'d sim':>12}{'d real':>12}{'k sim':>9}{'k real':>9}{'r2 real':>9}")
    print("-" * 57)
    for ax in s["sweep"]:
        if ax not in r["sweep"]:
            continue
        ds, ks, _ = deadband(s["sweep"][ax])
        dr, kr, r2 = deadband(r["sweep"][ax])
        u = 1e3
        print(f"{ax:<6}{ds*u:>10.3f}{'mm' if ax in 'xyz' else 'mr':>2}"
              f"{dr*u:>10.3f}{'mm' if ax in 'xyz' else 'mr':>2}"
              f"{ks:>9.3f}{kr:>9.3f}{r2:>9.3f}")
    print("\nd is what a fudge cannot represent: it is an offset, not a gain.")
    print("k is the gain, and k_real/k_sim is the only part a scalar can fix.")

    # Sim's rotation authority rolls off because its +/-12 Nm wrist ctrlrange
    # saturates; the FR3's does not. We apply that same clip inside run_controller, so
    # if these are zero where sim's travel is falling, the roll-off is NOT being
    # reproduced and no gain will fix the shape. Distinct from sat_frac above, which
    # watches the real hardware clamp and reads 0.000 whatever happens.
    any_clip = any(row.get("sim_clip_ticks") is not None
                   for ax in r["sweep"] for row in r["sweep"][ax])
    if any_clip:
        print("\nsim ctrlrange clip engagement (law ticks per trial, real backend):\n")
        print(f"{'axis':<6}" + "".join(f"{a:>8.2f}" for a in amps))
        print("-" * (6 + 8 * len(amps)))
        for ax in r["sweep"]:
            m = {row["amp"]: row.get("sim_clip_ticks") for row in r["sweep"][ax]}
            print(f"{ax:<6}" + "".join(
                f"{m[a]:>8d}" if m.get(a) is not None else f"{'-':>8}" for a in amps))
        print("\n0 at the top amplitudes = sim saturates there and we do not.")
    else:
        print("\nsim_clip_ticks not recorded in this run (pre-dates the counter).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("sim", "real"))
    ap.add_argument("--compare", nargs=2, metavar=("SIM_DIR", "REAL_DIR"))
    ap.add_argument("--axes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                    help="0-2 translation xyz, 3-5 rotation")
    ap.add_argument("--amps", type=float, nargs="+", default=list(Config.amps))
    ap.add_argument("--steps", type=int, default=Config.steps)
    ap.add_argument("--sign", type=float, default=1.0, choices=(1.0, -1.0))
    ap.add_argument("--pose", default=None)
    ap.add_argument("--profile", default="single_arm_franka")
    ap.add_argument("--arm", default="r", help="key prefix in --profile, not a side")
    ap.add_argument("--sag", action="store_true",
                    help="command zero delta and report the drift; use to null the "
                         "declared payload without needing Lambda")
    ap.add_argument("--sag-offset-steps", type=int, default=0,
                    help="drive this many full-amplitude steps along the first --axes "
                         "entry BEFORE measuring sag. The anchor is also the nullspace "
                         "reference, so sag at 0 cannot see any residual that only "
                         "exists off it -- which is where teleop drift is reported")
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.backend:
        ap.error("--backend or --compare is required")

    cfg = Config(amps=tuple(args.amps), steps=args.steps, axes=args.axes, sign=args.sign)
    q0 = np.asarray(fc.home_q(args.pose, key=args.arm), dtype=np.float64)

    if args.backend == "sim":
        backend = SimBackend()
    else:
        if not args.yes:
            resp = input(f"This MOVES the arm: {len(args.axes)} axes x {len(args.amps)} "
                         f"amplitudes, homing between each. Workspace clear? [y/N] ")
            if resp.strip().lower() not in ("y", "yes"):
                print("aborted")
                return
        spec = fc.profile(args.profile)
        if args.arm not in spec.arms:
            raise SystemExit(f"profile '{args.profile}' has no key '{args.arm}'")
        print(f"profile {args.profile}: key '{args.arm}' -> "
              f"{fc.arm(spec.arms[args.arm]).name} arm")
        backend = RealBackend(args.arm, args.profile)

    print(f"anchor q = {np.round(q0, 4)}\n")
    f0 = backend.faults() if hasattr(backend, "faults") else 0
    data = {}
    try:
        if args.sag:
            s = sag(backend, q0, cfg, offset_steps=args.sag_offset_steps)
            print(f"zero-delta drift over {s['seconds']:.1f} s "
                  f"({s['offset_steps']} offset steps, {s['dev_rad']:.3f} rad off anchor):")
            print(f"  displacement  {np.round(np.asarray(s['drift_m'])*1e3, 2)} mm")
            print(f"  SPEED         {s['speed_mm_s']:.1f} mm/s   (any axis; this is the "
                  f"no-op drift an operator sees)")
            print(f"  SAG RATE      {s['sag_rate_mm_s']:+.1f} mm/s   "
                  f"(+ = falling, declared mass too LOW; - = too high; 0 = correct)")
            data["sag"] = s
        else:
            sweep(backend, q0, cfg, out=data)
    except KeyboardInterrupt:
        # Everything completed so far is already in `data`; fall through and write it.
        print("\n  interrupted -- writing the trials completed so far")
    finally:
        if hasattr(backend, "close"):
            backend.close()
    faulted = (backend.faults() - f0) if hasattr(backend, "faults") else 0

    if not args.sag:
        print()
        print(f"{'axis':<6}{'deadband':>14}{'gain k':>10}{'r2':>8}")
        print("-" * 38)
        for ax, rows in data.items():
            d, k, r2 = deadband(rows)
            print(f"{ax:<6}{d*1e3:>11.3f} {'mm' if ax in 'xyz' else 'mrad':<4}"
                  f"{k:>10.3f}{r2:>8.3f}")
    if faulted:
        print(f"\n  WARNING: {faulted} recoverable fault(s); those rows are suspect.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out).expanduser() if args.out else OUT_ROOT / f"{stamp}_deltasweep_{backend.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "delta_sweep.json").write_text(json.dumps(
        dict(backend=backend.name, anchor_q=q0.tolist(), config=vars(cfg),
             pos_max_m=POS_MAX, rot_max_rad=ROT_MAX, faults=faulted, sweep=data),
        indent=2))
    print(f"\nwrote {out}/delta_sweep.json")


if __name__ == "__main__":
    main()
