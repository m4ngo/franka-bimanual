#!/usr/bin/env python3
"""Per-joint plant identification, run identically in sim and on the arm.

    python sysid/joint_id.py --backend real --yes            # anchors at the arm's current q
    PYTHONPATH=franka_config multi-fast/.venv/bin/python \
        sysid/joint_id.py --backend sim --q <the 7 numbers the real run printed>
    python sysid/joint_id.py --compare ~/sysid/outputs/<sim>/ ~/sysid/outputs/<real>/

Anchor the REAL run first and match sim to it with --q. Both plants must sit at the
same q for the comparison to mean anything, and only one of the two is free to move
without a human deciding it is safe.

--arm is a KEY PREFIX in a config/rig.yaml profile, not a side: single_arm_franka
maps `r` to the physical LEFT arm (luigi). The script prints which it resolved to
before it connects.

One joint at a time under OPEN-LOOP torque, every other joint impedance-held. tau
is the independent variable, so nothing here depends on a servo gain and the same
protocol runs unchanged against mujoco and against MODE_TORQUE.

Per (joint, pose, direction) it fits the LAUNCH of a ladder of constant-torque
bursts, each started from rest:

    tau = M*qddot0 + tau_c

At rest qdot = 0, so no viscous term can trade off against M. Fitting a viscous
coefficient as well was tried and abandoned: over a full burst it absorbed ~90% of
the torque into a term 30x mujoco's known damping and returned M ~ 0.

Getting M is the point. A breakaway staircase alone returns tau_c with M unknown,
and M is what decides whether a given tau_c matters -- tau_c/M is the joint's
deadband in rad/s^2, and it is that ratio, not tau_c, that differs between plants.

tau_c here is an upper bound: the intercept also absorbs the held neighbours'
reaction, and against mujoco's known 0.1 Nm it reads 0.09-0.23. Compare tau_c/M
across backends rather than reading tau_c as absolute friction.

The sim backend is also the estimator's self-test: mujoco's M (link inertia plus
armature) and frictionloss are known exactly, so a sim run that does not recover
them means the fit is wrong and the real numbers are not worth reading.

Both backends drive one joint at a time within a travel window and return it to
where it started. Clear the workspace before --backend real.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import franka_config as fc

NUM_JOINTS = 7
OUT_ROOT = Path("~/sysid/outputs").expanduser()

_JOINT_KP = np.asarray(fc.control("torque.joint_impedance.kp"), dtype=np.float64)
_JOINT_KD = np.asarray(fc.control("torque.joint_impedance.kd"), dtype=np.float64)
_KD_POLE_MAX = float(fc.control("torque.joint_impedance.kd_pole_max_rad_s"))
_JOINT_VMAX = np.asarray(fc.control("franka.max_joint_velocity_rad_s"), dtype=np.float64)
_TAU_LIMIT = np.asarray(fc.control("torque.limits.joint_torque_nm"), dtype=np.float64)


@dataclass
class Config:
    tau_step: float = 0.10          # staircase resolution, and the breakaway's error bar
    tau_max: float = 12.0
    # Net torque above breakaway. Only the first entry is used as-is: it is the probe
    # burst, and the rest of the ladder is resized from what it measures (see
    # identify_joint). The literal ladder is the fallback when the probe fails.
    tau_over: tuple = (0.25, 0.6, 1.2, 2.2, 3.5)
    n_bursts: int = 5
    launch_target_s: float = 0.30   # how long the launch should last, any joint
    launch_peak_frac: float = 0.80  # of the speed cap, at the top of the ladder
    ladder_spread: float = 8.0      # top:bottom net torque; sets the fit's conditioning
    over_min_nm: float = 0.02       # smallest net torque worth commanding
    ladder_floor_frac: float = 0.25  # lowest rung, as a fraction of breakaway
    ladder_hi_min: float = 3.0      # top rung, as a multiple of the floor
    stop_dq_frac: float = 0.85      # end the burst here; never ride the speed cap
    return_qdot_rad_s: float = 0.4  # bounded rate back to the anchor; see settle()
    brake_s: float = 0.25
    ramp_dt: float = 0.04
    abort_dq_frac: float = 0.60     # of the datasheet limit; abort the run above this
    window_rad: float = 0.05        # breakaway detection: short, so the joint barely moves
    burst_window_rad: float = 0.35  # response bursts need room to reach steady velocity
    dq_max_frac: float = 0.25       # of the datasheet limit; the wrist faults on its own
    dwell_s: float = 0.40           # must stay under torque.loop.stale_goal_timeout_s
    burst_s: float = 2.00
    # Poll/log rate. Real: set at or above 1000/torque.loop.publish_decimation, since
    # repeats are dropped anyway. Sim: the log decimation, so 500 Hz matches a NUC
    # publishing at decimation 2 and the two backends sample the same way.
    fps: float = 600.0
    dq_floor: float = 0.02          # below this the joint is presliding, not sliding
    # Staircase onset, and the minimum peak a burst must reach to be fitted. NOT
    # dq_floor: a joint creeps at a few hundredths of a rad/s while still in
    # presliding, so a floor-level trigger stops the staircase below true sliding
    # friction and every low rung of the ladder then creeps instead of launching.
    # Measured cost of getting this wrong: bands of 0.007-0.03 rad/s above the noise
    # floor, acceleration estimates that are noise, and ladder fits at r2 0.28-0.83.
    onset_dq: float = 0.15
    accel_window_s: float = 0.35    # hard cap on the launch window; see initial_accel
    launch_dq_frac: float = 0.35    # fit only below this fraction of the burst's peak qdot
    settle_s: float = 0.8
    settle_dq: float = 5.0e-3       # a burst must start from rest; see initial_accel
    settle_max_s: float = 4.0
    poses: list = field(default_factory=list)


# ------------------------------------------------------------------- fitting

def initial_accel(t, dq, direction, cfg) -> tuple:
    """(qddot at breakout, qdot slope diagnostic) from a burst that started at rest.

    Only the launch matters. Held at rest, qdot=0 makes the viscous term vanish
    identically, so tau = M*qddot + tau_c with nothing else in it -- no damping
    estimate to be traded off against M, and no dependence on how the neighbours
    yield later in the swing. The full-burst regression this replaces put ~90% of
    the torque into a viscous term 30x mujoco's known damping and returned M ~ 0.
    """
    s = np.sign(direction)
    qd = s * np.asarray(dq)
    if qd.size < 6:
        return float("nan"), float("nan")
    moving = np.flatnonzero(qd > cfg.dq_floor)
    if moving.size < 4:
        return float("nan"), float("nan")
    i0 = moving[0]
    # A burst that never reached sliding speed carries no usable acceleration: its
    # launch band collapses onto the noise floor and the slope is noise. Drop it
    # rather than let it set the ladder's slope.
    if float(np.max(qd[i0:])) < cfg.onset_dq:
        return float("nan"), float("nan")
    # Cut on VELOCITY, not on time. The launch is only free of velocity-dependent
    # terms while qdot is small, and how long that lasts scales with the joint: a
    # fixed window that suits joint 2 is ~5x too long for the wrist and reads its
    # deceleration as inertia (joint 7 came out 1.39x mujoco's M_jj that way, 1.00x
    # once cut here). Peak-relative so it needs no per-joint constant.
    # First CONTIGUOUS run below the threshold. Taking the last sample under it instead
    # runs past the window trip, where the joint has decelerated back below it.
    peak = float(np.max(qd[i0:])) if qd.size > i0 else 0.0
    above = np.flatnonzero(qd[i0:] >= cfg.launch_dq_frac * peak) if peak > 0 else np.array([])
    span = int(above[0]) if above.size else qd.size - i0
    span = max(5, span)
    # Bound the window in TIME, not in samples: the achieved sample rate depends on
    # the NUC's publish decimation and on RPyC round-trip time, so a sample count
    # derived from a nominal fps silently means different things per run.
    tt_all = np.asarray(t)
    within = np.flatnonzero(tt_all[i0:] - tt_all[i0] <= cfg.accel_window_s)
    span = max(5, min(span, int(within[-1]) + 1 if within.size else span))
    i1 = min(qd.size, i0 + span)
    tt, vv = tt_all[i0:i1], qd[i0:i1]
    if tt.size < 4 or tt[-1] - tt[0] < 1e-3:
        return float("nan"), float("nan")
    A = np.column_stack([tt - tt[0], np.ones(tt.size)])
    coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
    resid = vv - A @ coef
    ss = float(np.sum((vv - vv.mean()) ** 2))
    return float(coef[0]), float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")


def fit_plant(bursts, direction, cfg) -> dict:
    """tau = M*qddot0 + tau_c across the burst ladder. Slope is M, intercept tau_c."""
    x, y, fits, n_s, span_s = [], [], [], 0, 0.0
    for tau_nm, t, q, dq in bursts:
        tt = np.asarray(t)
        if tt.size > 2 and tt[-1] > tt[0]:
            n_s += tt.size - 1
            span_s += float(tt[-1] - tt[0])
        a, r2 = initial_accel(t, dq, direction, cfg)
        if not np.isfinite(a) or a <= 0.0:
            continue
        x.append(a)
        y.append(abs(tau_nm))
        fits.append(r2)
    # Samples per second over the whole burst, NOT 1/median(dt): dedup makes the gaps
    # bimodal (one poll period or two), and the median then picks the commoner mode and
    # reports above the true publish rate -- 580 Hz against a 500 Hz publish.
    rate = float(n_s / span_s) if span_s > 0 else float("nan")
    dropped = len(bursts) - len(x)
    if len(x) < 3:
        return dict(M=float("nan"), tau_c=float("nan"), r2=float("nan"),
                    accel_r2=float("nan"), n=len(x), bursts=len(bursts),
                    dropped=dropped, rate_hz=rate)
    x, y = np.asarray(x), np.asarray(y)
    A = np.column_stack([x, np.ones(x.size)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(M=float(coef[0]), tau_c=float(coef[1]),
                r2=float(1.0 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan"),
                accel_r2=float(np.mean(fits)), n=int(x.size), bursts=len(bursts),
                dropped=dropped, rate_hz=rate)


# ------------------------------------------------------------------ protocol

def _burst_done(q_j, dq_j, origin, window_rad, dq_max, cfg) -> bool:
    """Stop once the launch is captured, before the joint reaches its limits.

    Running to the travel window meant every burst slammed into a latched hold at
    full speed and rang there for the rest of the burst -- 30 hard stops per run,
    visible oscillation on joint 7, and a recoverable fault. Nothing past the launch
    is used by the fit, so there is no reason to collect it.
    """
    return (abs(dq_j) > cfg.stop_dq_frac * dq_max
            or abs(q_j - origin) > 0.8 * window_rad)


def settle(backend, joint, q_anchor, cfg) -> None:
    """Brake in place, walk the goal back to the anchor, then wait for stillness.

    NEVER step the goal to the anchor. The hold runs at joint_impedance.kp, so on
    joint 7 (kp 100 Nm/rad, clamp 20 Nm) anything past 0.20 rad saturates: 400 rad/s^2
    on its measured 0.05 kg m^2, arriving at 15 rad/s against a 2.61 rad/s datasheet
    limit. The overshoot reverses the error and saturates the other way -- a bang-bang
    limit cycle, which is the fast oscillation seen on joint 7, not a return. Joints
    5-7 saturate at 0.15-0.20 rad; 1-4 need 0.29 and are 30x heavier.

    Braking first matters too: the burst ends with the joint moving, and a position
    goal at the anchor would add a step error on top of that speed. Holding at the
    measured q leaves only -kd*dq, which is pure damping.
    """
    q = backend.state()[0].copy()
    backend.hold(q, cfg.brake_s)
    lead = backend.state()[0].copy()
    step = cfg.return_qdot_rad_s * cfg.ramp_dt
    for _ in range(int(np.ceil(2.0 * np.pi / max(step, 1e-6))) + 10):
        if np.max(np.abs(q_anchor - lead)) <= 1e-3:
            break
        lead = lead + np.clip(q_anchor - lead, -step, step)
        backend.hold(lead, cfg.ramp_dt)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < cfg.settle_max_s:
        backend.hold(q_anchor, 0.1)
        if abs(backend.state()[1][joint]) < cfg.settle_dq:
            return


def breakaway(backend, joint, q_anchor, direction, cfg) -> float:
    """Staircase the commanded torque until the joint slides. Upper bound by tau_step."""
    n = int(cfg.tau_max / cfg.tau_step)
    for i in range(1, n + 1):
        tau = direction * i * cfg.tau_step
        settle(backend, joint, q_anchor, cfg)
        t, q, dq = backend.burst(joint, tau, q_anchor, cfg.window_rad,
                                 cfg.dq_max_frac * _JOINT_VMAX[joint], cfg.dwell_s)
        if dq.size and np.max(np.sign(direction) * dq) >= cfg.onset_dq:
            return abs(tau)
    return float("nan")


def identify_joint(backend, joint, q_anchor, cfg) -> dict:
    out = {}
    for direction in (+1.0, -1.0):
        tb = breakaway(backend, joint, q_anchor, direction, cfg)
        if not np.isfinite(tb):
            out["pos" if direction > 0 else "neg"] = dict(
                breakaway=float("nan"), M=float("nan"), tau_c=float("nan"),
                r2=float("nan"), accel_r2=float("nan"), n=0, bursts=0)
            continue
        dq_max = cfg.dq_max_frac * _JOINT_VMAX[joint]
        tau_ceil = min(cfg.tau_max, _TAU_LIMIT[joint])

        def run(over):
            tau = direction * (tb + over)
            if abs(tau) > tau_ceil:
                return None
            settle(backend, joint, q_anchor, cfg)
            t, q, dq = backend.burst(joint, tau, q_anchor, cfg.burst_window_rad,
                                     dq_max, cfg.burst_s)
            return (tau, t, q, dq)

        probe = run(cfg.tau_over[0])
        if probe is None:
            out["pos" if direction > 0 else "neg"] = fit_plant([], direction, cfg)
            out["pos" if direction > 0 else "neg"]["breakaway"] = float(tb)
            continue
        # The probe only sizes the ladder; it is not kept as a data point, because a
        # ladder rebuilt around a light joint can land entirely below it.
        # A fixed ladder is calibrated to one inertia: real's wrist is 10-30x lighter
        # than sim's (sim's armature does not exist on the arm), so a ladder that suits
        # joint 2 puts joint 7 on the speed cap within ~2 samples, leaving no launch.
        a1, _ = initial_accel(probe[1], probe[3], direction, cfg)
        ladder = list(cfg.tau_over[:cfg.n_bursts])
        if np.isfinite(a1) and a1 > 1e-6:
            m_hat = cfg.tau_over[0] / a1
            # Whichever binds first, the speed cap or the travel window.
            a_cap = cfg.launch_peak_frac * dq_max / cfg.launch_target_s
            a_win = 2.0 * cfg.burst_window_rad / cfg.launch_target_s ** 2
            # Floor is over_min_nm, NOT a multiple of tau_step: tau_step is the
            # breakaway staircase's resolution, but ladder torques are commanded
            # directly and are not quantised by it. Tying the two forced joint 7's
            # ladder 4.6x above what the sizing asked for, leaving 8 samples to the
            # speed cap and too short a launch to fit.
            # The floor has to clear breakaway by a real margin, not by an absolute
            # constant: on the wrist tb is 0.4-0.8 Nm while the acceleration target
            # only asks for ~0.09 Nm, so the bottom rungs landed BELOW the true
            # friction and could not move the joint at all -- 2 of 5 bursts dead and
            # the rest spanning a 10% modulation on tb, which fits a negative M.
            # tau_step is in here because tb is a staircase result and is only known to
            # that resolution: a floor below it can sit under the true friction even
            # when the fraction of tb looks like ample margin.
            floor = max(cfg.over_min_nm, cfg.ladder_floor_frac * tb, cfg.tau_step)
            hi = float(np.clip(m_hat * min(a_cap, a_win),
                               cfg.ladder_hi_min * floor, tau_ceil - tb))
            # Light joint + large breakaway therefore gets a brisker ladder, while a
            # proximal joint (tb small next to the acceleration target) is untouched.
            lo = float(np.clip(max(hi / cfg.ladder_spread, floor),
                               cfg.over_min_nm, hi / 2.0))
            ladder = list(np.geomspace(lo, hi, cfg.n_bursts))
        bursts = []
        for over in ladder:
            b = run(over)
            if b is None:
                break
            bursts.append(b)
        r = fit_plant(bursts, direction, cfg)
        r["breakaway"] = float(tb)
        # Raw traces travel with the fit. When a run misbehaves on hardware the numbers
        # alone cannot say whether it was the plant, the ladder, or the return, and the
        # arm is not somewhere you can re-observe a transient cheaply.
        r["trace"] = [dict(tau=float(tau), t=np.round(t, 5).tolist(),
                           q=np.round(q, 6).tolist(), dq=np.round(dq, 5).tolist())
                      for tau, t, q, dq in bursts]
        out["pos" if direction > 0 else "neg"] = r
    return out


# ------------------------------------------------------------------ backends

class SimBackend:
    """mujoco under MODE_TORQUE semantics: hold every joint, replace one with tau_ff."""

    name = "sim"

    def __init__(self, env_name="Lift"):
        import robosuite
        from robosuite.controllers import load_controller_config
        cc = load_controller_config(default_controller="OSC_POSE")
        self.env = robosuite.make(env_name, robots="Panda", has_renderer=False,
                                  has_offscreen_renderer=False, use_camera_obs=False,
                                  controller_configs=cc, control_freq=20)
        self.env.reset()
        self.dt = self.env.sim.model.opt.timestep
        self._sync()

    def _sync(self):
        import mujoco
        self.sim = self.env.sim
        self.r = self.env.robots[0]
        self.qi = self.r._ref_joint_pos_indexes
        self.vi = self.r._ref_joint_vel_indexes
        self.ai = self.r._ref_joint_actuator_indexes
        self._m = self.sim.model._model
        self._scratch = mujoco.MjData(self._m)

    def _gravity(self):
        """g(q) alone, which is what libfranka adds under the real MODE_TORQUE.

        qfrc_bias is g + C(q,qdot)qdot; the real loop's held joints get Coriolis from
        the control law but the TESTED joint gets only libfranka's gravity term, so
        the sim has to separate them or it measures a different plant.
        """
        import mujoco
        self._scratch.qpos[:] = self.sim.data.qpos
        self._scratch.qvel[:] = 0.0
        mujoco.mj_forward(self._m, self._scratch)
        return self._scratch.qfrc_bias[self.vi].copy()

    def reference(self) -> dict:
        """mujoco's own answer, for checking the fit against ground truth."""
        M = np.zeros((self.sim.model.nv, self.sim.model.nv))
        import mujoco
        mujoco.mj_fullM(self.sim.model._model, M, self.sim.data.qM)
        return dict(M_diag=[float(M[i, i]) for i in self.vi],
                    armature=[float(v) for v in self.sim.model.dof_armature[self.vi]],
                    frictionloss=[float(v) for v in self.sim.model.dof_frictionloss[self.vi]],
                    damping=[float(v) for v in self.sim.model.dof_damping[self.vi]])

    def state(self):
        return (self.sim.data.qpos[self.qi].copy(), self.sim.data.qvel[self.vi].copy())

    def set_q(self, q):
        self.sim.data.qpos[self.qi] = q
        self.sim.data.qvel[self.vi] = 0.0
        self.sim.forward()

    def _hold_tau(self, q, dq, goal_q):
        import mujoco
        M = np.zeros((self.sim.model.nv, self.sim.model.nv))
        mujoco.mj_fullM(self.sim.model._model, M, self.sim.data.qM)
        Md = np.array([M[i, i] for i in self.vi])
        kd = np.minimum(_JOINT_KD, _KD_POLE_MAX * Md)
        bias = self.sim.data.qfrc_bias[self.vi]
        return _JOINT_KP * (goal_q - q) - kd * dq + bias

    def hold(self, goal_q, seconds):
        for _ in range(int(seconds / self.dt)):
            q = self.sim.data.qpos[self.qi].copy()
            dq = self.sim.data.qvel[self.vi].copy()
            self.sim.data.ctrl[self.ai] = self._hold_tau(q, dq, goal_q)
            self.sim.step()

    def burst(self, joint, tau_nm, hold_q, window_rad, dq_max, max_s):
        decim = max(1, int(round(1.0 / (Config.fps * self.dt))))
        origin = hold_q[joint]
        t_log, q_log, dq_log = [], [], []
        tripped = False
        for k in range(int(max_s / self.dt)):
            q = self.sim.data.qpos[self.qi].copy()
            dq = self.sim.data.qvel[self.vi].copy()
            tau = self._hold_tau(q, dq, hold_q)
            if not tripped:
                if abs(q[joint] - origin) > window_rad or abs(dq[joint]) > dq_max:
                    tripped = True
                else:
                    tau[joint] = tau_nm + self._gravity()[joint]
            self.sim.data.ctrl[self.ai] = np.clip(tau, -_TAU_LIMIT, _TAU_LIMIT)
            self.sim.step()
            if k % decim == 0:
                t_log.append(k * self.dt)
                q_log.append(q[joint])
                dq_log.append(dq[joint])
            if _burst_done(q[joint], dq[joint], origin, window_rad, dq_max, Config):
                break
        return (np.asarray(t_log), np.asarray(q_log), np.asarray(dq_log))


class RealBackend:
    """The FR3 through MODE_TORQUE. The travel/speed window is enforced on the NUC."""

    name = "real"

    def __init__(self, arm, server_ip, robot_ip, port, poll_fps=Config.fps):
        from lerobot_robot_bimanual_franka.franka_process import MultiRobotWrapper
        self.arm = arm
        # Poll above the NUC's publish rate and drop the repeats; see burst().
        self.poll_fps = float(poll_fps)
        self.mgr = MultiRobotWrapper()
        self.mgr.add_robot(arm, server_ip, robot_ip, port)
        # The raw plant is the measurement; an assist left by an earlier script is not.
        self.mgr.set_tuning_all(friction_kc=0.0)
        if not self.mgr.supports_torque_mode(arm):
            self.mgr.shutdown()
            raise SystemExit(
                "This NUC's server predates MODE_TORQUE. Run\n"
                "  bash scripts/deploy_nuc_server.sh <mario|luigi>\n"
                "Nothing has moved.")

    def reference(self) -> dict:
        return {}

    def recoveries(self) -> int:
        """Recoverable faults since the loop armed. A mid-run fault re-arms holding
        wherever the arm ended up, so any increase invalidates the bursts around it."""
        try:
            return int(self.mgr.recovery_counts()[self.arm])
        except Exception:
            return -1

    def state(self):
        q, dq, *_ = self.mgr.current_kinematic_state(self.arm)
        return np.asarray(q, dtype=np.float64), np.asarray(dq, dtype=np.float64)

    def set_q(self, q):
        self.hold(q, 1.0)

    def hold(self, goal_q, seconds, fps=30.0):
        for _ in range(int(seconds * fps)):
            self.mgr.move_joint_goal_batch({self.arm: (goal_q, 1.0, 1.0)})
            time.sleep(1.0 / fps)

    def burst(self, joint, tau_nm, hold_q, window_rad, dq_max, max_s):
        tau = np.zeros(NUM_JOINTS)
        tau[joint] = tau_nm
        # Pushed once. Re-pushing re-latches the loop's window and lets the joint
        # ratchet outward one window per push.
        self.mgr.move_torque_goal(self.arm, tau, joint, hold_q, window_rad, dq_max)
        t_log, q_log, dq_log = [], [], []
        prev = None
        t0 = time.perf_counter()
        while True:
            t = time.perf_counter() - t0
            if t >= max_s:
                break
            q, dq = self.state()
            # Keep only genuinely new publishes. get_kinematic_state returns whatever
            # the loop last published, so polling above the publish rate hands back
            # bitwise-identical repeats -- and a repeated dq at an advancing t flattens
            # the launch slope, biasing M high. Two consecutive publishes of a moving
            # joint never match to the bit.
            key = (q.tobytes(), dq.tobytes())
            if key != prev:
                prev = key
                t_log.append(t)
                q_log.append(q[joint])
                dq_log.append(dq[joint])
            if abs(dq[joint]) > Config.abort_dq_frac * _JOINT_VMAX[joint]:
                self.mgr.move_joint_goal_batch({self.arm: (q, 1.0, 1.0)})
                raise RuntimeError(
                    f"joint {joint+1} reached {dq[joint]:+.2f} rad/s, past "
                    f"{Config.abort_dq_frac:.0%} of its {_JOINT_VMAX[joint]:.2f} rad/s "
                    f"limit. Braked in place and stopped rather than letting it run on.")
            if _burst_done(q[joint], dq[joint], hold_q[joint], window_rad, dq_max, Config):
                break
            time.sleep(max(0.0, 1.0 / self.poll_fps - (time.perf_counter() - t0 - t)))
        # Brake where it IS, not at the anchor -- settle() walks it back. Commanding the
        # anchor here is the saturating step described in settle().
        q_now = self.state()[0]
        self.mgr.move_joint_goal_batch({self.arm: (q_now, 1.0, 1.0)})
        return (np.asarray(t_log), np.asarray(q_log), np.asarray(dq_log))


# ------------------------------------------------------------------ reporting

def report(results, ref, backend_name):
    print(f"\n{backend_name}: tau = M*qddot + tau_c at launch, per direction\n")
    head = (f"{'joint':<6}{'dir':<5}{'break Nm':>10}{'tau_c Nm':>10}{'M kgm2':>9}"
            f"{'tau_c/M':>9}{'fit r2':>8}{'accel r2':>10}{'n':>4}{'drop':>6}{'Hz':>7}")
    print(head)
    print("-" * len(head))
    for j in sorted(results):
        for d in ("pos", "neg"):
            r = results[j].get(d)
            if r is None:
                continue
            dead = r["tau_c"] / r["M"] if r["M"] and np.isfinite(r["M"]) else float("nan")
            print(f"{j+1:<6}{d:<5}{r['breakaway']:>10.2f}{r['tau_c']:>10.3f}"
                  f"{r['M']:>9.3f}{dead:>9.2f}{r['r2']:>8.3f}{r['accel_r2']:>10.3f}"
                  f"{r['n']:>4}{r.get('dropped', 0):>6}{r.get('rate_hz', float('nan')):>7.0f}")
    if ref.get("M_diag"):
        print("\nmujoco ground truth (the fit above should recover these):")
        print(f"  M_diag       {np.round(ref['M_diag'], 3)}")
        print(f"  armature     {np.round(ref['armature'], 3)}")
        print(f"  frictionloss {np.round(ref['frictionloss'], 3)}")
        print(f"  damping      {np.round(ref['damping'], 3)}")


def compare(sim_dir, real_dir):
    s = json.loads((Path(sim_dir).expanduser() / "joint_id.json").read_text())
    r = json.loads((Path(real_dir).expanduser() / "joint_id.json").read_text())
    print("\nreal / sim, per joint (both directions averaged)\n")
    head = (f"{'joint':<6}{'M sim':>8}{'M real':>8}{'M r/s':>8}"
            f"{'tc sim':>8}{'tc real':>9}{'tc r/s':>8}"
            f"{'dead sim':>10}{'dead real':>11}{'dead r/s':>10}")
    print(head)
    print("-" * len(head))
    for j in sorted(set(s["results"]) & set(r["results"]), key=int):
        def avg(src, key):
            v = [src["results"][j][d][key] for d in ("pos", "neg")
                 if d in src["results"][j] and np.isfinite(src["results"][j][d][key])]
            return float(np.mean(v)) if v else float("nan")
        Ms, Mr = avg(s, "M"), avg(r, "M")
        cs, cr = avg(s, "tau_c"), avg(r, "tau_c")
        ds, dr = cs / Ms, cr / Mr
        print(f"{int(j)+1:<6}{Ms:>8.3f}{Mr:>8.3f}{Mr/Ms:>8.2f}"
              f"{cs:>8.3f}{cr:>9.3f}{cr/cs:>8.2f}"
              f"{ds:>10.2f}{dr:>11.2f}{dr/ds:>10.2f}")
    print("\ndead = tau_c/M, rad/s^2 of commanded acceleration lost to friction.")
    print("It is the ratio that transfers, not tau_c: a joint can carry more friction")
    print("than sim and still track it if it carries proportionally more inertia.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("sim", "real"))
    ap.add_argument("--compare", nargs=2, metavar=("SIM_DIR", "REAL_DIR"))
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(1, 8)),
                    help="1-indexed")
    ap.add_argument("--pose", default=None, help="home_poses/<name>.json; default = current q")
    ap.add_argument("--q", type=float, nargs=7, default=None,
                    help="anchor joints directly, overriding --pose. Use this to point "
                         "the sim backend at whatever q the real run actually used.")
    ap.add_argument("--profile", default="single_arm_franka",
                    help="config/rig.yaml profile that maps the key to a physical arm")
    ap.add_argument("--arm", default="r", help="key prefix within --profile, not a side")
    ap.add_argument("--fps", type=float, default=Config.fps,
                    help="real backend poll rate. Set at or above the NUC's publish "
                         "rate (1000/torque.loop.publish_decimation); repeats are dropped")
    ap.add_argument("--tau-step", type=float, default=Config.tau_step)
    ap.add_argument("--tau-max", type=float, default=Config.tau_max)
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.backend:
        ap.error("--backend or --compare is required")

    cfg = Config(tau_step=args.tau_step, tau_max=args.tau_max)
    joints = [j - 1 for j in args.joints]

    if args.backend == "sim":
        backend = SimBackend()
        if args.q is not None:
            q0 = np.asarray(args.q, dtype=np.float64)
        else:
            q0 = np.asarray(fc.home_q(args.pose, key=args.arm), dtype=np.float64)
        backend.set_q(q0)
        backend.hold(q0, 0.5)
    else:
        if not args.yes:
            resp = input(f"This MOVES the arm: joints {args.joints}, up to "
                         f"{np.degrees(cfg.burst_window_rad):.0f} deg each, returned "
                         f"between bursts. Workspace clear? [y/N] ")
            if resp.strip().lower() not in ("y", "yes"):
                print("aborted")
                return
        # Through the profile, never key->side directly: single_arm_franka exposes the
        # LEFT arm under the `r` key, so the prefix is not the physical arm.
        spec = fc.profile(args.profile)
        if args.arm not in spec.arms:
            raise SystemExit(f"profile '{args.profile}' has no key '{args.arm}' "
                             f"(has {sorted(spec.arms)})")
        arm = fc.arm(spec.arms[args.arm])
        print(f"profile {args.profile}: key '{args.arm}' -> {arm.name} arm "
              f"({arm.nuc_host}, {arm.robot_ip})")
        backend = RealBackend(args.arm, arm.server_ip, arm.robot_ip, arm.rpyc_port,
                              poll_fps=args.fps)
        q0, _ = backend.state()
        if args.q is not None or args.pose is not None:
            target = (np.asarray(args.q, dtype=np.float64) if args.q is not None
                      else np.asarray(fc.home_q(args.pose, key=args.arm), dtype=np.float64))
            gap = float(np.max(np.abs(target - q0)))
            # Refuse rather than drive: hold() is a step onto a kp=300 goal, so closing
            # a large gap with it is a jerk. home_pose.py ramps; this script does not.
            if gap > 0.05:
                backend.mgr.shutdown()
                raise SystemExit(
                    f"arm is {np.degrees(gap):.1f} deg from '{args.pose}' -- home it first:\n"
                    f"  python scripts/home_pose.py apply {args.pose} --single-arm\n"
                    "Nothing has moved.")
            q0 = target
        backend.hold(q0, 1.0)

    print(f"anchor q = {np.round(q0, 4)}")
    ref = backend.reference()
    rec0 = backend.recoveries() if args.backend == "real" else 0
    results, faulted = {}, 0
    try:
        for j in joints:
            results[j] = identify_joint(backend, j, q0, cfg)
            backend.hold(q0, cfg.settle_s)
    finally:
        if args.backend == "real":
            faulted = backend.recoveries() - rec0
            # Brake where the arm is. Holding at q0 on the way out is the same
            # saturating step settle() exists to avoid, and it would fire on exactly
            # the paths that reach here abnormally.
            backend.hold(backend.state()[0], 0.4)
            backend.mgr.shutdown()

    report(results, ref, backend.name)
    if faulted > 0:
        print(f"\n  WARNING: {faulted} recoverable fault(s) during the run. The loop "
              f"re-arms holding\n  wherever the arm ended up, so these numbers are "
              f"suspect. If publish_decimation\n  was just lowered, raise it back and "
              f"re-run before trusting anything above.")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out).expanduser() if args.out else OUT_ROOT / f"{stamp}_jointid_{backend.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "joint_id.json").write_text(json.dumps(
        dict(backend=backend.name, anchor_q=q0.tolist(), reference=ref,
             config=vars(cfg), results={str(k): v for k, v in results.items()}),
        indent=2))
    print(f"\nwrote {out}/joint_id.json")


if __name__ == "__main__":
    main()
