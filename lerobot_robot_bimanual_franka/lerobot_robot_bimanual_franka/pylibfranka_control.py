#!/usr/bin/env python3
"""The 1 kHz torque control loop, in its own process.

Deliberately contains NO RPyC and no networking beyond libfranka itself. That is
the entire point: co-hosting an RPyC server with this loop meant every request
handler held the GIL for ~2 ms of protocol work against a 1 ms tick budget, so
the loop overran, libfranka aborted with communication_constraints_violation,
and each abort latched a stiff joint hold mid-trajectory -- a visible jerk and
the reason repeated sweeps disagreed.

Goals arrive and state leaves through pylibfranka_shm; see that module for the
layout. Single-threaded on purpose, so the GIL is uncontended: publishing is a
decimated block of numpy stores into shared memory, cheap enough to keep inline
rather than pay a second thread's wakeup. BLAS is pinned to one thread above --
at 7x7 its pool costs more in barrier sync than the arithmetic it splits.

The loop is PIPELINED, and that ordering is load-bearing. What the robot grades
is not how long a tick takes but how soon the command follows the state: with
the law computed before the write, response ran ~400 us on the ticks that
recomputed and libfranka dropped them, pinning control_command_success_rate at
0.47-0.72 no matter how much total slack the tick had. So each tick writes the
torque prepared during the previous tick's slack -- only ``_enforce_limits`` and
the write itself sit in the response path -- and then spends the remaining
~600 us computing the next one. Measured 1.000 success rate holding and
0.99-1.00 tracking, zero aborts either way.

The cost is that tau is one tick (1 ms) older than the state it is applied
against. That is strictly better than the alternative it replaced: a dropped
command leaves the robot on its *previous* command anyway, for an unbounded and
nondeterministic staleness rather than a fixed 1 ms. ``_enforce_limits`` is
deliberately left in the response path so it still sees ``tau_J_d`` from the
fresh state -- it is the only bound on what reaches the joints.

Launched as a child of pylibfranka_server.py:
    python pylibfranka_control.py --robot-ip <ip> --shm <name>
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

# Must precede numpy: BLAS reads these at load and the pool size is then fixed.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pylibfranka

try:
    from .franka_jacobian import zero_jacobian
    from .osc_torque_controller import (
        JOINT_TORQUE_LIMITS, JointImpedanceController, OSCTorqueController,
        mat_to_quat_xyzw,
    )
    from . import pylibfranka_shm as shm
    from . import sim_dynamics
    from .torque_config import torque
except ImportError:
    from franka_jacobian import zero_jacobian  # type: ignore[no-redef]
    from osc_torque_controller import (  # type: ignore[no-redef]
        JOINT_TORQUE_LIMITS, JointImpedanceController, OSCTorqueController,
        mat_to_quat_xyzw,
    )
    import pylibfranka_shm as shm  # type: ignore[no-redef]
    import sim_dynamics  # type: ignore[no-redef]
    from torque_config import torque  # type: ignore[no-redef]

logger = logging.getLogger("control")

_DIAG = np.diag_indices(7)

# Every constant below comes from config/control.yaml (`torque:`), which is where
# the rationale for each value lives. This file runs on the NUC, so it reads them
# through torque_config -- see that module for how the values get there.
NUM_JOINTS = 7

# Recompute the control law every Nth tick and hold tau in between. This is not
# an approximation of the sim -- it IS the sim: robosuite steps mujoco at
# macros.SIMULATION_TIMESTEP = 2 ms and calls run_controller() once per substep
# (robots/single_arm.py: set_goal only on policy_step, run_controller always), so
# the law runs at 500 Hz there while the goal changes at 20 Hz. Running it at the
# full 1 kHz was the less faithful choice, and it cost ~150 us/tick we did not
# have -- compute ran 355 us against a 1000 us budget with read taking 640, so
# over half the ticks landed late and libfranka aborted on
# communication_constraints_violation. The torque limiter still runs every tick:
# it is the only bound on what actually reaches the joints.
_CONTROL_DECIMATION = int(torque("loop.control_decimation"))

# The ONE limit layer. osc.py's clip_torques plus the rate limit libfranka
# mandates -- nothing else rescales tau anywhere in this stack.
_TAU_LIMIT = np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64)
_MAX_TORQUE_RATE = float(torque("limits.max_torque_rate_nm_s"))

# Measured on THIS arm (scripts/measure_joint_friction.py) at 0.10-0.20 rad/s.
# What this assist must clear is BREAKAWAY, i.e. static friction, which is >= the
# kinetic curve; extrapolating to the zero-speed intercept instead under-assists
# (it cost X 80% -> 39% of command). Joints 5-6 remain poorly resolved.
_FRICTION_COULOMB = np.asarray(torque("friction.coulomb_nm"), dtype=np.float64)
# Band width sets the assist's incremental gain, 1 + friction_kc/tau_eps_fraction, and
# tau carries the measured dq, so the assist re-injects encoder noise at that gain.
# Widen the band to reduce it; do NOT filter the assist. See _friction_feedforward.
_FRICTION_TAU_EPS = float(torque("friction.tau_eps_fraction")) * _FRICTION_COULOMB

# What the assist actually cancels: real breakaway MINUS the friction sim already
# has. mujoco's Panda carries dof_frictionloss on every joint, so driving the FR3
# to frictionless would overshoot the reference, not match it -- the residual is
# meant to play sim's own frictionloss. Never negative: joint 7's breakaway is only
# ~3x this, and a joint measured below it needs no assist rather than a reversed one.
_SIM_FRICTIONLOSS = np.asarray(torque("friction.sim_frictionloss_nm"), dtype=np.float64)
_FRICTION_ASSIST_NM = np.maximum(_FRICTION_COULOMB - _SIM_FRICTIONLOSS, 0.0)

# Standing torque libfranka's model misses, ADDED to the OSC law. osc.py is pure PD
# and cannot reject a static disturbance; EE_DELTA re-anchors the goal on the measured
# pose, so that error becomes drift rather than an offset. See config/control.yaml.
_BIAS_NM = np.asarray(torque("bias.joint_nm"), dtype=np.float64)

# Run the OSC law on robosuite's plant model rather than the FR3's, and realise the
# joint acceleration it produces here. osc.py's uncoupling error is a function of M,
# so the law is only osc.py's law when it is evaluated on osc.py's M. See
# osc_torque_controller.run_controller and config/control.yaml.
_EMULATE_SIM_PLANT = bool(torque("osc.emulate_sim_plant"))

# Reflected rotor inertia, which libfranka's Model.mass() omits -- it returns the
# LINK rigid-body inertia only. Added to the diagonal so the whole loop (OSC lambda,
# nullspace, joint impedance) sees the arm's true joint inertia. See config/control.yaml
# for the measurement; joint 7's model value is 31x low, which is why yaw was dead.
_ROTOR_INERTIA = np.asarray(torque("rotor_inertia_kg_m2"), dtype=np.float64)

_STALE_GOAL_TIMEOUT_S = float(torque("loop.stale_goal_timeout_s"))
_PUBLISH_DECIMATION = int(torque("loop.publish_decimation"))
_RECOVERABLE = ("communication_constraint", "communication_constrains", "reflex",
                "udp receive: timeout", "command not possible in the current mode")
_ARM_ATTEMPTS = int(torque("loop.arm_attempts"))
_ARM_BACKOFF_S = float(torque("loop.arm_backoff_s"))
# Reflex thresholds. Not a bound on tau -- _enforce_limits remains the only one -- but
# on the external wrench libfranka INFERS, which our own commanded force inflates.
_COLLISION = tuple(torque(f"collision.{k}") for k in
                   ("lower_torque_nm", "upper_torque_nm", "lower_force_n", "upper_force_n"))


def _friction_feedforward(kc_pos, kc_neg, tau, tau_bias):
    """Assist the commanded torque past breakaway; zero command, zero assist.

    `tau_bias` is everything in tau that is NOT a task command -- the coriolis
    feedforward and the OSC's nullspace regulator -- so `tau - tau_bias` is the
    command alone. Both have to come out, and the nullspace term is the one that
    is easy to miss: it is zero only while the arm sits at its reference, it grows
    the moment teleop moves the arm off it, and being EE-invariant it is invisible
    to every task-space check. The band below saturates at ~5% of breakaway, so an
    assist keyed off it is a near-full Coulomb vector that is NOT in the nullspace
    -- measured 0.3-2.2 N of EE force under a strictly zero delta. EE_DELTA
    re-anchors goal_pos on the measured pose every step, so that force does not
    settle to an offset: it integrates into drift.

    DIRECTIONAL: breakaway on this arm differs by which way a joint turns, so
    each joint carries two gains and the one applied is selected by the sign of
    the commanded torque. `kc_pos` applies where (tau - tau_bias) > 0.

    Selecting on the commanded torque -- the same quantity the tanh takes -- is
    what keeps this continuous. The gain switches exactly where tanh is zero, so
    the product never jumps; only its slope changes. Selecting on dq instead
    would both reintroduce a discontinuity at zero crossing AND cancel the
    friction holding a still arm, which is the walking failure below.

    Never sign this by dq. EE_DELTA re-anchors its goal on the measured pose, so
    residual friction is the only thing holding the arm at zero command -- cancel
    it and a nudge makes the arm walk.

    Keep it MEMORYLESS. The tanh band gives this a large incremental gain
    (1 + kc/band) and tau carries the measured dq, so it does amplify encoder
    noise -- but low-passing the assist to fix that is a trap:
    a first-order lag inside a gain-(1+g) path is a lag network, pole before
    zero, and a 25 Hz corner cost 39 deg of phase at 52 Hz. Tried on hardware,
    it made the shake worse. The only safe lever is the gain itself
    (_FRICTION_TAU_EPS), or kc, which set_tuning can sweep live.
    """
    band = np.tanh((tau - tau_bias) / _FRICTION_TAU_EPS)
    kc = np.where(band >= 0.0, kc_pos, kc_neg)
    return kc * _FRICTION_ASSIST_NM * band


class ControlLoop:
    # Per-section worst-case us, reset at each health log. Class-level so an
    # instance built with object.__new__ (the tests) still has one.
    # "pub" is split out from "read" because it used to hide inside it: _prev_t3 was
    # stamped before _publish, so the publish cost was billed to the idle wait and a
    # publish-rate change looked free in the health line. It is not -- raising the
    # publish rate to 500 Hz cost ~2% of the command success rate.
    _PROF_KEYS = dict.fromkeys(("read", "resp", "law", "pub"), 0.0)
    _prof = dict(_PROF_KEYS)
    _prof_sum = dict(_PROF_KEYS)
    _prof_n = dict(_PROF_KEYS)
    _prev_t3 = 0.0
    _raw_tau = None

    def __init__(self, robot_ip: str, channel: shm.ShmChannel):
        self.robot_ip = robot_ip
        self.ch = channel
        cfg = pylibfranka.RealtimeConfig.kEnforce
        self.robot = pylibfranka.Robot(robot_ip, cfg)
        self.model = self.robot.load_model()

        # Passed explicitly, never left to the constructor default. This argument was
        # omitted for as long as the default was True, so config/control.yaml's
        # uncouple_pos_ori could not reach the loop at all and the arm ran sim-parity
        # uncoupling no matter what the yaml said -- silently, on both sides of a
        # deploy. Anything the law reads must arrive here by name.
        self.osc = OSCTorqueController(num_joints=NUM_JOINTS,
                                       uncouple_pos_ori=bool(torque("osc.uncouple_pos_ori")),
                                       lambda_rcond=float(torque("osc.lambda_rcond")))
        self.joint = JointImpedanceController(num_joints=NUM_JOINTS)
        self.control = None
        self.recovery_count = 0
        # Ticks the law asked past the clamp: beyond it the arm is under maximum
        # force, not under the OSC law. Signals a gain out of range for the pose,
        # and is the only saturation signal now that nothing else scales tau.
        self.clamp_trips = 0
        # MODE_TORQUE window state. Counted like clamp_trips because a tripped window
        # means the row the caller is about to record is not a measurement.
        self.torque_trips = 0
        self._tau_origin = np.zeros(NUM_JOINTS)
        self._tau_tripped = False
        self._last_tau = np.zeros(NUM_JOINTS)
        self._raw_tau = None
        self._last_J = None
        self._last_J_q = None
        # Set on re-arm: MODE_OSC must not resume driving toward the goal it held
        # before the fault. CLAUDE.md claimed the arm re-arms "holding the pose it
        # actually ended up in"; that was only ever true in MODE_HOLD, and in OSC the
        # stale goal is a step input on a freshly armed controller -- one visible jerk
        # per recovery, which is what six reflexes in a 10 s episode looked like.
        self._reanchor = False
        # Per-loop rather than read straight from the module constant so the parity
        # harness can exercise the ported law and the emulation separately -- they
        # assert different things (identical tau vs identical joint acceleration).
        self._emulate_sim = _EMULATE_SIM_PLANT
        self._last_mode = None
        self._last_cmd_seq = -1.0
        self._stale = False
        # Last seqlock-consistent goal; the torn-read fallback in _goal().
        self._last_goal = None

        self._prof = dict(self._PROF_KEYS)
        self._prof_sum = dict(self._PROF_KEYS)
        self._prof_n = dict(self._PROF_KEYS)
        self._prev_t3 = 0.0

    # ---- arming -----------------------------------------------------------

    def arm(self) -> None:
        last = None
        # Before arming, not after: thresholds cannot be changed while control is active.
        try:
            self.robot.set_collision_behavior(*_COLLISION)
        except Exception as e:
            logger.warning("set_collision_behavior failed, running factory reflexes: %s", e)
        for attempt in range(_ARM_ATTEMPTS):
            try:
                self.control = self.robot.start_torque_control()
                if attempt:
                    logger.info("armed on attempt %d", attempt + 1)
                return
            except Exception as e:
                last = e
                logger.warning("arming attempt %d failed: %s", attempt + 1, e)
                try:
                    self.robot.automatic_error_recovery()
                except Exception:
                    pass
                time.sleep(_ARM_BACKOFF_S)
        raise RuntimeError(f"could not arm {self.robot_ip}: {last}")

    # ---- helpers ----------------------------------------------------------

    def _jacobian(self, q, ee_pos):
        if self._last_J is not None and np.array_equal(q, self._last_J_q):
            return self._last_J
        J = zero_jacobian(q, ee_pos_base=ee_pos)
        self._last_J, self._last_J_q = J, q
        return J


    @staticmethod
    def _enforce_limits(tau, tau_prev, duration):
        """The single limit layer: rate-limit against tau_J_d, then clamp.

        Runs every tick in the response path, so it sees the fresh state even on
        the ticks the law is not recomputed. Nothing upstream or downstream of
        this rescales tau -- a gain that asks for more than the clamp reads as a
        saturated joint (clamp_trips), not as a silently shrunk control law.
        """
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        # Capped AT the 1 ms control period, never above it. libfranka grades
        # controller_torque_discontinuity per control cycle, so a late tick does not
        # earn a bigger allowed step -- it is the one time you least want one. The old
        # upper clip of 1e-2 let a single cycle move 8 Nm against libfranka's 1 Nm,
        # which stayed latent only while the law commanded small torques; adding the
        # rotor inertia made them large enough to trip it on every late tick.
        dt = min(float(duration.to_sec()), 1e-3)
        step = _MAX_TORQUE_RATE * max(dt, 1e-4)
        return np.clip(np.clip(tau, tau_prev - step, tau_prev + step), -_TAU_LIMIT, _TAU_LIMIT)

    def _compute_tau(self, state, goal):
        q = np.asarray(state.q, dtype=np.float64)
        dq = np.asarray(state.dq, dtype=np.float64)
        M = np.asarray(self.model.mass(state), dtype=np.float64).reshape(7, 7, order="F")
        M[_DIAG] += _ROTOR_INERTIA
        coriolis = np.asarray(self.model.coriolis(state), dtype=np.float64)

        mode = goal[shm.G_MODE]
        cmd_seq = goal[shm.G_CMD_SEQ]
        if cmd_seq != self._last_cmd_seq:
            self._last_cmd_seq = cmd_seq
            self._goal_ts = time.monotonic()
            self._stale = False
            if mode == shm.MODE_OSC:
                # All-zero G_NULLSPACE is the block's "unset" encoding, not a
                # posture: the caller passed nullspace_q=None and franka_process
                # zero-filled it. q=0 is not even reachable (joint 4's range is
                # entirely negative), so treating it as a reference would pull the
                # arm toward a configuration it cannot hold. Leave whatever
                # reference is already latched -- run() seeds it from the arm's
                # own q at arm time.
                ns = goal[shm.G_NULLSPACE]
                self.osc.set_goal(goal[shm.G_POS], _quat_to_mat(goal[shm.G_QUAT]),
                                  goal[shm.G_KP], goal[shm.G_KD],
                                  ns if np.any(ns) else None)
            elif mode == shm.MODE_JOINT:
                self.joint.set_goal(goal_q=goal[shm.G_JOINT_Q], kp=goal[shm.G_JOINT_KP],
                                    damping_ratio=goal[shm.G_JOINT_RATIO])
            elif mode == shm.MODE_JOINT_VEL:
                # The velocity setpoint rides G_JOINT_Q.
                self.joint.set_goal(goal_dq=goal[shm.G_JOINT_Q],
                                    damping_ratio=goal[shm.G_JOINT_RATIO])
            elif mode == shm.MODE_TORQUE:
                # Latch the window's origin on the GOAL, not on the live q: taking it
                # from q each tick would let the joint walk indefinitely, one window
                # per tick, which is the failure this window exists to prevent.
                self.joint.set_goal(goal_q=goal[shm.G_JOINT_Q], kp=1.0,
                                    damping_ratio=1.0)
                self._tau_origin = np.asarray(goal[shm.G_JOINT_Q], dtype=np.float64).copy()
                self._tau_tripped = False

        if mode != shm.MODE_FLOAT and getattr(self, "_goal_ts", 0.0):
            if time.monotonic() - self._goal_ts > _STALE_GOAL_TIMEOUT_S:
                if not self._stale:
                    logger.warning("goal stale, holding current pose")
                    self._stale = True
                mode = shm.MODE_HOLD

        # Travel window for MODE_TORQUE, in the same place and the same form as the
        # stale-goal transition above: it swaps the MODE, it does not touch tau. Once
        # tripped it stays tripped until a new goal arrives, so a joint cannot chatter
        # in and out of the window at 500 Hz.
        if mode == shm.MODE_TORQUE:
            j = int(goal[shm.G_TAU_JOINT])
            window = float(goal[shm.G_TAU_WINDOW])
            if not (0 <= j < NUM_JOINTS) or window <= 0.0:
                mode = shm.MODE_HOLD          # malformed goal is never a torque command
            else:
                dq_max = float(goal[shm.G_TAU_DQ_MAX])
                over_travel = abs(q[j] - self._tau_origin[j]) > window
                # Velocity is the binding one on the wrist and travel is on the
                # proximal joints; neither substitutes for the other. dq_max <= 0 is
                # treated as "not set" and leaves the velocity check off, so an older
                # caller still gets the travel window.
                over_speed = dq_max > 0.0 and abs(dq[j]) > dq_max
                if self._tau_tripped or over_travel or over_speed:
                    if not self._tau_tripped:
                        logger.warning(
                            "torque window: joint %d at %.4f rad past goal (limit %.4f), "
                            "%.3f rad/s (limit %.3f) -- latching to hold", j + 1,
                            q[j] - self._tau_origin[j], window, dq[j], dq_max)
                        self._tau_tripped = True
                        self.torque_trips += 1
                    mode = shm.MODE_HOLD

        if mode == shm.MODE_HOLD and self._last_mode != shm.MODE_HOLD:
            self.joint.set_goal(goal_q=q, kp=1.0, damping_ratio=1.0)
        self._last_mode = mode

        if mode == shm.MODE_FLOAT:
            return np.zeros(NUM_JOINTS), q, dq, None
        if mode == shm.MODE_TORQUE:
            # Impedance-hold every joint, then REPLACE the held joint's torque with
            # the feedforward value -- not add to it. Adding would leave a kp*e term
            # in series with the command, which is the position-servo coupling this
            # mode exists to remove: tau would stop being the independent variable.
            # The other joints stay held because one joint under open-loop torque with
            # the rest limp is an arm falling over.
            tau = self.joint.run_controller(q, dq, M, coriolis, position_hold=True)
            j = int(goal[shm.G_TAU_JOINT])
            tau[j] = float(goal[shm.G_TAU_FF][j])
            return tau, q, dq, None
        if mode == shm.MODE_OSC:
            T = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
            if self._reanchor:
                self.osc.reset_goal(T[:3, 3], T[:3, :3])
                self._reanchor = False
            J = self._jacobian(q, T[:3, 3])
            tw = J @ dq
            M_sim = b_sim = None
            if self._emulate_sim:
                # One call, one FK pass: ~141 us against the law's ~1 ms of slack.
                # Not cached on q -- q changes every tick while moving, so a cache
                # keyed on it only ever adds a comparison.
                M_sim, b_sim = sim_dynamics.mass_and_bias(q, dq)
            tau = self.osc.run_controller(
                ee_pos=T[:3, 3], ee_ori_mat=T[:3, :3], ee_pos_vel=tw[:3], ee_ori_vel=tw[3:],
                J_full=J, q=q, dq=dq, mass_matrix=M, coriolis=coriolis,
                mass_matrix_sim=M_sim, bias_sim=b_sim)
        else:
            tau = self.joint.run_controller(q, dq, M, coriolis,
                                            position_hold=(mode != shm.MODE_JOINT_VEL))
        # OSC only: home/hold have no sim counterpart and friction is what keeps
        # them quiet -- assisting there buzzes on the hold's own standing torque.
        kc_pos, kc_neg = goal[shm.G_FRICTION_KC_POS], goal[shm.G_FRICTION_KC_NEG]
        if mode == shm.MODE_OSC and (np.any(kc_pos) or np.any(kc_neg)):
            # Nullspace torque is bias, not command: it is nonzero whenever the arm
            # is off its reference, and assisting it leaks task-space force.
            tau = tau + _friction_feedforward(kc_pos, kc_neg, tau,
                                              coriolis + self.osc.nullspace_torque)
        # AFTER the assist, never before: this is a plant correction, not a command,
        # and folding it in earlier would let the assist read it as one and amplify it.
        if mode == shm.MODE_OSC:
            tau = tau + _BIAS_NM
        return tau, q, dq, None

    def _goal(self):
        """Latest consistent goal, or the last one if the seqlock kept losing.

        The fallback must be the last GOOD goal, not an unlocked re-read of the
        block. A torn copy can pair a bumped G_CMD_SEQ with a stale G_POS, and
        that latches: _compute_tau sees a new sequence and calls osc.set_goal on
        the half-written pose, which then drives the arm until the next push.
        Holding the previous goal leaves G_CMD_SEQ unchanged, so nothing latches
        and the stale-goal watchdog still runs.
        """
        goal = self.ch.read_goal()
        if goal is not None:
            self._last_goal = goal
        elif self._last_goal is None:
            self._last_goal = np.zeros(shm.GOAL_SIZE)
            self._last_goal[shm.G_MODE] = shm.MODE_HOLD
        return self._last_goal

    def _acc(self, key: str, us: float) -> None:
        if us > self._prof[key]:
            self._prof[key] = us
        self._prof_sum[key] += us
        self._prof_n[key] += 1

    def _publish(self, state, q, dq):
        T = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
        J = self._jacobian(q, T[:3, 3])
        self.ch.write_state({
            shm.S_Q: q, shm.S_DQ: dq, shm.S_POS: T[:3, 3],
            shm.S_QUAT: mat_to_quat_xyzw(T[:3, :3]), shm.S_TWIST: J @ dq,
            shm.S_RECOVERY: float(self.recovery_count),
            shm.S_TAU_CMD: self._last_tau,
            shm.S_TAU_MEAS: np.asarray(state.tau_J, dtype=np.float64),
            shm.S_TAU_EXT: np.asarray(state.tau_ext_hat_filtered, dtype=np.float64),
            shm.S_SUCCESS_RATE: float(getattr(state, "control_command_success_rate", 1.0)),
            shm.S_CLAMP_TRIPS: float(self.clamp_trips),
            shm.S_TORQUE_TRIP: float(self.torque_trips),
            shm.S_SIM_CLIP: float(self.osc.sim_clip_ticks),
            shm.S_LAMBDA_TRUNC: float(self.osc.lambda_trunc_ticks),
            shm.S_ALIVE: 1.0,
        })

    def run(self) -> None:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
        except (PermissionError, OSError):
            logger.warning("no SCHED_FIFO; user needs @realtime")
        gc.collect(); gc.freeze(); gc.disable()

        state0 = self.robot.read_once()
        self.osc.initial_joint = np.asarray(state0.q, dtype=np.float64)
        self.arm()
        logger.info("torque control armed on %s", self.robot_ip)

        tick, health_ts, worst_us = 0, time.monotonic(), 0.0
        while self.ch.goal[shm.G_RUNNING] != 0.0:
            try:
                if self.control is None:
                    self._reanchor = True
                    self._last_mode = None
                    self._raw_tau = None
                    self.arm()
                state, duration = self.control.readOnce()
                _t0 = time.perf_counter()
                q = np.asarray(state.q, dtype=np.float64)
                dq = np.asarray(state.dq, dtype=np.float64)
                if self._raw_tau is None:               # first tick, or just re-armed
                    self._raw_tau = self._compute_tau(state, self._goal())[0]
                # Shape and send FIRST, from the torque prepared last tick. Only
                # the limiter and the write sit between the state arriving and
                # the command leaving.
                tau = self._enforce_limits(
                    self._raw_tau, np.asarray(state.tau_J_d, dtype=np.float64), duration)
                self._last_tau = tau
                self.control.writeOnce(pylibfranka.Torques(tau.tolist()))
                _t1 = time.perf_counter()
                self._acc("resp", (_t1 - _t0) * 1e6)

                # Everything below runs in the slack before the next state.
                if np.any(np.abs(self._raw_tau) > _TAU_LIMIT):
                    self.clamp_trips += 1
                if tick % _CONTROL_DECIMATION == 0:
                    self._raw_tau = self._compute_tau(state, self._goal())[0]
                _t2 = time.perf_counter()
                self._acc("law", (_t2 - _t1) * 1e6)
                if self._prev_t3:
                    self._acc("read", (_t0 - self._prev_t3) * 1e6)

                worst_us = max(worst_us, (_t1 - _t0) * 1e6)
                # Incremented last so publish lands on a law tick and _publish's
                # Jacobian hits the cache instead of always missing it.
                if tick % _PUBLISH_DECIMATION == 0:
                    self._publish(state, q, dq)
                    self._acc("pub", (time.perf_counter() - _t2) * 1e6)
                # Stamped after publish, so "read" is the idle wait for the next state
                # and nothing the loop itself does is billed to it.
                self._prev_t3 = time.perf_counter()

                # Out of the publish branch: the health line is what says whether the
                # publish rate is affordable, so it must not be sampled only on the
                # ticks that pay for it.
                now = time.monotonic()
                if now - health_ts >= 10.0:
                    health_ts = now
                    rate = float(getattr(state, "control_command_success_rate", 1.0))
                    log = logger.warning if rate < 0.95 else logger.info
                    log("success rate %.3f, worst response %.0f us, recoveries %d, "
                        "clamp trips %d%s  [us mean/max: %s]", rate, worst_us,
                        self.recovery_count, self.clamp_trips,
                        "  <- MISSING DEADLINES" if rate < 0.95 else "",
                        " ".join(
                            f"{k}={self._prof_sum[k] / max(self._prof_n[k], 1):.0f}/{v:.0f}"
                            for k, v in self._prof.items()))
                    worst_us = 0.0
                    for k in self._prof:
                        self._prof[k] = self._prof_sum[k] = self._prof_n[k] = 0.0
                tick += 1
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if any(t in msg.lower() for t in _RECOVERABLE):
                    self.recovery_count += 1
                    try:
                        self.robot.automatic_error_recovery()
                    except Exception:
                        pass
                self.control = None
                logger.warning("rt loop: %s", msg)
                self.ch.state[shm.S_RECOVERY] = float(self.recovery_count)
                time.sleep(0.001)

        self.ch.state[shm.S_ALIVE] = 0.0
        try:
            self.robot.stop()
        except Exception:
            pass
        logger.info("control loop exited")


def _quat_to_mat(q):
    x, y, z, w = np.asarray(q, dtype=np.float64) / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True)
    ap.add_argument("--shm", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s CONTROL %(levelname)s %(message)s")
    sys.setswitchinterval(5.0e-5)
    ch = shm.ShmChannel(name=args.shm)
    try:
        ControlLoop(args.robot_ip, ch).run()
    finally:
        ch.state[shm.S_ALIVE] = 0.0
        ch.close()


if __name__ == "__main__":
    main()
