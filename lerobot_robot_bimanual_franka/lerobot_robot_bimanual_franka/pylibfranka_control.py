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
torque prepared during the previous tick's slack -- only the speed guard, the
rate limiter and the write itself sit in the response path (~92 us mean, 207 us
worst) -- and then spends the remaining ~600 us computing the next one. Measured
1.000 success rate holding and 0.99-1.00 tracking, zero aborts either way.

The cost is that tau is one tick (1 ms) older than the state it is applied
against. That is strictly better than the alternative it replaced: a dropped
command leaves the robot on its *previous* command anyway, for an unbounded and
nondeterministic staleness rather than a fixed 1 ms. The guard and the rate
limiter are deliberately left in the response path so both still see the fresh
state -- they bound what actually reaches the joints.

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
        JOINT_TORQUE_LIMITS, JointImpedanceController, OSCTorqueController, mat_to_quat_xyzw,
    )
    from . import pylibfranka_shm as shm
except ImportError:
    from franka_jacobian import zero_jacobian  # type: ignore[no-redef]
    from osc_torque_controller import (  # type: ignore[no-redef]
        JOINT_TORQUE_LIMITS, JointImpedanceController, OSCTorqueController, mat_to_quat_xyzw,
    )
    import pylibfranka_shm as shm  # type: ignore[no-redef]

logger = logging.getLogger("control")

NUM_JOINTS = 7
_TORQUE_THRESHOLD, _FORCE_THRESHOLD = 100.0, 200.0
_JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

# Recompute the control law every Nth tick and hold tau in between. This is not
# an approximation of the sim -- it IS the sim: robosuite steps mujoco at
# macros.SIMULATION_TIMESTEP = 2 ms and calls run_controller() once per substep
# (robots/single_arm.py: set_goal only on policy_step, run_controller always), so
# the law runs at 500 Hz there while the goal changes at 20 Hz. Running it at the
# full 1 kHz was the less faithful choice, and it cost ~150 us/tick we did not
# have -- compute ran 355 us against a 1000 us budget with read taking 640, so
# over half the ticks landed late and libfranka aborted on
# communication_constraints_violation. The guard and the rate limiter still run
# every tick: both are bounds on what actually reaches the joints.
_CONTROL_DECIMATION = 2

_TAU_SAFETY_FACTOR = 0.8
_TAU_LIMIT = np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64) * _TAU_SAFETY_FACTOR
_MAX_TORQUE_RATE = 1000.0

# Speed guard on the RESULT: every client-settable knob multiplies commanded
# torque and several compose. Must stay OUTSIDE the envelope a sim-parity action
# produces (0.31 m/s, 3.06 rad/s -- the latter is ~0.9 of rated wrist velocity)
# or it rescales the control law instead of bounding a runaway.
_JOINT_VELOCITY_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
_JOINT_VELOCITY_TRIP = 0.95
_EE_LINEAR_TRIP, _EE_ANGULAR_TRIP = 1.20, 6.00
_GUARD_HARD_STOP = 1.5
_BRAKE_KD = np.array([40.0, 40.0, 40.0, 40.0, 20.0, 15.0, 10.0])

# Zero-speed intercepts measured on THIS arm (scripts/measure_joint_friction.py);
# the torque at the sweep speed carries ~1.0 Nms/rad of viscous that must not be
# fed back as a speed-independent term. Joints 5-6 are poorly resolved, low side.
_FRICTION_COULOMB = np.array([1.02, 1.04, 0.67, 1.04, 0.15, 0.25, 0.41])
_FRICTION_TAU_EPS = 0.20 * _FRICTION_COULOMB   # stiction band; sharper limit-cycles
_FRICTION_DQ_EPS = 0.02   # rad/s below which a joint counts as stuck

_STALE_GOAL_TIMEOUT_S = 0.5
_PUBLISH_DECIMATION = 10
_RECOVERABLE = ("communication_constraint", "communication_constrains", "reflex",
                "udp receive: timeout", "command not possible in the current mode")
_ARM_ATTEMPTS, _ARM_BACKOFF_S = 6, 0.3


def _friction_feedforward(kc, tau, coriolis):
    """Assist the commanded torque past breakaway; zero command, zero assist.

    Never sign this by dq. EE_DELTA re-anchors its goal on the measured pose, so
    residual friction is the only thing holding the arm at zero command -- cancel
    it and a nudge makes the arm walk.
    """
    return kc * _FRICTION_COULOMB * np.tanh((tau - coriolis) / _FRICTION_TAU_EPS)


class ControlLoop:
    # Per-section worst-case us, reset at each health log. Class-level so an
    # instance built with object.__new__ (the tests) still has one.
    _PROF_KEYS = dict.fromkeys(("read", "resp", "law"), 0.0)
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
        self.robot.set_collision_behavior(
            [_TORQUE_THRESHOLD] * NUM_JOINTS, [_TORQUE_THRESHOLD] * NUM_JOINTS,
            [_FORCE_THRESHOLD] * 6, [_FORCE_THRESHOLD] * 6)
        self.robot.set_joint_impedance(_JOINT_STIFFNESS)
        self.model = self.robot.load_model()

        self.osc = OSCTorqueController(num_joints=NUM_JOINTS)
        self.joint = JointImpedanceController(num_joints=NUM_JOINTS)
        self.control = None
        self.recovery_count = 0
        self.guard_trips = 0
        # Ticks the law asked past the clamp: beyond it the arm is under maximum
        # force, not under the OSC law. Signals a gain out of range for the pose.
        self.clamp_trips = 0
        self._last_tau = np.zeros(NUM_JOINTS)
        self._raw_tau = None
        self._last_J = None
        self._last_J_q = None
        self._last_mode = None
        self._last_cmd_seq = -1.0
        self._stale = False
        self._guard_ts = 0.0

        self._prof = dict(self._PROF_KEYS)
        self._prof_sum = dict(self._PROF_KEYS)
        self._prof_n = dict(self._PROF_KEYS)
        self._prev_t3 = 0.0

    # ---- arming -----------------------------------------------------------

    def arm(self) -> None:
        last = None
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

    def _speed_guard(self, tau, dq):
        over = float(np.max(np.abs(dq) / (_JOINT_VELOCITY_LIMITS * _JOINT_VELOCITY_TRIP)))
        if self._last_J is not None:
            tw = self._last_J @ dq
            over = max(over, float(np.linalg.norm(tw[:3]) / _EE_LINEAR_TRIP),
                       float(np.linalg.norm(tw[3:]) / _EE_ANGULAR_TRIP))
        if over <= 1.0:
            return tau
        self.guard_trips += 1
        now = time.monotonic()
        if now - self._guard_ts > 1.0:
            self._guard_ts = now
            logger.warning("SPEED GUARD %.2fx over envelope -> %s", over,
                           "braking" if over >= _GUARD_HARD_STOP else "cutting back")
        if over >= _GUARD_HARD_STOP:
            return np.clip(-_BRAKE_KD * dq, -_TAU_LIMIT, _TAU_LIMIT)
        return tau * ((_GUARD_HARD_STOP - over) / (_GUARD_HARD_STOP - 1.0))

    @staticmethod
    def _limit(tau, tau_prev, duration):
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        dt = float(np.clip(duration.to_sec(), 1e-3, 1e-2))
        step = _MAX_TORQUE_RATE * dt
        return np.clip(np.clip(tau, tau_prev - step, tau_prev + step), -_TAU_LIMIT, _TAU_LIMIT)

    def _compute_tau(self, state, goal):
        q = np.asarray(state.q, dtype=np.float64)
        dq = np.asarray(state.dq, dtype=np.float64)
        M = np.asarray(self.model.mass(state), dtype=np.float64).reshape(7, 7, order="F")
        coriolis = np.asarray(self.model.coriolis(state), dtype=np.float64)

        mode = goal[shm.G_MODE]
        cmd_seq = goal[shm.G_CMD_SEQ]
        if cmd_seq != self._last_cmd_seq:
            self._last_cmd_seq = cmd_seq
            self._goal_ts = time.monotonic()
            self._stale = False
            if mode == shm.MODE_OSC:
                self.osc.set_goal(goal[shm.G_POS], _quat_to_mat(goal[shm.G_QUAT]),
                                  goal[shm.G_KP], goal[shm.G_KD], goal[shm.G_NULLSPACE])
            elif mode == shm.MODE_JOINT:
                self.joint.set_goal(goal_q=goal[shm.G_JOINT_Q], kp=goal[shm.G_JOINT_KP],
                                    damping_ratio=goal[shm.G_JOINT_RATIO])
            elif mode == shm.MODE_JOINT_VEL:
                # The velocity setpoint rides G_JOINT_Q.
                self.joint.set_goal(goal_dq=goal[shm.G_JOINT_Q],
                                    damping_ratio=goal[shm.G_JOINT_RATIO])
            self.osc.uncoupling = bool(goal[shm.G_UNCOUPLE])

        if mode != shm.MODE_FLOAT and getattr(self, "_goal_ts", 0.0):
            if time.monotonic() - self._goal_ts > _STALE_GOAL_TIMEOUT_S:
                if not self._stale:
                    logger.warning("goal stale, holding current pose")
                    self._stale = True
                mode = shm.MODE_HOLD

        if mode == shm.MODE_HOLD and self._last_mode != shm.MODE_HOLD:
            self.joint.set_goal(goal_q=q, kp=1.0, damping_ratio=1.0)
        self._last_mode = mode

        if mode == shm.MODE_FLOAT:
            return np.zeros(NUM_JOINTS), q, dq, None
        if mode == shm.MODE_OSC:
            T = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
            J = self._jacobian(q, T[:3, 3])
            tw = J @ dq
            tau = self.osc.run_controller(
                ee_pos=T[:3, 3], ee_ori_mat=T[:3, :3], ee_pos_vel=tw[:3], ee_ori_vel=tw[3:],
                J_full=J, q=q, dq=dq, mass_matrix=M, coriolis=coriolis,
                joint_damping_kv=float(goal[shm.G_JOINT_DAMPING_KV]))
        else:
            tau = self.joint.run_controller(q, dq, M, coriolis,
                                            position_hold=(mode != shm.MODE_JOINT_VEL))
        # OSC only: home/hold have no sim counterpart and friction is what keeps
        # them quiet -- assisting there buzzes on the hold's own standing torque.
        kc = float(goal[shm.G_FRICTION_KC])
        if kc and mode == shm.MODE_OSC:
            tau = tau + _friction_feedforward(kc, tau, coriolis)
        return tau, q, dq, None

    def _goal(self):
        goal = self.ch.read_goal()
        return self.ch.goal.copy() if goal is None else goal   # torn read -> hold

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
            shm.S_GUARD_TRIPS: float(self.guard_trips),
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
                # the guard, the rate limiter and the write sit between the state
                # arriving and the command leaving.
                tau = self._speed_guard(self._raw_tau, dq)
                tau = self._limit(tau, np.asarray(state.tau_J_d, dtype=np.float64), duration)
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
                self._prev_t3 = _t2

                worst_us = max(worst_us, (_t1 - _t0) * 1e6)
                # Incremented last so publish lands on a law tick and _publish's
                # Jacobian hits the cache instead of always missing it.
                if tick % _PUBLISH_DECIMATION == 0:
                    self._publish(state, q, dq)
                    now = time.monotonic()
                    if now - health_ts >= 10.0:
                        health_ts = now
                        rate = float(getattr(state, "control_command_success_rate", 1.0))
                        log = logger.warning if rate < 0.95 else logger.info
                        log("success rate %.3f, worst response %.0f us, recoveries %d, "
                            "guard trips %d, clamp trips %d%s  [us mean/max: %s]", rate, worst_us,
                            self.recovery_count, self.guard_trips, self.clamp_trips,
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
