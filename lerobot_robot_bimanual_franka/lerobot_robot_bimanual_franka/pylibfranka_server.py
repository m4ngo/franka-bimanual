#!/usr/bin/env python3
"""RPyC torque-control server for one NUC. Runs on mario/luigi, not the workstation.

Replaces the ``rpyc_classic`` + net_franky/franky server. franky exposes no
torque interface, so the OSC law could only ever be approximated in the velocity
domain; pylibfranka's ``start_torque_control()`` gives the real thing.

Why the control law lives here and not in BimanualFranka
--------------------------------------------------------
pylibfranka's realtime interface is ``control = robot.start_torque_control()``
then a tight ``readOnce()`` / ``writeOnce(Torques(tau))`` loop. ``control`` is a
live handle into libfranka's realtime thread -- not picklable, not shareable
across a process boundary -- so the loop must run inside this process. And the
loop has to *recompute* tau each tick rather than replay a latched one: this is
what robosuite does (``run_controller()`` every 2 ms sim step with the goal held
between 20 Hz policy steps), and a constant torque held across a 30-50 ms policy
period has no live ``-kd * ee_vel`` term, so the arm accelerates unopposed.

The workstation therefore pushes *goals* at policy rate (``set_osc_goal`` /
``set_joint_goal``, both non-blocking) and reads state whenever it likes
(``get_state``, non-blocking). This thread owns everything in between. Measured
budget for the full robosuite-parity law on this hardware: ~107 us mean,
~164 us max, against a 1000 us tick.

Wire-format and lifecycle constraints
-------------------------------------
- **Tuples, not lists.** brine encodes immutable values; lists cross as netrefs
  costing a round-trip per element and spamming AttributeError when numpy probes
  ``__array__``. Every payload in and out of here is a tuple of native floats.
- **libfranka matrices are column-major.** ``O_T_EE`` and ``Model.mass`` return
  vectorized column-major arrays, so every reshape here is ``order="F"``.
- **``Model.zero_jacobian`` is unusable on this build.** It returns all zeros
  (verified on hardware: mass/coriolis/gravity are all sane from the same
  RobotState, only the Jacobian is zeroed), and ``franka::Frame`` is not exported
  by pylibfranka so the frame-taking overload cannot be reached from Python.
  The Jacobian therefore comes from ``franka_jacobian.zero_jacobian`` -- the
  analytic modified-DH one, anchored on the measured ``O_T_EE`` translation so it
  stays consistent with the pose error the OSC law forms. franky's
  ``zero_jacobian`` had the identical all-zero behaviour, so this is a property
  of the robot's model library here, not of either binding.
- **Torque rate limiting is mandatory.** libfranka rejects jumps beyond
  kMaxTorqueRate (1000 Nm/s); nothing enforces it for us the way
  ``franka::limitRate`` does in the C++ examples, so it is applied here against
  ``state.tau_J_d``.
- **SlaveService base is load-bearing.** FrankaGripper connects to this same
  host:port with ``rpyc.classic.connect``, which hardcodes the client to expect
  a SlaveService peer (``conn.execute``, ``conn.namespace``, ``getmodule``).
  Dropping the base to "simplify" this breaks the gripper with
  ``AttributeError: ... has no attribute 'getmodule'``.

Launch via ``run_server.sh`` (taskset + chrt -f 80); see
``scripts/deploy_nuc_server.sh`` on the workstation for the deploy path.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import threading
import time

import numpy as np
import pylibfranka
import rpyc
from rpyc.core.service import SlaveService
from rpyc.utils.server import ThreadedServer

try:  # deployed flat next to this file on the NUC; importable as a package on the workstation
    from .franka_jacobian import zero_jacobian
    from .osc_torque_controller import (
        DEFAULT_JOINT_KP,
        JOINT_TORQUE_LIMITS,
        JointImpedanceController,
        OSCTorqueController,
        mat_to_quat_xyzw,
    )
except ImportError:
    from franka_jacobian import zero_jacobian  # type: ignore[no-redef]
    from osc_torque_controller import (  # type: ignore[no-redef]
        DEFAULT_JOINT_KP,
        JOINT_TORQUE_LIMITS,
        JointImpedanceController,
        OSCTorqueController,
        mat_to_quat_xyzw,
    )

logger = logging.getLogger(__name__)

DEFAULT_PORT = 18812
NUM_JOINTS = 7

# Unchanged from the franky server so switching backends does not silently
# change the robot's safety envelope (franky's set_collision_behavior(100, 200)
# set both lower and upper thresholds to these).
_TORQUE_THRESHOLD = 100.0
_FORCE_THRESHOLD = 200.0
_JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

# Joint-space velocity damping added on top of the OSC torque. A deviation from
# osc.py, which needs none: mujoco has no joint friction/backlash and no ~1 tick
# of command latency, so the task-space -kd*ee_vel term alone is enough there. On
# hardware it leaves high-frequency joint motion undamped. Applied in OSC mode
# only, tunable live via set_tuning(); 0.0 restores exact robosuite behaviour.
_JOINT_DAMPING_KV = 2.0

# Warn when libfranka reports it is dropping our commands. This is the metric
# that turns into ["communication_constraints_violation"] once it decays far
# enough, so it is the early warning for a loop that is too slow.
_SUCCESS_RATE_WARN = 0.95
_HEALTH_LOG_PERIOD_S = 10.0

_TAU_SAFETY_FACTOR = 0.8
_TAU_LIMIT = np.asarray(JOINT_TORQUE_LIMITS, dtype=np.float64) * _TAU_SAFETY_FACTOR
_MAX_TORQUE_RATE = 1000.0  # Nm/s, libfranka kMaxTorqueRate

# Client reads at <=30 Hz; publishing every tick spends ~10% of the budget
# packing tuples nobody reads.
_PUBLISH_DECIMATION = 10

# A goal older than this means the client died or stalled. Latch a hold at
# wherever the arm is now rather than keep driving toward a stale setpoint.
_STALE_GOAL_TIMEOUT_S = 0.5

# GIL handoff granularity. The default 5 ms lets an RPC handler stall the RT
# thread past its deadline even at SCHED_FIFO 80.
_GIL_SWITCH_INTERVAL_S = 2.0e-4

# Substrings, matched case-insensitively. Note "constraint" without the closing
# 's': libfranka 0.18 emits "communication_constraints_violation" where the
# franky-era code matched "communication_constrains_violation", so nothing ever
# matched and automatic_error_recovery() was never called -- the arm sat faulted
# while the loop retried start_torque_control() forever.
_RECOVERABLE_ERRORS = (
    "communication_constraint",
    "communication_constrains",
    "reflex",
    "udp receive: timeout",
    "control_command_success_rate",
)

MODE_FLOAT = "float"
MODE_HOLD = "hold"
MODE_JOINT = "joint"
MODE_JOINT_VEL = "joint_vel"
MODE_OSC = "osc"


class _ArmSession:
    """Robot + Model + ActiveControlBase for one arm, plus the thread driving them."""

    def __init__(self, robot_ip: str, use_realtime: bool = True):
        cfg = pylibfranka.RealtimeConfig.kEnforce if use_realtime else pylibfranka.RealtimeConfig.kIgnore
        logger.info("connecting to Robot(%s)", robot_ip)
        self.robot_ip = robot_ip
        self.robot = pylibfranka.Robot(robot_ip, cfg)
        self.robot.set_collision_behavior(
            [_TORQUE_THRESHOLD] * NUM_JOINTS,
            [_TORQUE_THRESHOLD] * NUM_JOINTS,
            [_FORCE_THRESHOLD] * 6,
            [_FORCE_THRESHOLD] * 6,
        )
        self.robot.set_joint_impedance(_JOINT_STIFFNESS)
        self.model = self.robot.load_model()
        logger.info("Robot(%s) connected, model loaded", robot_ip)

        self.osc = OSCTorqueController(num_joints=NUM_JOINTS)
        self.joint = JointImpedanceController(num_joints=NUM_JOINTS)

        self.control = None
        self.recovery_count = 0

        self._session_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._mode = MODE_HOLD
        # None (not MODE_HOLD) so the first tick latches the arming pose.
        self._last_mode: str | None = None
        self._goal_ts = 0.0
        self._stale = False

        self._state_cond = threading.Condition()
        self._bundle: tuple | None = None
        self._err: str | None = None
        self._seq = 0

        self._thread: threading.Thread | None = None
        self._running = False
        # Last Jacobian computed by _compute_tau, reused by _bundle_state so a
        # publish tick doesn't pay for it twice.
        self._last_J: np.ndarray | None = None
        self._last_J_q: np.ndarray | None = None
        self.joint_damping_kv = _JOINT_DAMPING_KV
        self._health_ts = 0.0
        self._compute_max_us = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        with self._session_lock:
            if self._thread is not None:
                return
            # Hold wherever the arm currently is, so arming never produces a step.
            state = self.robot.read_once()
            self.osc.initial_joint = np.asarray(state.q, dtype=np.float64)
            self._mode = MODE_HOLD
            self._last_mode = None
            self.control = self.robot.start_torque_control()
            self._running = True
            self._thread = threading.Thread(target=self._rt_loop, name=f"rt-{self.robot_ip}", daemon=True)
            self._thread.start()
        logger.info("torque control armed on %s", self.robot_ip)

    def stop(self) -> None:
        self._running = False
        try:
            # Deliberately outside _session_lock: must be able to interrupt a
            # start_torque_control() that is mid-call and holding the lock.
            self.robot.stop()
        except Exception as e:
            logger.warning("stop(%s): %s", self.robot_ip, e)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.error("RT thread for %s still alive after join timeout", self.robot_ip)
            self._thread = None
        with self._session_lock:
            self.control = None

    # ---- goal setters, all non-blocking ----

    def set_osc_goal(self, goal_pos, goal_quat_xyzw, kp, kd, nullspace_q) -> None:
        from_quat = np.asarray(goal_quat_xyzw, dtype=np.float64)
        with self._cmd_lock:
            self.osc.set_goal(
                np.asarray(goal_pos, dtype=np.float64),
                _quat_to_mat(from_quat),
                np.asarray(kp, dtype=np.float64),
                np.asarray(kd, dtype=np.float64),
                None if nullspace_q is None else np.asarray(nullspace_q, dtype=np.float64),
            )
            self._mode = MODE_OSC
            self._goal_ts = time.monotonic()
            self._stale = False

    def set_joint_goal(self, goal_q, kp, damping_ratio) -> None:
        with self._cmd_lock:
            self.joint.set_goal(goal_q=np.asarray(goal_q, dtype=np.float64), kp=kp, damping_ratio=damping_ratio)
            self._mode = MODE_JOINT
            self._goal_ts = time.monotonic()
            self._stale = False

    def set_joint_velocity_goal(self, goal_dq, kd_scale) -> None:
        with self._cmd_lock:
            self.joint.set_goal(goal_dq=np.asarray(goal_dq, dtype=np.float64),
                                damping_ratio=float(kd_scale))
            self._mode = MODE_JOINT_VEL
            self._goal_ts = time.monotonic()
            self._stale = False

    def set_mode(self, mode: str) -> None:
        with self._cmd_lock:
            self._mode = mode
            self._goal_ts = time.monotonic()
            self._stale = False

    def set_tuning(self, joint_damping_kv: float | None = None) -> None:
        with self._cmd_lock:
            if joint_damping_kv is not None:
                self.joint_damping_kv = float(joint_damping_kv)
                logger.info("%s: joint_damping_kv = %.3f", self.robot_ip, self.joint_damping_kv)

    # ---- state accessors ----

    def get_state(self) -> tuple:
        with self._state_cond:
            return self._bundle, self._err, self._seq

    def wait_next(self, since_seq: int, timeout: float | None) -> tuple:
        with self._state_cond:
            ok = self._state_cond.wait_for(lambda: self._seq > since_seq, timeout=timeout)
            return self._bundle, self._err, self._seq, ok

    # ---- the realtime loop ----

    def _rt_loop(self) -> None:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
        except (PermissionError, OSError):
            logger.warning("could not set SCHED_FIFO on the RT thread; user needs @realtime")

        tick = 0
        while self._running:
            try:
                if self.control is None:
                    self._rearm()

                state, duration = self.control.readOnce()
                t0 = time.perf_counter()
                tau = self._compute_tau(state)
                tau = self._limit(tau, np.asarray(state.tau_J_d, dtype=np.float64), duration)
                self.control.writeOnce(pylibfranka.Torques([float(x) for x in tau]))
                self._compute_max_us = max(self._compute_max_us, (time.perf_counter() - t0) * 1e6)

                tick += 1
                if tick % _PUBLISH_DECIMATION == 0:
                    self._publish(self._bundle_state(state), None)
                    self._log_health(state)

            except Exception as e:  # keep the RT thread alive across recoverable faults
                msg = str(e)
                if any(t in msg.lower() for t in _RECOVERABLE_ERRORS):
                    self.recovery_count += 1
                    try:
                        self.robot.automatic_error_recovery()
                    except Exception:
                        pass
                self.control = None
                logger.warning("_rt_loop(%s): %s", self.robot_ip, msg)
                self._publish(None, msg)
                time.sleep(0.001)

    def _rearm(self) -> None:
        """Re-arm after a fault, holding the pose the arm actually ended up in."""
        with self._session_lock:
            if self.control is not None:
                return
            with self._cmd_lock:
                self._mode = MODE_HOLD
                self._last_mode = None
            self.control = self.robot.start_torque_control()

    def _compute_tau(self, state) -> np.ndarray:
        q = np.asarray(state.q, dtype=np.float64)
        dq = np.asarray(state.dq, dtype=np.float64)
        M = np.asarray(self.model.mass(state), dtype=np.float64).reshape(NUM_JOINTS, NUM_JOINTS, order="F")
        coriolis = np.asarray(self.model.coriolis(state), dtype=np.float64)

        # Held across the whole law: the controllers read their goal off self,
        # so releasing early lets a concurrent setter tear pos against ori.
        with self._cmd_lock:
            mode = self._mode
            if mode != MODE_FLOAT and self._goal_ts and (time.monotonic() - self._goal_ts) > _STALE_GOAL_TIMEOUT_S:
                if not self._stale:
                    logger.warning("%s: goal stale, holding current pose", self.robot_ip)
                    self._stale = True
                mode = self._mode = MODE_HOLD

            # HOLD means "stay where you are", so it has to latch q on entry --
            # otherwise it would drive to whatever joint goal happened to be left
            # over from the last home(), which is nowhere near the current pose
            # after a session of OSC teleop.
            if mode == MODE_HOLD and self._last_mode != MODE_HOLD:
                self.joint.set_goal(goal_q=q, kp=DEFAULT_JOINT_KP, damping_ratio=1.0)
            self._last_mode = mode

            if mode == MODE_FLOAT:
                return np.zeros(NUM_JOINTS)
            if mode == MODE_OSC:
                O_T_EE = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
                J = self._jacobian(q, O_T_EE[:3, 3])
                ee_twist = J @ dq
                tau = self.osc.run_controller(
                    ee_pos=O_T_EE[:3, 3],
                    ee_ori_mat=O_T_EE[:3, :3],
                    ee_pos_vel=ee_twist[:3],
                    ee_ori_vel=ee_twist[3:],
                    J_full=J,
                    q=q,
                    dq=dq,
                    mass_matrix=M,
                    coriolis=coriolis,
                )
                if self.joint_damping_kv:
                    tau = tau - self.joint_damping_kv * dq
                return tau
            return self.joint.run_controller(q, dq, M, coriolis, position_hold=(mode != MODE_JOINT_VEL))

    def _log_health(self, state) -> None:
        """Periodic loop-health line: libfranka's own accounting of how many of
        our commands it accepted, plus our worst compute time since the last log."""
        now = time.monotonic()
        if now - self._health_ts < _HEALTH_LOG_PERIOD_S:
            return
        self._health_ts = now
        rate = float(getattr(state, "control_command_success_rate", 1.0))
        msg = "%s: command success rate %.3f, worst compute %.0f us, recoveries %d"
        args = (self.robot_ip, rate, self._compute_max_us, self.recovery_count)
        if rate < _SUCCESS_RATE_WARN:
            logger.warning(msg + "  <- loop is missing deadlines", *args)
        else:
            logger.info(msg, *args)
        self._compute_max_us = 0.0

    def _jacobian(self, q: np.ndarray, ee_pos: np.ndarray) -> np.ndarray:
        """Analytic Jacobian for this q, reusing the last one when q is unchanged.

        Only skips recomputation on an exact q match, which is what happens when
        a publish tick follows the control tick that already built it -- never a
        staleness window, since a new RobotState always has a new q.
        """
        if self._last_J is not None and np.array_equal(q, self._last_J_q):
            return self._last_J
        J = zero_jacobian(q, ee_pos_base=ee_pos)
        self._last_J, self._last_J_q = J, q
        return J

    @staticmethod
    def _limit(tau: np.ndarray, tau_prev: np.ndarray, duration) -> np.ndarray:
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        tau = np.clip(tau, -_TAU_LIMIT, _TAU_LIMIT)
        dt = float(np.clip(duration.to_sec(), 1e-3, 1e-2))
        step = _MAX_TORQUE_RATE * dt
        return np.clip(tau, tau_prev - step, tau_prev + step)

    def _publish(self, bundle, err) -> None:
        """bundle=None keeps the last good snapshot (used when reporting a fault)."""
        with self._state_cond:
            if bundle is not None:
                self._bundle = bundle
            self._err = err
            self._seq += 1
            self._state_cond.notify_all()

    def _bundle_state(self, state) -> tuple:
        """One RobotState -> everything the workstation needs, all from the same tick.

        recovery_count rides along so the client's per-tick recovery_counts()
        costs no extra round-trip.
        """
        q = tuple(float(v) for v in state.q)
        dq = tuple(float(v) for v in state.dq)
        O_T_EE = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
        J = self._jacobian(np.asarray(q, dtype=np.float64), O_T_EE[:3, 3])
        ee_twist = J @ np.asarray(dq, dtype=np.float64)
        return (
            q,
            dq,
            tuple(float(x) for x in J.flat),
            tuple(float(x) for x in O_T_EE[:3, 3]),
            tuple(float(x) for x in mat_to_quat_xyzw(O_T_EE[:3, :3])),
            tuple(float(x) for x in ee_twist),
            int(self.recovery_count),
        )


def _quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw / max(float(np.linalg.norm(q_xyzw)), 1e-12)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class FrankaTorqueService(SlaveService):
    """Torque-control RPCs plus, via the SlaveService base, FrankaGripper's
    classic ``conn.execute``/``conn.namespace`` interface on the same port.

    Sessions are keyed by robot_ip and shared process-wide, so a reconnecting
    client re-attaches to the running RT loop instead of re-arming the robot.
    """

    _sessions: dict[str, _ArmSession] = {}
    _sessions_lock = threading.Lock()

    def exposed_init_robot(self, robot_ip: str, use_realtime: bool = True) -> bool:
        with self._sessions_lock:
            if robot_ip not in self._sessions:
                self._sessions[robot_ip] = _ArmSession(robot_ip, use_realtime)
        return True

    def exposed_start_control(self, robot_ip: str) -> bool:
        self._sessions[robot_ip].start()
        return True

    def exposed_read_state(self, robot_ip: str) -> tuple:
        """One-shot read that does not arm torque control. For reachability
        checks and offline Jacobian/pose verification."""
        session = self._sessions[robot_ip]
        return session._bundle_state(session.robot.read_once())

    def exposed_set_osc_goal(self, robot_ip, goal_pos, goal_quat_xyzw, kp, kd, nullspace_q=None) -> bool:
        self._sessions[robot_ip].set_osc_goal(goal_pos, goal_quat_xyzw, kp, kd, nullspace_q)
        return True

    def exposed_set_joint_goal(self, robot_ip, goal_q, kp=None, damping_ratio=None) -> bool:
        self._sessions[robot_ip].set_joint_goal(goal_q, kp, damping_ratio)
        return True

    def exposed_set_joint_velocity_goal(self, robot_ip, goal_dq, kd_scale=1.0) -> bool:
        self._sessions[robot_ip].set_joint_velocity_goal(goal_dq, kd_scale)
        return True

    def exposed_set_mode(self, robot_ip: str, mode: str) -> bool:
        self._sessions[robot_ip].set_mode(mode)
        return True

    def exposed_set_tuning(self, robot_ip: str, joint_damping_kv: float | None = None) -> bool:
        self._sessions[robot_ip].set_tuning(joint_damping_kv)
        return True

    def exposed_get_state(self, robot_ip: str) -> tuple:
        return self._sessions[robot_ip].get_state()

    def exposed_wait_next(self, robot_ip: str, since_seq: int, timeout: float | None = None) -> tuple:
        return self._sessions[robot_ip].wait_next(since_seq, timeout)

    def exposed_recovery_count(self, robot_ip: str) -> int:
        return self._sessions[robot_ip].recovery_count

    def exposed_stop(self, robot_ip: str) -> bool:
        self._sessions[robot_ip].stop()
        return True

    def exposed_set_load(self, robot_ip: str, load_mass: float, f_x_cload, load_inertia) -> bool:
        self._sessions[robot_ip].robot.set_load(load_mass, list(f_x_cload), list(load_inertia))
        return True

    # SlaveService's allow_all_attrs _rpyc_getattr looks the name up directly and
    # so skips rpyc's exposed_ prefix stripping; without these aliases only the
    # exposed_-prefixed spelling resolves over the wire.
    init_robot = exposed_init_robot
    start_control = exposed_start_control
    read_state = exposed_read_state
    set_osc_goal = exposed_set_osc_goal
    set_joint_goal = exposed_set_joint_goal
    set_joint_velocity_goal = exposed_set_joint_velocity_goal
    set_mode = exposed_set_mode
    set_tuning = exposed_set_tuning
    get_state = exposed_get_state
    wait_next = exposed_wait_next
    recovery_count = exposed_recovery_count
    stop = exposed_stop
    set_load = exposed_set_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.setswitchinterval(_GIL_SWITCH_INTERVAL_S)
    # The RT loop allocates only numpy arrays and tuples -- no reference cycles --
    # so refcounting alone reclaims it. Leaving the cycle collector on just buys
    # an unpredictable multi-hundred-microsecond pause inside a 1 ms tick.
    gc.collect()
    gc.freeze()
    gc.disable()

    server = ThreadedServer(
        FrankaTorqueService,
        port=args.port,
        hostname=args.host,
        protocol_config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": 10},
    )
    logger.info("pylibfranka torque server listening on %s:%d", args.host, args.port)
    server.start()


if __name__ == "__main__":
    main()
