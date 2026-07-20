#!/usr/bin/env python3
"""RPyC service exposing a single pylibfranka.Robot's torque-control loop.

Design
------
pylibfranka's real-time interface is: call `start_torque_control()` once to
get an `ActiveControlBase`, then in a tight loop call `readOnce()` (blocks
until the next 1kHz tick) and `writeOnce(Torques(tau))`. The `ActiveControlBase`
object is a live handle into libfranka's realtime thread and is not picklable
/ shareable across a process boundary, so it must be created and driven from
*inside* this server process.

Control-loop frequency
-----------------------
The read/write pair now runs on a dedicated background thread per session
(`_RobotSession._rt_loop`), started once in `start_torque_control()` and
free-running for the life of the session. This thread is the *only* thing
that calls `readOnce`/`writeOnce`, and it never waits on the client: it
always writes whatever torque command was most recently pushed to it
(defaulting to zero torques until the client sends one), so the actual
control rate is exactly whatever `readOnce()` paces to (libfranka enforces
1kHz on its realtime thread) rather than being bounded by RPC round-trip
time or client call cadence.

The client (BimanualFranka's RobotDriver, over RPyC) now interacts with the
loop asynchronously instead of tick-by-tick:
  - `exposed_set_tau(robot_ip, tau)`: cheap, non-blocking; overwrites the
    torque command the RT thread will write on its *next* cycle. Safe to
    call slower or faster than 1kHz -- if the client falls behind, the RT
    thread just keeps re-applying the last command it has; it never stalls
    waiting for a new one.
  - `exposed_get_latest(robot_ip)`: non-blocking; returns the most recent
    state bundle plus a monotonically increasing sequence number. Calling
    this faster than 1kHz will return the same bundle/seq more than once.
  - `exposed_wait_next(robot_ip, since_seq, timeout)`: blocks (via a
    condition variable, no polling) until the RT thread has produced a
    bundle newer than `since_seq`, or until `timeout` elapses. This is the
    closest analogue to the old per-tick round trip, for callers whose
    control law wants to run in lockstep with the real 1kHz loop, but it
    never *drives* the loop the way the old `tick()` did -- the RT thread
    keeps running even if nobody ever calls this.

`exposed_tick` is kept only as a thin backwards-compatible wrapper (push
tau, block for the next bundle) -- new code should prefer set_tau +
wait_next/get_latest since `tick` still couples the *caller's* observed
rate to the RT thread's, even though the RT thread itself is no longer
throttled by the caller.

Run with (matches the existing net_franky launch pattern):
    chrt -f 80 rpyc_classic -p 18812 --host 0.0.0.0
but classic rpyc won't auto-expose this service; run this file directly
instead (see __main__ below), on the same core/priority setup as before.
Note: with the RT loop now running continuously on its own thread inside
this process regardless of client activity, that thread is the one that
actually needs the realtime scheduling priority -- see `_RobotSession start`
which sets the thread's priority via `threading` is not sufficient on its
own for RT scheduling; the process-level `chrt -f 80` on this launch still
governs the OS scheduling class/priority for all threads in the process,
including the new RT loop thread.

IMPORTANT: FrankaGripper (franka_gripper.py) shares this same server_ip:port
and connects via `rpyc.classic.connect`, which hardcodes the client to
expect the server to be a `SlaveService` (arbitrary code execution --
`conn.execute(...)`, `conn.namespace`, `getmodule`, etc. -- this is how the
gripper installs its own `init_gripper`/`grasp_gripper`/... helpers at
connect time, same pattern the old franky server used for the arm). Because
of this, FrankaTorqueService below inherits from SlaveService rather than
plain rpyc.Service, so one server process/port serves both: the gripper's
classic arbitrary-execution interface, and this file's custom exposed_*
methods for torque control. Dropping the SlaveService base (e.g. to
"simplify" this service) will break FrankaGripper's connection with
`AttributeError: '...' object has no attribute 'getmodule'`.
"""


from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pylibfranka
import rpyc
import os
from rpyc.core.service import SlaveService
from rpyc.utils.server import ThreadedServer

logger = logging.getLogger(__name__)

DEFAULT_PORT = 18812

# franky's Rust-side collision defaults were (100 Nm torque, 200 N force);
# kept identical here so switching backends doesn't silently change the
# robot's safety envelope.
_TORQUE_THRESHOLD = 100.0
_FORCE_THRESHOLD = 200.0
_JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

# Recoverable-error substrings, same list used by the franky RobotDriver.
_RECOVERABLE_ERRORS = (
    "communication_constrains_violation",
    "reflex",
    "Reflex",
    "libfranka: ",  # generic transient control exceptions worth a recovery attempt
)

# If the client hasn't pushed a fresh tau in this long, the RT loop falls
# back to zero torques rather than continuing to replay a stale command
# indefinitely (e.g. client crashed / RPC connection dropped mid-motion).
_STALE_TAU_TIMEOUT_S = 0.5


def _quat_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (x, y, z, w) quaternion."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


class _RobotSession:
    """Owns the Robot + Model + ActiveControlBase for one arm, and the
    dedicated thread that free-runs the readOnce/writeOnce loop at
    whatever pace libfranka paces `readOnce()` to (nominally 1kHz),
    independent of client RPC cadence."""

    def __init__(self, robot_ip: str, use_realtime: bool = True):
        cfg = pylibfranka.RealtimeConfig.kEnforce if use_realtime else pylibfranka.RealtimeConfig.kIgnore
        logger.info("connecting to Robot(%s)", robot_ip)
        self.robot = pylibfranka.Robot(robot_ip, cfg)
        logger.info("Robot connected")
        self.robot.set_collision_behavior(
            [_TORQUE_THRESHOLD] * 7,
            [_TORQUE_THRESHOLD] * 7,
            [_FORCE_THRESHOLD] * 6,
            [_FORCE_THRESHOLD] * 6,
        )
        self.robot.set_joint_impedance(_JOINT_STIFFNESS)
        self.model = self.robot.load_model()
        logger.info("model loaded")
        self.control = None  # ActiveControlBase, created lazily in start_torque_control
        self.recovery_count = 0

        # --- state shared with the RT thread ---
        self._session_lock = threading.Lock()  # guards control/session lifecycle
        self._tau_lock = threading.Lock()  # guards _pending_tau / _pending_tau_ts
        self._pending_tau = (0.0,) * 7
        self._pending_tau_ts = 0.0  # monotonic time tau was last set; 0 = never set

        self._state_cond = threading.Condition()  # guards + signals _latest*
        self._latest_bundle = None
        self._latest_err = None
        self._seq = 0

        self._rt_thread: threading.Thread | None = None
        self._running = False

    # ---- lifecycle ----

    def start_torque_control(self) -> None:
        with self._session_lock:
            if self.control is not None:
                return
            logger.info("calling start_torque_control()")
            self.control = self.robot.start_torque_control()
            logger.info("torque control armed")
            self._running = True
            self._rt_thread = threading.Thread(
                target=self._rt_loop, name="franka-rt-loop", daemon=True
            )
            self._rt_thread.start()

    def stop(self) -> None:
        self._running = False  # plain bool write; RT loop only reads this
        try:
            self.robot.stop()  # call immediately, WITHOUT _session_lock --
                                # must be able to interrupt start_torque_control_locked()
                                # even while it's mid-call and holding _session_lock
        except Exception as e:
            logger.warning("stop(): %s", e)

        if self._rt_thread is not None:
            self._rt_thread.join(timeout=2.0)
            if self._rt_thread.is_alive():
                logger.error("RT thread still alive after stop()+join timeout")
            self._rt_thread = None

        with self._session_lock:
            self.control = None

    # ---- client-facing, non-blocking-ish accessors ----

    def set_tau(self, tau: tuple) -> None:
        with self._tau_lock:
            self._pending_tau = tau
            self._pending_tau_ts = time.monotonic()

    def get_latest(self) -> tuple:
        """Non-blocking: whatever the RT thread most recently produced."""
        with self._state_cond:
            return self._latest_bundle, self._latest_err, self._seq

    def wait_next(self, since_seq: int, timeout: float | None) -> tuple:
        """Block until seq > since_seq or timeout elapses."""
        with self._state_cond:
            ok = self._state_cond.wait_for(
                lambda: self._seq > since_seq, timeout=timeout
            )
            if not ok:
                return self._latest_bundle, self._latest_err, self._seq, False
            return self._latest_bundle, self._latest_err, self._seq, True

    # ---- the RT loop itself ----

    def _rt_loop(self) -> None:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
        except PermissionError:
            logger.warning("could not set SCHED_FIFO on RT loop thread; needs @realtime group")

        logger.info("starting realtime control loop")
        while self._running:
            try:
                # logger.info("entered rt control")
                if self.control is None:
                    # Session faulted; try to re-establish before the next
                    # readOnce so the loop keeps free-running rather than
                    # dying on the first recoverable error.
                    self.start_torque_control_locked()

                # logger.info("attempt control read")
                state, _duration = self.control.readOnce()
                # print('desired', state.tau_J_d)
                # print('actual', state.tau_J)
                logger.info("complete control read")

                with self._tau_lock:
                    tau = self._pending_tau 
                    tau_ts = self._pending_tau_ts
                if tau_ts == 0.0 or (time.monotonic() - tau_ts) > _STALE_TAU_TIMEOUT_S:
                    tau = (0.0,) * 7  # no fresh command from client -> hold zero torque

                cmd = pylibfranka.Torques(list(tau))
                self.control.writeOnce(cmd)

                bundle = self._bundle_state(state)
                with self._state_cond:
                    self._latest_bundle = bundle
                    self._latest_err = None
                    self._seq += 1
                    self._state_cond.notify_all()

            except Exception as e:  # noqa: BLE001 - keep the RT thread alive
                msg = str(e)
                if any(t in msg for t in _RECOVERABLE_ERRORS):
                    self.recovery_count += 1
                    try:
                        self.robot.automatic_error_recovery()
                    except Exception:
                        pass
                    self.control = None  # re-armed at top of next loop iteration
                logger.warning("_rt_loop(): %s", msg)
                with self._state_cond:
                    self._latest_err = msg
                    self._seq += 1
                    self._state_cond.notify_all()
                # Brief backoff so a persistent fault doesn't spin the loop
                # at full CPU with no readOnce() to pace it.
                time.sleep(0.001)

    def start_torque_control_locked(self) -> None:
        """Re-arm torque control from within the RT thread after a fault.
        Distinct from the public start_torque_control() (which also spawns
        the thread) -- here the thread already exists and is calling this."""
        with self._session_lock:
            if self.control is None:
                self.control = self.robot.start_torque_control()

    def _bundle_state(self, state) -> tuple:
        """Pack one RobotState snapshot into everything the client's OSC law
        needs, all derived from the *same* state so M/C/g/J/pose are mutually
        consistent for this tick."""
        q = tuple(float(v) for v in state.q)
        dq = tuple(float(v) for v in state.dq)

        # pylibfranka returns these as flat column-major arrays (matching
        # libfranka's std::array<double,N> layout), not pre-shaped matrices --
        # reshape here, before anything downstream (e.g. the J @ dq twist
        # fallback below) treats them as 2-D.
        J = np.asarray(self.model.zero_jacobian(state)).reshape(6, 7, order="C")
        M = np.asarray(self.model.mass(state)).reshape(7, 7, order="C")
        C = np.asarray(self.model.coriolis(state))  # (7,)
        g = np.asarray(self.model.gravity(state))  # (7,)

        O_T_EE = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
        ee_pos = O_T_EE[:3, 3]
        ee_quat_xyzw = _quat_from_rotation_matrix(O_T_EE[:3, :3])

        # O_dP_EE_d: desired EE twist as tracked by the internal motion
        # generator. Not populated in torque-control-only mode on all
        # firmware; fall back to J @ dq (measured twist from the Jacobian)
        # when it's all zero, which is the common case here.
        raw_twist = getattr(state, "O_dP_EE_c", None)
        if raw_twist is not None:
            ee_twist = np.asarray(raw_twist, dtype=np.float64)
            if ee_twist.shape != (6,) or not np.any(ee_twist):
                ee_twist = J @ np.array(dq)
        else:
            ee_twist = J @ np.array(dq)

        return (
            q,
            dq,
            tuple(float(x) for x in J.flat),
            tuple(float(x) for x in ee_pos),
            tuple(float(x) for x in ee_quat_xyzw),
            tuple(float(x) for x in ee_twist),
            tuple(float(x) for x in M.flat),
            tuple(float(x) for x in C),
            tuple(float(x) for x in g),
        )


class FrankaTorqueService(SlaveService):
    """Inherits SlaveService so FrankaGripper's `rpyc.classic.connect` (which
    hardcodes the client to expect a SlaveService peer -- see module
    docstring) keeps working against this same server_ip:port, while also
    exposing our own exposed_* methods for torque control.

    Sessions keyed by robot_ip so multiple arms can share one server process
    if desired.
    """

    _sessions: dict[str, _RobotSession] = {}
    _sessions_lock = threading.Lock()

    def on_connect(self, conn):
        self._conn = conn
        # Required: SlaveService.on_connect sets up allow_all_attrs / pickle /
        # getattr / custom-exception config the gripper's classic client
        # handshake and conn.execute()/conn.namespace depend on. Skipping
        # this breaks the gripper exactly like a plain rpyc.Service does.
        super().on_connect(conn)

    def on_disconnect(self, conn):
        super().on_disconnect(conn)

    def exposed_init_robot(self, robot_ip: str, use_realtime: bool = True) -> bool:
        with self._sessions_lock:
            if robot_ip not in self._sessions:
                self._sessions[robot_ip] = _RobotSession(robot_ip, use_realtime)
        return True

    def exposed_start_torque_control(self, robot_ip: str) -> bool:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(80))
        except PermissionError:
            logger.warning("could not RT-promote calling thread for start_torque_control")
        self._sessions[robot_ip].start_torque_control()
        return True

    def exposed_set_tau(self, robot_ip: str, tau) -> bool:
        """Push a new torque command. Non-blocking, does not wait on the RT
        loop -- just overwrites what it will write on its next cycle."""
        tau_t = tuple(float(x) for x in tau)
        self._sessions[robot_ip].set_tau(tau_t)
        return True

    def exposed_get_latest(self, robot_ip: str) -> tuple:
        """Non-blocking read of the most recent state bundle."""
        return self._sessions[robot_ip].get_latest()

    def exposed_wait_next(self, robot_ip: str, since_seq: int, timeout: float | None = None) -> tuple:
        """Blocks until the RT loop has produced a bundle newer than
        since_seq, or timeout elapses. Does not itself throttle the RT
        loop -- the loop keeps running at its own pace regardless of
        whether/how often callers use this."""
        return self._sessions[robot_ip].wait_next(since_seq, timeout)

    def exposed_tick(self, robot_ip: str, tau) -> tuple:
        """Backwards-compatible wrapper: push tau (if given), then block for
        the next fresh bundle. Prefer set_tau + wait_next/get_latest in new
        code -- this still couples the *caller's* observed rate to the RT
        loop, even though the RT loop itself is no longer throttled by the
        caller the way it was before."""
        session = self._sessions[robot_ip]
        if tau is not None:
            session.set_tau(tuple(float(x) for x in tau))
        _, _, seq = session.get_latest()
        bundle, err, _, _ = session.wait_next(seq, timeout=1.0)
        return bundle, err

    def exposed_recovery_count(self, robot_ip: str) -> int:
        return self._sessions[robot_ip].recovery_count

    def exposed_stop(self, robot_ip: str) -> bool:
        self._sessions[robot_ip].stop()
        return True

    def exposed_set_load(self, robot_ip: str, load_mass: float, f_x_cload, load_inertia) -> bool:
        self._sessions[robot_ip].robot.set_load(load_mass, list(f_x_cload), list(load_inertia))
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server = ThreadedServer(
        FrankaTorqueService,
        port=args.port,
        hostname=args.host,
        protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 10,
        },
    )
    logger.info("pylibfranka RPyC torque server listening on %s:%d", args.host, args.port)
    server.start()