"""Franka gripper driver with non-blocking width commands.

The gripper runs on its own RPyC connection so width commands never share a
transport with arm motion. A single background worker keeps `grasp()` and
`open()` off the caller thread.

Backed by `pylibfranka.Gripper` (libfranka's gripper TCP port, separate from the
FCI control channel, so it coexists with the arm's torque loop in the same server
process). pylibfranka has no `grasp_async`/`open_async` the way franky did -- both
`grasp()` and `move()` block until the motion finishes -- so the executor is now
the only thing keeping callers off the wire, and an in-flight command is dropped
rather than queued behind the worker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time
import threading

import rpyc

logger = logging.getLogger(__name__)

RPYC_TIMEOUT_S = 10


class FrankaGripper:
    GRIPPER_TRUE_MAX_MM = 80.0
    _MOVE_SPEED_M_S = 0.5
    _INTERPOLATE_SPEED = 110
    _START_OFFSET_S = 0.6
    # _ASYNC_MOVE_SPEED_M_S = 0.20
    # Keep every meaningful width update so the latest command reaches the gripper.
    # _TARGET_CHANGE_THRESH_MM = 0.8
    _DEFAULT_FORCE = 10.0

    def __init__(self, name: str = "", server_ip: str = "", robot_ip: str = "", port: int = 0, do_print: bool = False):
        self.name = name
        self.do_print = do_print
        self._position_mm = self.GRIPPER_TRUE_MAX_MM
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}gripper")

        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(
            """
import pylibfranka as _pf

# Franka Hand tops out at 0.5 m/s; libfranka raises on anything above it,
# where franky silently accepted the same value.
_MAX_SPEED = 0.5

def init_gripper(ip):
    return _pf.Gripper(ip)

def home_gripper(controller):
    controller.homing()
    return bool(controller.move(0.04, 0.5))

def grasp_gripper(controller, width_m, speed_m_s, force_n):
    # Wide epsilons: we want the fingers to close and hold, not to assert that
    # an object of a known width ended up between them.
    return bool(controller.grasp(width_m, 1.0, force_n, 1.0, 1.0))
    # return bool(controller.move(0.002, min(speed_m_s, _MAX_SPEED)))

def open_gripper(controller, speed_m_s):
    return bool(controller.move(0.04, min(speed_m_s, _MAX_SPEED)))

def close_gripper(controller):
    return None
"""
        )
        ns = self._conn.namespace
        self._controller = ns["init_gripper"](robot_ip)
        self._rpc_home = ns["home_gripper"]
        self._rpc_close = ns["close_gripper"]
        self._rpc_grasp = ns["grasp_gripper"]
        self._rpc_open = ns["open_gripper"]
        self._is_open = True
        self._position_ts: float | None = None
        self._last_send: float = time.time()
        # Worker handoff: at most one command in flight, at most one queued.
        self._lock = threading.Lock()
        self._pending: tuple | None = None
        self._running = False

    @staticmethod
    def _clamp_mm(position_mm: float) -> float:
        return float(max(0.0, min(FrankaGripper.GRIPPER_TRUE_MAX_MM, position_mm)))

    @property
    def position(self) -> float | None:
        if self._position_ts is None:
            return self._position_mm
        elapsed = (time.monotonic() - (self._position_ts + self._START_OFFSET_S))
        if elapsed < 0:
            return self._position_mm
        return self._clamp_mm(self._position_mm + (-1 if self._position_mm == self.GRIPPER_TRUE_MAX_MM else 1) * elapsed * self._INTERPOLATE_SPEED)

    @property
    def gripper_state(self) -> int | None:
        return None

    def move(self, position_mm: float, speed: float = _MOVE_SPEED_M_S, blocking: bool = False) -> bool:
        if position_mm < self.GRIPPER_TRUE_MAX_MM / 2 and self._is_open and time.time() - self._last_send > 0.5:
            self._is_open = False
            self.grasp(0.0, speed, self._DEFAULT_FORCE, from_mm=self.GRIPPER_TRUE_MAX_MM)
            self._last_send = time.time()
        elif position_mm > self.GRIPPER_TRUE_MAX_MM / 2 and not self._is_open and time.time() - self._last_send > 0.5:
            self._is_open = True
            self.open(speed, from_mm=0.0)
            self._last_send = time.time()
        return True

    def home(self) -> bool:
        result = bool(self._rpc_home(self._controller))
        self._position_mm = self.GRIPPER_TRUE_MAX_MM
        self._position_ts = None
        return result

    def home_async(self) -> threading.Thread:
        thread = threading.Thread(target=self.home, daemon=True)
        thread.start()
        return thread

    def grip(self, force_n: float, speed: float, blocking: bool = True):
        return self.grasp(10.0, speed, force_n)

    def grasp(self, width: float, speed: float, force_n: float, from_mm: float | None = None):
        self._submit(self._rpc_grasp, (self._controller, width, speed, force_n), from_mm)

    def open(self, speed: float, from_mm: float | None = None):
        self._submit(self._rpc_open, (self._controller, speed), from_mm)

    def _submit(self, rpc, args: tuple, from_mm: float | None = None) -> None:
        """Coalesce onto the worker, latest command wins.

        pylibfranka's grasp/move block for the whole motion, so queueing every
        command would replay stale ones seconds late. But DROPPING one desyncs
        `_is_open`, which move() has already committed: the open issued while a
        grasp was still running vanished, `_is_open` said open, and the elif
        never fired again -- the gripper stayed shut until reconnect.
        """
        with self._lock:
            self._pending = (rpc, args, from_mm)
            if self._running:
                return
            self._running = True
        self._executor.submit(self._drain)

    def _drain(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
                if job is None:
                    self._running = False
                    return
            rpc, args, from_mm = job
            if from_mm is not None:
                # Stamp when the motion starts, not when it was queued: position
                # is dead-reckoned from here and a queued command can sit behind
                # a blocking grasp for most of a second.
                self._position_mm = from_mm
                self._position_ts = time.monotonic()
            self._call(rpc, *args)

    @staticmethod
    def _call(rpc, *args) -> None:
        try:
            rpc(*args)
        except Exception as e:
            # A failed grasp (nothing between the fingers) raises; not fatal.
            logger.warning("gripper command failed: %s", e)

    def ack_fast_stop(self) -> bool:
        return True

    def set_verbose(self, verbose: bool = True) -> bool:
        return True

    def bye(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        try:
            self._rpc_close(self._controller)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
