"""Franka gripper driver with non-blocking width commands.

The gripper runs on its own RPyC connection so width commands never share a
transport with arm motion. A single background worker keeps `grasp()` and
`open()` off the caller thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import rpyc

RPYC_TIMEOUT_S = 10


class FrankaGripper:
    GRIPPER_TRUE_MAX_MM = 80.0
    _MOVE_SPEED_M_S = 1.0
    # _ASYNC_MOVE_SPEED_M_S = 0.20
    # Keep every meaningful width update so the latest command reaches the gripper.
    # _TARGET_CHANGE_THRESH_MM = 0.8
    _DEFAULT_FORCE = 10.0

    def __init__(self, name: str = "", server_ip: str = "", robot_ip: str = "", port: int = 0, do_print: bool = False):
        self.name = name
        self.do_print = do_print
        self._position_mm = self.GRIPPER_TRUE_MAX_MM
        self._last_sent_position_mm: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{name}gripper")

        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(
            """
import franky as _fr

def init_gripper(ip):
    return _fr.Gripper(ip)

def home_gripper(controller):
    controller.homing()
    return True

def grasp_gripper(controller, width_m, speed_m_s, force_n):
    controller.move_async(width_m, speed_m_s)
    return True

def open_gripper(controller, speed_m_s):
    controller.open_async(speed_m_s)
    return True

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

    @staticmethod
    def _clamp_mm(position_mm: float) -> float:
        return float(max(0.0, min(FrankaGripper.GRIPPER_TRUE_MAX_MM, position_mm)))

    @property
    def position(self) -> float | None:
        return self._position_mm

    @property
    def gripper_state(self) -> int | None:
        return None

    def move(self, position_mm: float, speed: float = _MOVE_SPEED_M_S, blocking: bool = False) -> bool:
        if position_mm < self.GRIPPER_TRUE_MAX_MM / 2 and self._is_open:
            self._is_open = False
            self.grasp(0.0, speed, self._DEFAULT_FORCE)
        elif position_mm > self.GRIPPER_TRUE_MAX_MM / 2 and not self._is_open:
            self._is_open = True
            self.open(speed)
        return True

    def home(self) -> bool:
        result = bool(self._rpc_home(self._controller))
        self._position_mm = self.GRIPPER_TRUE_MAX_MM
        self._last_sent_position_mm = self.GRIPPER_TRUE_MAX_MM
        return result

    def home_async(self) -> threading.Thread:
        thread = threading.Thread(target=self.home, daemon=True)
        thread.start()
        return thread

    def grip(self, force_n: float, speed: float, blocking: bool = True):
        return self.grasp(10.0, speed, force_n)

    def grasp(self, width: float, speed: float, force_n: float):
        self._executor.submit(self._rpc_grasp, self._controller, width, speed, force_n)

    def open(self, speed: float):
        self._executor.submit(self._rpc_open, self._controller, speed)

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
