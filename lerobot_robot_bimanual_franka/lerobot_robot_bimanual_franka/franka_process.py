"""Direct-RPyC pylibfranka driver for the bimanual plugin.

Replaces the franky/net_franky driver. franky has no torque interface, so the
OSC law could only be approximated in the velocity domain; the NUC now runs
``pylibfranka_server.py``, which owns a 1 kHz torque loop per arm.

Consequence for this file: it no longer ships *commands* per tick, it ships
*goals*. The control law runs server-side (see that module's docstring for why),
so every method here is a cheap non-blocking push or a non-blocking read of the
latest published state -- nothing in the per-tick path blocks on the robot.

brine encodes immutable values only: tuples of native floats cross by value,
lists cross as netrefs costing a round-trip per element and spamming
AttributeError when numpy probes ``__array__``. Everything on the wire is a tuple.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rpyc
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

NUM_JOINTS = 7
EE_DELTA_DIMS = 6

DEFAULT_REQUEST_TIMEOUT_S = 5.0
RPYC_TIMEOUT_S = 10
FIRST_STATE_TIMEOUT_S = 5.0

# Server-side control modes (pylibfranka_server.MODE_*).
MODE_FLOAT = "float"
MODE_HOLD = "hold"

# (q, dq, jacobian, ee_pos, ee_rot_xyzw, ee_twist)
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]


def _unpack(bundle: tuple) -> tuple[KinematicSnapshot, int]:
    q, dq, J_flat, ee_pos, ee_quat_xyzw, ee_twist, recovery_count = bundle
    snap: KinematicSnapshot = (
        np.array(q),
        np.array(dq),
        np.array(J_flat).reshape(6, NUM_JOINTS),
        np.array(ee_pos),
        np.array(ee_quat_xyzw),
        np.array(ee_twist),
    )
    return snap, int(recovery_count)


class RobotDriver:
    """One arm: one RPyC connection to that arm's torque-server session.

    Single-threaded per-instance; the wrapper executor owns serialization.
    """

    def __init__(self, server_ip: str, robot_ip: str, port: int, use_ee_delta: bool = False):
        # Retained for call-site compatibility with the franky driver. Torque
        # control has no separate EE transport mode -- BimanualFranka picks the
        # server-side control law instead.
        self.use_ee_delta = use_ee_delta
        self.robot_ip = robot_ip
        # Cumulative recoverable-error recoveries (reflexes etc.), mirrored from
        # the server on every state read so recording scripts can flag ticks
        # where tracking was interrupted.
        self.recovery_count = 0
        self._last_snap: KinematicSnapshot | None = None

        self._conn = rpyc.connect(
            server_ip,
            port,
            config={"sync_request_timeout": RPYC_TIMEOUT_S, "allow_pickle": True},
        )
        root = self._conn.root
        root.init_robot(robot_ip, True)
        root.start_control(robot_ip)
        self._root = root

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def get_kinematic_state(self) -> KinematicSnapshot:
        """Latest state published by the RT loop. Non-blocking once running."""
        bundle, err, _ = self._root.get_state(self.robot_ip)
        if bundle is None:
            bundle, err, _, ok = self._root.wait_next(self.robot_ip, -1, FIRST_STATE_TIMEOUT_S)
            if not ok or bundle is None:
                raise ConnectionError(
                    f"no state from {self.robot_ip} after {FIRST_STATE_TIMEOUT_S}s: {err}"
                )
        if err is not None:
            logger.warning("RobotDriver(%s): %s", self.robot_ip, err)
        snap, self.recovery_count = _unpack(bundle)
        self._last_snap = snap
        return snap

    def send_osc_goal(
        self,
        goal_pos: np.ndarray,
        goal_quat_xyzw: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        nullspace_q: np.ndarray | None = None,
    ) -> None:
        self._push(
            self._root.set_osc_goal,
            _t(goal_pos),
            _t(goal_quat_xyzw),
            _t(kp),
            _t(kd),
            None if nullspace_q is None else _t(nullspace_q),
        )

    def send_joint_goal(
        self, goal_q: np.ndarray, kp: float | None = None, damping_ratio: float | None = None
    ) -> None:
        self._push(self._root.set_joint_goal, _t(goal_q), kp, damping_ratio)

    def send_joint_velocity(self, vel: list[float], kd_scale: float = 1.0) -> None:
        """Joint-velocity setpoint, tracked by a server-side impedance law."""
        self._push(self._root.set_joint_velocity_goal, _t(vel), float(kd_scale))

    def set_mode(self, mode: str) -> None:
        self._push(self._root.set_mode, mode)

    def set_tuning(self, joint_damping_kv: float | None = None) -> None:
        """Live server-side tuning; 0.0 joint damping restores exact osc.py behaviour."""
        self._push(self._root.set_tuning, joint_damping_kv)

    def _push(self, rpc, *args) -> None:
        try:
            rpc(self.robot_ip, *args)
        except Exception as e:
            logger.warning("%s(%s): %s", getattr(rpc, "__name__", "push"), self.robot_ip, e)

    def stop(self) -> None:
        """Park in a joint hold at the current pose; the arm stays where it is."""
        self.set_mode(MODE_HOLD)

    def shutdown(self) -> None:
        try:
            self._root.stop(self.robot_ip)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


def _t(v) -> tuple:
    return tuple(float(x) for x in np.asarray(v, dtype=np.float64).ravel())


class MultiRobotWrapper:
    """Manager dispatching to per-arm RobotDriver instances in parallel."""

    def __init__(self):
        self.drivers: dict[str, RobotDriver] = {}
        self._pool = ThreadPoolExecutor(max_workers=4)

    def add_robot(self, name: str, server_ip: str, robot_ip: str, port: int, use_ee_delta: bool = False) -> None:
        if name in self.drivers:
            raise ValueError(f"Robot '{name}' already connected")
        self.drivers[name] = RobotDriver(server_ip, robot_ip, port, use_ee_delta)

    @property
    def num_alive(self) -> int:
        return sum(1 for d in self.drivers.values() if d.is_alive)

    def recovery_counts(self) -> dict[str, int]:
        """Cumulative recoverable-error recoveries per arm, as of the last state read."""
        return {n: d.recovery_count for n, d in self.drivers.items()}

    def _gather(self, fn, names, timeout_s: float | None = None) -> dict[str, Any]:
        futs = [(n, self._pool.submit(fn, n)) for n in names]
        return {n: f.result(timeout=timeout_s) for n, f in futs}

    def current_kinematic_state(self, name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> KinematicSnapshot:
        return self.drivers[name].get_kinematic_state()

    def current_kinematic_state_batch(
        self, names: list[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> dict[str, KinematicSnapshot]:
        return self._gather(lambda n: self.drivers[n].get_kinematic_state(), names, timeout_s)

    def move_osc_goal_batch(self, goals: dict[str, tuple]) -> None:
        """goals[arm] = (goal_pos(3), goal_quat_xyzw(4), kp(6), kd(6), nullspace_q(7) | None)."""
        self._gather(lambda n: self.drivers[n].send_osc_goal(*goals[n]), list(goals))

    def move_joint_goal_batch(self, goals: dict[str, tuple]) -> None:
        """goals[arm] = (goal_q(7), kp | None, damping_ratio | None)."""
        self._gather(lambda n: self.drivers[n].send_joint_goal(*goals[n]), list(goals))

    def move_joint_velocity_batch(self, vels: dict[str, list], asynchronous: bool = True) -> None:
        self._gather(lambda n: self.drivers[n].send_joint_velocity(vels[n]), list(vels))

    def set_mode_all(self, mode: str) -> None:
        self._gather(lambda n: self.drivers[n].set_mode(mode), list(self.drivers))

    def set_tuning_all(self, joint_damping_kv: float | None = None) -> None:
        self._gather(lambda n: self.drivers[n].set_tuning(joint_damping_kv), list(self.drivers))

    def stop_all_motion(self) -> None:
        self._gather(lambda n: self.drivers[n].stop(), [n for n, d in self.drivers.items() if d.is_alive])

    def shutdown(self) -> None:
        try:
            self.stop_all_motion()
        except Exception:
            pass
        for d in self.drivers.values():
            d.shutdown()
        self.drivers.clear()
        self._pool.shutdown(wait=False)
