"""Direct-RPyC Franka driver for the bimanual plugin.

Bypasses net_franky.franky (singleton (IP,PORT)) by calling rpyc.classic.connect
once per arm. Motion construction and state packing run server-side via helpers
installed at connect time, so each per-loop op is one RPyC round-trip per arm.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rpyc
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

VELOCITY_COMMAND_DURATION_MS = 100
NUM_JOINTS = 7
EE_DELTA_DIMS = 6

DEFAULT_REQUEST_TIMEOUT_S = 5.0
RPYC_TIMEOUT_S = 10

_JACOBIAN_CACHE_Q_THRESHOLD = 0.50  # rad, L-inf
_JOINT_RELATIVE_DYNAMICS = (1.0, 0.25, 1.0)
_EE_DELTA_RELATIVE_DYNAMICS = (1.0, 0.3, 1.0)
_TORQUE_THRESHOLD = 100.0  # Nm
_FORCE_THRESHOLD = 200.0   # N
_JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

_RECOVERABLE_ERRORS = (
    "UDP receive: Timeout",
    "communication_constrains_violation",
    'current mode ("Reflex")',
    "type of motion cannot change",
)

# (q, dq, jacobian, ee_pos, ee_rot_xyzw, ee_twist)
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]


# All motion construction lives here so each loop op is one RPyC round-trip.
# brine encodes immutable types only — tuples of native floats are brineable,
# but lists are NOT (they cross the wire as netrefs, costing one round-trip
# per element access and spamming AttributeError into the server log when
# numpy probes them for __array__). Always exchange data as tuples.
# CBRobot.get_last_callback_data leaks state_mutex on AttributeError, so we
# read cb_robot.state directly under a `with` block and recover any mutex left
# locked by a previously crashed session.
_SERVER_HELPERS = f"""
import threading, numpy as _np
import franky as _fr
import net_franky.cb_robot as _cbm

if not _cbm.state_mutex.acquire(blocking=False):
    _cbm.state_mutex = threading.Lock()
    _cbm.state = None
else:
    _cbm.state_mutex.release()

_DUR = _fr.Duration({VELOCITY_COMMAND_DURATION_MS})
_EE_DYN = _fr.RelativeDynamicsFactor(*{_EE_DELTA_RELATIVE_DYNAMICS!r})
_ZERO3 = _np.zeros(3)
_ZERO_J = _np.zeros({NUM_JOINTS})

def init_robot(ip, ee):
    r = _cbm.CBRobot(ip)
    r.recover_from_errors()
    if ee:
        r.relative_dynamics_factor = _fr.RelativeDynamicsFactor(*{_EE_DELTA_RELATIVE_DYNAMICS!r})
    else:
        r.relative_dynamics_factor = _fr.RelativeDynamicsFactor(*{_JOINT_RELATIVE_DYNAMICS!r})
    r.set_collision_behavior({_TORQUE_THRESHOLD}, {_FORCE_THRESHOLD})
    r.set_joint_impedance({_JOINT_STIFFNESS!r})
    return r

def get_state(robot):
    with _cbm.state_mutex:
        s = _cbm.state
    s = s.robot_state if s is not None else robot.state
    return (
        tuple(float(x) for x in s.q),
        tuple(float(x) for x in s.dq),
        tuple(float(x) for x in s.O_T_EE.translation),
        tuple(float(x) for x in s.O_T_EE.quaternion),
        tuple(float(x) for x in s.O_dP_EE_c.linear) + tuple(float(x) for x in s.O_dP_EE_c.angular),
    )

def get_jacobian(robot):
    j = _np.asarray(robot.model.zero_jacobian(_fr.Frame.EndEffector, robot.state))
    return tuple(float(x) for x in j.flat)

def send_jv(robot, vel):
    robot.move(_fr.JointVelocityMotion(_np.asarray(vel, dtype=_np.float64), _DUR), asynchronous=True)

def send_ee(robot, twist):
    t = _np.asarray(twist, dtype=_np.float64)
    robot.move(_fr.CartesianVelocityMotion(_fr.Twist(t[:3], t[3:]), _DUR, _EE_DYN), asynchronous=True)

def stop(robot, use_ee):
    if use_ee:
        m = _fr.CartesianVelocityMotion(_fr.Twist(_ZERO3, _ZERO3), _DUR, _EE_DYN)
    else:
        m = _fr.JointVelocityMotion(_ZERO_J, _DUR)
    robot.move(m, asynchronous=False)
"""


class RobotDriver:
    """One arm: one RPyC connection, one robot handle, one set of helpers.

    Single-threaded per-instance; the wrapper executor owns serialization.
    """

    def __init__(self, server_ip: str, robot_ip: str, port: int, use_ee_delta: bool = False):
        self.use_ee_delta = use_ee_delta
        self._jac: NDArray | None = None
        self._jac_q: NDArray | None = None

        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(_SERVER_HELPERS)
        ns = self._conn.namespace
        self._rpc_state = ns["get_state"]
        self._rpc_jacobian = ns["get_jacobian"]
        self._rpc_send_jv = ns["send_jv"]
        self._rpc_send_ee = ns["send_ee"]
        self._rpc_stop = ns["stop"]
        self.robot = ns["init_robot"](robot_ip, use_ee_delta)

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def get_kinematic_state(self) -> KinematicSnapshot:
        q_l, dq_l, p_l, r_l, v_l = self._rpc_state(self.robot)
        q = np.array(q_l)
        if self._jac is None or float(np.max(np.abs(q - self._jac_q))) > _JACOBIAN_CACHE_Q_THRESHOLD:
            self._jac = np.array(self._rpc_jacobian(self.robot)).reshape(6, 7)
            self._jac_q = q.copy()
        return q, np.array(dq_l), self._jac, np.array(p_l), np.array(r_l), np.array(v_l)

    def send_joint_velocity(self, vel: list[float]) -> None:
        """Joint-space velocity RPC (joint-mode teleop and `home()` when not in EE mode)."""
        # tuple() so brine encodes by value (lists go over as netrefs).
        try:
            self._rpc_send_jv(self.robot, tuple(vel))
        except Exception as e:
            if any(t in str(e) for t in _RECOVERABLE_ERRORS):
                try:
                    self.robot.recover_from_errors()
                except Exception:
                    pass
            logger.warning("send_joint_velocity: %s", e)

    def send_velocity(self, vel: list[float]) -> None:
        """Cartesian twist when `use_ee_delta`, else joint velocity (normal teleop)."""
        # tuple() so brine encodes by value (lists go over as netrefs).
        rpc = self._rpc_send_ee if self.use_ee_delta else self._rpc_send_jv
        try:
            rpc(self.robot, tuple(vel))
        except Exception as e:
            if any(t in str(e) for t in _RECOVERABLE_ERRORS):
                try:
                    self.robot.recover_from_errors()
                except Exception:
                    pass
            logger.warning("send_velocity: %s", e)

    def stop(self) -> None:
        self._rpc_stop(self.robot, self.use_ee_delta)

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


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

    def _gather(self, fn, names, timeout_s: float | None = None) -> dict[str, Any]:
        futs = [(n, self._pool.submit(fn, n)) for n in names]
        return {n: f.result(timeout=timeout_s) for n, f in futs}

    def current_kinematic_state(self, name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> KinematicSnapshot:
        return self.drivers[name].get_kinematic_state()

    def current_kinematic_state_batch(self, names: list[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> dict[str, KinematicSnapshot]:
        return self._gather(lambda n: self.drivers[n].get_kinematic_state(), names, timeout_s)

    def move_joint_velocity_batch(self, vels: dict[str, list], asynchronous: bool = True) -> None:
        self._gather(lambda n: self.drivers[n].send_joint_velocity(vels[n]), list(vels))

    def move_ee_delta_batch(self, twists: dict[str, list], asynchronous: bool = True) -> None:
        self._gather(lambda n: self.drivers[n].send_velocity(twists[n]), list(twists))

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
