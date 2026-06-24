"""Direct-RPyC Franka driver for the bimanual plugin.

Bypasses net_franky.franky (singleton (IP,PORT)) by calling rpyc.classic.connect
once per arm. Motion construction and state packing run server-side via helpers
installed at connect time, so each per-loop op is one RPyC round-trip per arm.

All control modes (JOINT_POS, EE_DELTA, EE_POS) send JointVelocityMotion; the
caller computes joint velocities (including VOsc for EE modes) before dispatch.
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

DEFAULT_REQUEST_TIMEOUT_S = 5.0
RPYC_TIMEOUT_S = 10

_JOINT_RELATIVE_DYNAMICS = (1.0, 0.25, 1.0)
_TORQUE_THRESHOLD = 100.0   # Nm
_FORCE_THRESHOLD = 200.0    # N
_JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

_RECOVERABLE_ERRORS = (
    "UDP receive: Timeout",
    "communication_constrains_violation",
    'current mode ("Reflex")',
    "type of motion cannot change",
)

# (q, dq, jacobian, mass, ee_pos, ee_rot_xyzw, ee_twist)
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]


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
_DYN = _fr.RelativeDynamicsFactor(*{_JOINT_RELATIVE_DYNAMICS!r})
_ZERO_J = _np.zeros({NUM_JOINTS})

def init_robot(ip):
    r = _cbm.CBRobot(ip)
    r.recover_from_errors()
    r.relative_dynamics_factor = _DYN
    r.set_collision_behavior({_TORQUE_THRESHOLD}, {_FORCE_THRESHOLD})
    r.set_joint_impedance({_JOINT_STIFFNESS!r})
    return r

def get_state(robot):
    with _cbm.state_mutex:
        s = _cbm.state
    s = s.robot_state if s is not None else robot.state
    q   = tuple(float(x) for x in s.q)
    dq  = tuple(float(x) for x in s.dq)
    pos = tuple(float(x) for x in s.O_T_EE.translation)
    quat = tuple(float(x) for x in s.O_T_EE.quaternion)
    vel = tuple(float(x) for x in s.O_dP_EE_c.linear) + tuple(float(x) for x in s.O_dP_EE_c.angular)
    rs = robot.state
    j = tuple(float(x) for x in _np.asarray(robot.model.zero_jacobian(_fr.Frame.EndEffector, rs)).flat)
    m = tuple(float(x) for x in _np.asarray(robot.model.mass(rs)).flat)
    return q, dq, pos, quat, vel, j, m

def send_jv(robot, vel):
    robot.move(_fr.JointVelocityMotion(_np.asarray(vel, dtype=_np.float64), _DUR), asynchronous=True)

def stop(robot):
    robot.move(_fr.JointVelocityMotion(_ZERO_J, _DUR))
"""


class RobotDriver:
    """One arm: one RPyC connection, one robot handle, one set of helpers.

    Single-threaded per-instance; the wrapper executor owns serialization.
    """

    def __init__(self, server_ip: str, robot_ip: str, port: int):
        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(_SERVER_HELPERS)
        ns = self._conn.namespace
        self._rpc_state = ns["get_state"]
        self._rpc_send_jv = ns["send_jv"]
        self._rpc_stop = ns["stop"]
        self.robot = ns["init_robot"](robot_ip)

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def get_kinematic_state(self) -> KinematicSnapshot:
        q_l, dq_l, p_l, r_l, v_l, j_l, m_l = self._rpc_state(self.robot)
        return (
            np.array(q_l),
            np.array(dq_l),
            np.array(j_l).reshape(6, NUM_JOINTS),
            np.array(m_l).reshape(NUM_JOINTS, NUM_JOINTS),
            np.array(p_l),
            np.array(r_l),
            np.array(v_l),
        )

    def send_joint_velocity(self, vel: list[float]) -> None:
        try:
            self._rpc_send_jv(self.robot, tuple(vel))
        except Exception as e:
            if any(t in str(e) for t in _RECOVERABLE_ERRORS):
                try:
                    self.robot.recover_from_errors()
                except Exception:
                    pass
            logger.warning("send_joint_velocity: %s", e)

    def stop(self) -> None:
        self._rpc_stop(self.robot)

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

    def add_robot(self, name: str, server_ip: str, robot_ip: str, port: int) -> None:
        if name in self.drivers:
            raise ValueError(f"Robot '{name}' already connected")
        self.drivers[name] = RobotDriver(server_ip, robot_ip, port)

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

    def move_joint_velocity_batch(self, vels: dict[str, list]) -> None:
        self._gather(lambda n: self.drivers[n].send_joint_velocity(vels[n]), list(vels))

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
