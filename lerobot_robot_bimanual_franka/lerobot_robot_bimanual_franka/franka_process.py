"""Direct-RPyC Franka driver for the bimanual plugin.

EE modes dispatch CartesianVelocityMotion; joint mode uses JointVelocityMotion.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rpyc
from numpy.typing import NDArray

from .franka_fk import franka_jacobian

logger = logging.getLogger(__name__)

VELOCITY_COMMAND_DURATION_MS = 50
NUM_JOINTS = 7

DEFAULT_REQUEST_TIMEOUT_S = 5.0
RPYC_TIMEOUT_S = 10

_JOINT_RELATIVE_DYNAMICS = (1.0, 0.25, 1.0)
_EE_RELATIVE_DYNAMICS = (1.0, 0.3, 1.0)
_TORQUE_THRESHOLD = 100.0
_FORCE_THRESHOLD = 200.0
_JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

# (q, dq, jacobian, mass, ee_pos, ee_rot_xyzw, ee_twist)
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]

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
_JV_DYN = _fr.RelativeDynamicsFactor(*{_JOINT_RELATIVE_DYNAMICS!r})
_EE_DYN = _fr.RelativeDynamicsFactor(*{_EE_RELATIVE_DYNAMICS!r})
_ZERO_J = _np.zeros({NUM_JOINTS})

def init_robot(ip):
    r = _cbm.CBRobot(ip)
    r.recover_from_errors()
    r.relative_dynamics_factor = _JV_DYN
    r.set_collision_behavior({_TORQUE_THRESHOLD}, {_FORCE_THRESHOLD})
    r.set_joint_impedance({_JOINT_STIFFNESS!r})
    return r

def get_state(robot):
    with _cbm.state_mutex:
        s = _cbm.state
    s = s.robot_state if s is not None else robot.state
    q    = tuple(float(x) for x in s.q)
    dq   = tuple(float(x) for x in s.dq)
    pos  = tuple(float(x) for x in s.O_T_EE.translation)
    quat = tuple(float(x) for x in s.O_T_EE.quaternion)
    rs   = robot.state
    m    = tuple(float(x) for x in _np.asarray(robot.model.mass(rs)).flat)
    return q, dq, pos, quat, m

def send_ee(robot, twist):
    t = _np.asarray(twist, dtype=_np.float64)
    robot.move(
        _fr.CartesianVelocityMotion(_fr.Twist(t[:3], t[3:]), _DUR, _EE_DYN),
        asynchronous=True,
    )

def send_jv(robot, vel):
    robot.move(_fr.JointVelocityMotion(_np.asarray(vel, dtype=_np.float64), _DUR), asynchronous=True)

def stop_ee(robot):
    robot.move(_fr.CartesianVelocityMotion(_fr.Twist(_np.zeros(3), _np.zeros(3)), _DUR, _EE_DYN))

def stop_jv(robot):
    robot.move(_fr.JointVelocityMotion(_ZERO_J, _DUR))
"""


class RobotDriver:
    def __init__(self, server_ip: str, robot_ip: str, port: int):
        self._conn = rpyc.classic.connect(server_ip, port)
        self._conn._config["sync_request_timeout"] = RPYC_TIMEOUT_S
        self._conn.execute(_SERVER_HELPERS)
        ns = self._conn.namespace
        self._rpc_state = ns["get_state"]
        self._rpc_send_ee = ns["send_ee"]
        self._rpc_send_jv = ns["send_jv"]
        self._rpc_stop_ee = ns["stop_ee"]
        self._rpc_stop_jv = ns["stop_jv"]
        self.robot = ns["init_robot"](robot_ip)
        self.use_cartesian = True

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def get_kinematic_state(self) -> KinematicSnapshot:
        q_l, dq_l, p_l, r_l, m_l = self._rpc_state(self.robot)
        q = np.array(q_l, dtype=np.float64)
        dq = np.array(dq_l, dtype=np.float64)
        jacobian = franka_jacobian(q)
        ee_twist = jacobian @ dq
        return (
            q,
            dq,
            jacobian,
            np.array(m_l).reshape(NUM_JOINTS, NUM_JOINTS),
            np.array(p_l, dtype=np.float64),
            np.array(r_l, dtype=np.float64),
            ee_twist,
        )

    def send_cartesian_velocity(self, twist: list[float]) -> None:
        try:
            self._rpc_send_ee(self.robot, tuple(float(v) for v in twist))
        except Exception as e:
            logger.warning("send_cartesian_velocity: %s", e)

    def send_joint_velocity(self, vel: list[float]) -> None:
        try:
            self._rpc_send_jv(self.robot, tuple(float(v) for v in vel))
        except Exception as e:
            logger.warning("send_joint_velocity: %s", e)

    def stop(self) -> None:
        if self.use_cartesian:
            self._rpc_stop_ee(self.robot)
        else:
            self._rpc_stop_jv(self.robot)

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
    def __init__(self):
        self.drivers: dict[str, RobotDriver] = {}
        self._pool = ThreadPoolExecutor(max_workers=4)

    def add_robot(
        self, name: str, server_ip: str, robot_ip: str, port: int, *, use_cartesian: bool = True,
    ) -> None:
        if name in self.drivers:
            raise ValueError(f"Robot '{name}' already connected")
        driver = RobotDriver(server_ip, robot_ip, port)
        driver.use_cartesian = use_cartesian
        self.drivers[name] = driver

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

    def move_cartesian_velocity_batch(self, twists: dict[str, list]) -> None:
        self._gather(lambda n: self.drivers[n].send_cartesian_velocity(twists[n]), list(twists))

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
