"""Direct-RPyC pylibfranka driver for the bimanual plugin.

Replaces the old franky-based driver. The 1 kHz torque control loop
(readOnce/writeOnce on pylibfranka's ActiveControlBase) must run inside the
server process that owns the Robot -- see pylibfranka_server.py -- so each
per-loop op here is one RPyC round-trip per arm, same shape as the old
franky driver, just calling `tick(tau)` instead of `send_jv`/`send_ee`.

Because BimanualFranka's OSC torque law needs the mass matrix, Coriolis
vector, and Jacobian to all come from the *same* RobotState snapshot (see
osc_torque_controller.py's module docstring), get_kinematic_state() now
returns those alongside q/dq/pose/twist in one bundle, and send_torque()
folds the *previous* tick's read together with *this* tick's torque write
into a single RPyC call (mirroring pylibfranka's own readOnce-then-writeOnce
control loop shape).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rpyc
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

NUM_JOINTS = 7
DEFAULT_REQUEST_TIMEOUT_S = 5.0
RPYC_TIMEOUT_S = 10
_CONNECT_TIMEOUT_S = 10.0

# (q, dq, jacobian, ee_pos, ee_rot_xyzw, ee_twist)
# Kept as the public shape consumed by bimanual_franka.py's KinematicSnapshot
# unpacking (`q, dq, J, ee_pos, ee_quat_xyzw, ee_twist = snap`); the extra
# dynamics terms needed for torque control (mass matrix, Coriolis) are
# carried on a separate DynamicsSnapshot returned alongside it, since they
# weren't part of the original interface and existing non-torque call sites
# (e.g. _patch_jacobian) don't need them.
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
# (mass_matrix (7,7), coriolis (7,))
DynamicsSnapshot = tuple[NDArray, NDArray]


def _unpack_bundle(bundle: tuple) -> tuple[KinematicSnapshot, DynamicsSnapshot]:
    """Unpack the flat tuple returned by pylibfranka_server._RobotSession._bundle_state."""
    q, dq, J_flat, ee_pos, ee_quat_xyzw, ee_twist, M_flat, C, g = bundle
    J = np.array(J_flat).reshape(6, NUM_JOINTS)
    M = np.array(M_flat).reshape(NUM_JOINTS, NUM_JOINTS)
    snap: KinematicSnapshot = (
        np.array(q),
        np.array(dq),
        J,
        np.array(ee_pos),
        np.array(ee_quat_xyzw),
        np.array(ee_twist),
    )
    dyn: DynamicsSnapshot = (M, np.array(C))
    return snap, dyn


class RobotDriver:
    """One arm: one RPyC connection to the pylibfranka torque server.

    Single-threaded per-instance; the wrapper executor owns serialization.
    """

    def __init__(self, server_ip: str, robot_ip: str, port: int, use_ee_delta: bool = False):
        # use_ee_delta kept for constructor-signature compatibility with the
        # old franky driver's call sites; torque control doesn't have a
        # separate "ee delta" transport mode -- BimanualFranka's OSC layer
        # is what turns ee-delta actions into torques now.
        self.use_ee_delta = use_ee_delta
        self.recovery_count = 0
        self.robot_ip = robot_ip

        self._conn = rpyc.connect(
            server_ip, port,
            config={"sync_request_timeout": RPYC_TIMEOUT_S, "allow_pickle": True},
        )
        self._conn.root.init_robot(robot_ip, True)
        self._conn.root.start_torque_control(robot_ip)

        self._last_dyn: DynamicsSnapshot | None = None

    @property
    def is_alive(self) -> bool:
        return not self._conn.closed

    def get_kinematic_state(self) -> KinematicSnapshot:
        """Read-only tick (writes no torque). Also caches the dynamics terms
        from this snapshot so a subsequent send_torque() in the same control
        step reuses a consistent (M, C) rather than issuing an extra RPC."""
        bundle, err = self._conn.root.tick(self.robot_ip, None)
        if err is not None:
            self._handle_error(err)
            raise ConnectionError(f"pylibfranka tick() failed for {self.robot_ip}: {err}")
        snap, dyn = _unpack_bundle(bundle)
        self._last_dyn = dyn
        return snap

    def dynamics_from_last_read(self) -> DynamicsSnapshot:
        """(mass_matrix, coriolis) from the most recent get_kinematic_state()
        call on this arm. Raises if none has been taken yet this step."""
        if self._last_dyn is None:
            raise RuntimeError(
                "dynamics_from_last_read() called before get_kinematic_state(); "
                "torque control needs a state read this tick first."
            )
        return self._last_dyn

    def send_torque(self, tau: list[float]) -> KinematicSnapshot:
        """Write a torque command and return the resulting fresh state
        (i.e. this doubles as next tick's read, same as pylibfranka's own
        readOnce/writeOnce loop shape)."""
        bundle, err = self._conn.root.tick(self.robot_ip, list(tau))
        if err is not None:
            self._handle_error(err)
            raise ConnectionError(f"pylibfranka tick() failed for {self.robot_ip}: {err}")
        snap, dyn = _unpack_bundle(bundle)
        self._last_dyn = dyn
        return snap

    def _handle_error(self, err: str) -> None:
        self.recovery_count += 1
        logger.warning("RobotDriver(%s): %s", self.robot_ip, err)
        try:
            # Server already attempted automatic_error_recovery() and torn
            # down its ActiveControlBase; re-arm torque control here so the
            # next tick() call succeeds instead of failing again.
            self._conn.root.start_torque_control(self.robot_ip)
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self._conn.root.stop(self.robot_ip)
        except Exception as e:
            logger.warning("stop(): %s", e)

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

    def recovery_counts(self) -> dict[str, int]:
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

    def current_dynamics_batch(self, names: list[str]) -> dict[str, DynamicsSnapshot]:
        """(mass_matrix, coriolis) per arm, from each arm's most recent read."""
        return {n: self.drivers[n].dynamics_from_last_read() for n in names}

    def move_joint_torque_batch(self, taus: dict[str, list], asynchronous: bool = True) -> dict[str, KinematicSnapshot]:
        """Send torques for this control step; returns the fresh state read
        back with each write (see RobotDriver.send_torque)."""
        return self._gather(lambda n: self.drivers[n].send_torque(taus[n]), list(taus))

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