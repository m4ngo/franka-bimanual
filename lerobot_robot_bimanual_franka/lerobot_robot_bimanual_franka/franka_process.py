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

import franka_config as fc
import numpy as np
import rpyc
from numpy.typing import NDArray

from .franka_jacobian import zero_jacobian

logger = logging.getLogger(__name__)

# Wire-level constants, all from config/control.yaml (franka: section).
NUM_JOINTS = fc.num_joints()

DEFAULT_REQUEST_TIMEOUT_S = fc.control("franka.request_timeout_s")
RPYC_TIMEOUT_S = fc.control("franka.rpyc_timeout_s")
FIRST_STATE_TIMEOUT_S = fc.control("franka.first_state_timeout_s")

# Server-side control modes (pylibfranka_server.MODE_*).
MODE_FLOAT = "float"
MODE_HOLD = "hold"

# (q, dq, jacobian, ee_pos, ee_rot_xyzw, ee_twist)
KinematicSnapshot = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]


# (tau_commanded, tau_measured, tau_external_estimate), each (7,)
TorqueSnapshot = tuple[NDArray, NDArray, NDArray]

_ZERO_TAU = (np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS))


def _unpack(bundle: tuple) -> tuple[KinematicSnapshot, int, TorqueSnapshot]:
    """Flat 49-float bundle: q(7) dq(7) pos(3) quat(4) twist(6) rec(1) tau x3(21)."""
    b = np.asarray(bundle, dtype=np.float64)
    q, pos = b[0:7], b[14:17]
    # J is not on the wire; rebuild it from the same q/ee_pos the server used.
    snap: KinematicSnapshot = (
        q, b[7:14], zero_jacobian(q, ee_pos_base=pos), pos, b[17:21], b[21:27],
    )
    return snap, int(b[27]), (b[28:35], b[35:42], b[42:49])


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
        # Refreshed on every state read; see TorqueSnapshot.
        self.last_torques: TorqueSnapshot = _ZERO_TAU

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
        bundle, err, seq = self._root.get_state(self.robot_ip)
        if bundle is None:
            # Wait past the seq we just read, not past -1: the counter starts at
            # 0, so `_seq > -1` is already true and wait_next would return the
            # empty bundle immediately instead of blocking for the first publish.
            bundle, err, _, ok = self._root.wait_next(self.robot_ip, seq, FIRST_STATE_TIMEOUT_S)
            if not ok or bundle is None:
                raise ConnectionError(
                    f"no state from {self.robot_ip} after {FIRST_STATE_TIMEOUT_S}s: {err}"
                )
        if err is not None:
            logger.warning("RobotDriver(%s): %s", self.robot_ip, err)
        snap, self.recovery_count, self.last_torques = _unpack(bundle)
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
        # All-zero is the goal block's "no reference supplied" encoding -- the loop
        # tests for it and keeps the reference it already has. It cannot collide with
        # a real posture: q=0 is outside joint 4's range.
        ns = np.zeros(NUM_JOINTS) if nullspace_q is None else np.asarray(nullspace_q, dtype=np.float64)
        packed = np.concatenate([
            np.asarray(goal_pos, dtype=np.float64).ravel(),
            np.asarray(goal_quat_xyzw, dtype=np.float64).ravel(),
            np.asarray(kp, dtype=np.float64).ravel(),
            np.asarray(kd, dtype=np.float64).ravel(),
            ns.ravel(),
        ])
        self._push(self._root.set_osc_goal_flat, tuple(float(x) for x in packed))

    def send_joint_goal(
        self, goal_q: np.ndarray, kp: float | None = None, damping_ratio: float | None = None
    ) -> None:
        self._push(self._root.set_joint_goal, _t(goal_q), kp, damping_ratio)

    def send_joint_velocity(self, vel: list[float], kd_scale: float = 1.0) -> None:
        """Joint-velocity setpoint, tracked by a server-side impedance law."""
        self._push(self._root.set_joint_velocity_goal, _t(vel), float(kd_scale))

    def send_torque_goal(self, tau_ff: np.ndarray, joint: int, hold_q: np.ndarray,
                         window_rad: float, dq_max: float = 0.0) -> None:
        """Open-loop torque on ONE joint; the rest impedance-hold at `hold_q`.

        Friction identification only. `window_rad` (travel) and `dq_max` (speed) are
        both enforced by the CONTROL LOOP -- not advisory, and not enforceable from
        here: 2 Nm on joint 7 is ~200 rad/s^2, a quarter radian inside one round trip
        of this call.

        PASS dq_max. The travel window alone does not protect the wrist: joint 7
        passes its 2.61 rad/s datasheet limit after ~0.03 rad, so the robot faults on
        joint_velocity_violation while a 0.25 rad window is still waiting. 0.0 means
        unset and disables the speed check.

        Re-sending bumps the command sequence, which re-arms a tripped window and
        re-latches its origin, so a caller that keeps pushing the same torque after
        a trip will keep re-releasing the joint. Read S_TORQUE_TRIP and stop.

        RAISES, like set_tuning and unlike the other goal pushes. Those are per-tick
        and best-effort because the next one corrects a dropped one. This one is not:
        it is pushed once per torque step, so a swallowed failure means no torque was
        ever applied and the staircase measures nothing while reporting cleanly.
        """
        try:
            self._root.set_torque_goal(self.robot_ip, _t(tau_ff), int(joint),
                                       _t(hold_q), float(window_rad), float(dq_max))
        except AttributeError as e:
            raise RuntimeError(
                f"set_torque_goal({self.robot_ip}) failed: {e} -- this server predates "
                f"MODE_TORQUE. Run scripts/deploy_nuc_server.sh for this arm; the shm "
                f"goal block also grew, so the server and control child must be "
                f"redeployed together.") from e
        except Exception as e:
            raise RuntimeError(f"set_torque_goal({self.robot_ip}) failed: {e}") from e

    def torque_trips(self) -> int:
        """Times MODE_TORQUE's window latched to hold. A blocking read, so keep it out
        of the ramp's inner loop; check it once per torque step.

        RAISES rather than returning 0 on failure: this is a motion DETECTOR, and a
        silent 0 reads as "the joint never moved" -- which is the one answer that would
        make the caller step the torque up again.
        """
        try:
            return int(self._root.torque_trips(self.robot_ip))
        except Exception as e:
            raise RuntimeError(f"torque_trips({self.robot_ip}) failed: {e} -- if this "
                               f"server predates MODE_TORQUE, run "
                               f"scripts/deploy_nuc_server.sh for this arm") from e

    def sim_clip_ticks(self) -> int:
        """Law ticks on which SIM's ctrlrange clip bound. The rotation-overshoot
        measurement: sim's wrist saturates at +/-12 Nm and the FR3's does not, so
        whether that clip engages here decides whether the roll-off is reproduced.

        RAISES rather than returning 0, for the same reason as torque_trips: a silent
        0 reads as "the clip never engaged", which is precisely the finding this call
        exists to establish, and an older server would fake it perfectly.
        """
        try:
            return int(self._root.sim_clip_ticks(self.robot_ip))
        except Exception as e:
            raise RuntimeError(
                f"sim_clip_ticks({self.robot_ip}) failed: {e} -- if this server "
                f"predates the sim-clip counter, run scripts/deploy_nuc_server.sh") from e

    def supports_torque_mode(self) -> bool:
        """Preflight: does the deployed server know MODE_TORQUE? Checked before the
        arm moves, so a stale NUC costs a message rather than a half-run."""
        try:
            return hasattr(self._root, "set_torque_goal")
        except Exception:
            return False

    def set_mode(self, mode: str) -> None:
        self._push(self._root.set_mode, mode)

    def set_tuning(self, friction_kc: float | np.ndarray | tuple | None = None) -> None:
        """Live server-side tuning. friction_kc=0 restores exact osc.py behaviour.

        Scalar, a per-joint 7-vector, or a 2x7 (per joint per rotation
        direction: [positive-commanded-torque, negative]). Flattened here
        because brine only crosses flat tuples of native floats.

        RAISES on failure, unlike every other call in this class. The goal
        pushes are per-tick and best-effort -- a dropped one is corrected by the
        next. This is session configuration, and server sessions are keyed by
        robot_ip and outlive their client, so a swallowed failure leaves the arm
        on whatever assist an earlier script set and the run measures a
        controller nobody configured. That is the exact failure BimanualFranka's
        "ALWAYS push" comment exists to prevent, and routing this through _push
        silently defeated it.
        """
        if friction_kc is not None and np.ndim(friction_kc) > 0:
            friction_kc = _t(friction_kc)
        try:
            self._root.set_tuning(self.robot_ip, friction_kc)
        except Exception as e:
            n = np.size(friction_kc) if friction_kc is not None else 0
            hint = ""
            if "broadcast" in str(e) and n == 2 * NUM_JOINTS:
                hint = (f" -- the server rejected a {n}-element friction_kc, which means "
                        f"it predates the directional assist. Run "
                        f"scripts/deploy_nuc_server.sh for this arm.")
            raise RuntimeError(
                f"set_tuning({self.robot_ip}) failed: {e}{hint}") from e

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

    def sim_clip_ticks(self, name: str) -> int:
        """Law ticks on which SIM's ctrlrange clip bound on that arm."""
        return self.drivers[name].sim_clip_ticks()

    def torque_snapshot(self, name: str) -> TorqueSnapshot:
        """(commanded, measured, external estimate) from that arm's last state read.

        commanded is post clamp and rate limit -- what was actually written to
        libfranka, not what the control law produced before limiting.
        """
        return self.drivers[name].last_torques

    def torque_snapshot_batch(self, names: list[str]) -> dict[str, TorqueSnapshot]:
        return {n: self.drivers[n].last_torques for n in names}

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

    def move_joint_velocity_batch(self, vels: dict[str, list], asynchronous: bool = True,
                                  kd_scale: float = 1.0) -> None:
        self._gather(lambda n: self.drivers[n].send_joint_velocity(vels[n], kd_scale), list(vels))

    def move_torque_goal(self, name: str, tau_ff, joint: int, hold_q, window_rad: float,
                         dq_max: float = 0.0) -> None:
        """Single-arm by design: MODE_TORQUE is a one-joint identification mode, and
        there is no meaning to running it on both arms at once."""
        self.drivers[name].send_torque_goal(tau_ff, joint, hold_q, window_rad, dq_max)

    def torque_trips(self, name: str) -> int:
        return self.drivers[name].torque_trips()

    def supports_torque_mode(self, name: str) -> bool:
        return self.drivers[name].supports_torque_mode()

    def set_mode_all(self, mode: str) -> None:
        self._gather(lambda n: self.drivers[n].set_mode(mode), list(self.drivers))

    def set_tuning_all(self, friction_kc: float | np.ndarray | tuple | None = None) -> None:
        self._gather(lambda n: self.drivers[n].set_tuning(friction_kc), list(self.drivers))

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
