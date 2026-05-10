"""Subprocess-based Franka robot driver for the bimanual plugin.

Each arm runs in its own RobotProcess. MultiRobotWrapper is the parent-side
facade that dispatches commands and collects responses.
"""

import logging
import threading
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, cast
import signal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Duration (ms) attached to each streamed velocity command. Parent must re-issue faster than this.
VELOCITY_COMMAND_DURATION_MS = 250

# Recompute Jacobian only when joints move more than this (L-inf, rad) from the cached config.
_JACOBIAN_CACHE_Q_THRESHOLD = 0.50

JOINT_RELATIVE_DYNAMICS = (1.0, 0.25, 1.0)
TORQUE_THRESHOLD = 100.0  # Nm
FORCE_THRESHOLD = 200.0   # N
JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

EE_DELTA_RELATIVE_DYNAMICS = (0.4, 0.25, 0.15)

NUM_JOINTS = 7
EE_DELTA_DIMS = 6  # linear(3) + angular(3)

# (q, dq, jacobian, ee_pos, ee_rot_xyzw, ee_twist) snapshot from one robot.state read.
KinematicSnapshot = tuple[
    NDArray[np.float64], # joint pos
    NDArray[np.float64], # joint velocities
    NDArray[np.float64], # jacobian
    NDArray[np.float64], # ee pos
    NDArray[np.float64], # ee rot
    NDArray[np.float64], # ee twist (velocity)
]

DEFAULT_REQUEST_TIMEOUT_S = 5.0
SHUTDOWN_STOP_TIMEOUT_S = 2.0
SHUTDOWN_JOIN_TIMEOUT_S = 5.0
TERMINATE_JOIN_TIMEOUT_S = 1.0

_RECOVERABLE_ERRORS = (
    "UDP receive: Timeout",
    "communication_constrains_violation",
    'current mode ("Reflex")',
    "type of motion cannot change",
)


def _validate_vector(name: str, values, expected_len: int) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple of length {expected_len}, got {type(values).__name__}")
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}")
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must contain only numeric values") from e


class RobotProcess:
    """Worker that owns a single Franka connection and executes queued commands."""

    def __init__(
        self,
        server_ip: str,
        robot_ip: str,
        port: int,
        command_queue: Queue,
        response_queue: Queue,
        use_ee_delta: bool = False,
    ):
        self.server_ip = server_ip
        self.robot_ip = robot_ip
        self.port = port
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.use_ee_delta = use_ee_delta

    def run(self):
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            from net_franky import setup_net_franky
            setup_net_franky(self.server_ip, self.port)

            from net_franky.franky import (
                CartesianVelocityMotion,
                Duration,
                Frame,
                JointVelocityMotion,
                Robot,
                Twist,
                RelativeDynamicsFactor,
            )

            robot = Robot(self.robot_ip)
            robot.recover_from_errors()
            robot.relative_dynamics_factor = RelativeDynamicsFactor(*JOINT_RELATIVE_DYNAMICS)
            robot.set_collision_behavior(TORQUE_THRESHOLD, FORCE_THRESHOLD)
            robot.set_joint_impedance(JOINT_STIFFNESS)

            ee_dynamics = RelativeDynamicsFactor(*EE_DELTA_RELATIVE_DYNAMICS)
            zero_lin = np.zeros(3, dtype=np.float64)
            zero_joint = np.zeros(NUM_JOINTS, dtype=np.float64)

            _cached_jacobian: NDArray[np.float64] | None = None
            _cached_jacobian_q: NDArray[np.float64] | None = None
            # get_last_callback_data() raises AttributeError before any motion starts (state is None).
            # Only use the fast callback path after the robot has received at least one move command.
            _motion_started = False

            def make_prime_motion():
                if self.use_ee_delta:
                    return CartesianVelocityMotion(
                        Twist(cast(Any, zero_lin), cast(Any, zero_lin)),
                        Duration(VELOCITY_COMMAND_DURATION_MS),
                        ee_dynamics,
                    )
                return JointVelocityMotion(cast(Any, zero_joint), Duration(VELOCITY_COMMAND_DURATION_MS))

        except Exception as e:
            self.response_queue.put(("error", f"Failed to initialize robot: {e}"))
            return

        while True:
            try:
                command, args, kwargs = self.command_queue.get()

                if command == "move_joint_velocity":
                    velocity = np.asarray(
                        _validate_vector("move_joint_velocity", args[0], NUM_JOINTS),
                        dtype=np.float64,
                    )
                    motion = JointVelocityMotion(cast(Any, velocity), Duration(VELOCITY_COMMAND_DURATION_MS))
                    result = robot.move(motion, asynchronous=args[1])
                    _motion_started = True
                    self.response_queue.put(("success", result))

                elif command == "move_ee_delta":
                    delta = _validate_vector("move_ee_delta position", args[0], EE_DELTA_DIMS)
                    linear: NDArray[np.float64] = np.asarray(delta[:3], dtype=np.float64)
                    angular: NDArray[np.float64] = np.asarray(delta[3:], dtype=np.float64)
                    motion = CartesianVelocityMotion(
                        Twist(cast(Any, linear), cast(Any, angular)),
                        Duration(VELOCITY_COMMAND_DURATION_MS),
                        ee_dynamics,
                    )
                    result = robot.move(motion, asynchronous=args[1])
                    _motion_started = True
                    self.response_queue.put(("success", result))

                elif command == "current_kinematic_state":
                    # Prefer callback-buffered state (1 kHz, no RPC) over robot.state,
                    # but only after motion has started — before that the callback state is None.
                    _cb_state = None
                    if _motion_started:
                        try:
                            _cb_state, _, _, _, _ = robot.get_last_callback_data()
                        except Exception:
                            pass
                    state = _cb_state if _cb_state is not None else robot.state

                    q = np.asarray(state.q, dtype=np.float64)
                    dq = np.asarray(state.dq, dtype=np.float64)
                    ee_pos = np.asarray(state.O_T_EE.translation, dtype=np.float64).flatten()
                    ee_rot = np.asarray(state.O_T_EE.quaternion, dtype=np.float64).flatten()
                    ee_vel = np.concatenate([state.O_dP_EE_c.linear, state.O_dP_EE_c.angular])

                    if (
                        _cached_jacobian is None
                        or _cached_jacobian_q is None
                        or float(np.max(np.abs(q - _cached_jacobian_q))) > _JACOBIAN_CACHE_Q_THRESHOLD
                    ):
                        try:
                            _raw_j = robot.model.zero_jacobian(Frame.EndEffector, state)
                        except Exception:
                            _raw_j = robot.model.zero_jacobian(Frame.EndEffector, robot.state)
                        _cached_jacobian = np.asarray(_raw_j, dtype=np.float64)
                        _cached_jacobian_q = q.copy()

                    self.response_queue.put(("success", (q, dq, _cached_jacobian, ee_pos, ee_rot, ee_vel)))

                elif command == "move_joint_velocity_async":
                    _vel = np.asarray(args[0], dtype=np.float64)
                    _motion = JointVelocityMotion(cast(Any, _vel), Duration(VELOCITY_COMMAND_DURATION_MS))
                    _motion_started = True

                    def _run_jv(_m=_motion):
                        try:
                            robot.move(_m, asynchronous=True)
                        except Exception as _e:
                            if any(tok in str(_e) for tok in _RECOVERABLE_ERRORS):
                                try:
                                    robot.recover_from_errors()
                                except Exception:
                                    pass
                            logger.warning("move_joint_velocity_async error: %s", _e)

                    threading.Thread(target=_run_jv, daemon=True).start()

                elif command == "move_ee_delta_async":
                    _d = args[0]
                    _motion = CartesianVelocityMotion(
                        Twist(
                            cast(Any, np.asarray(_d[:3], dtype=np.float64)),
                            cast(Any, np.asarray(_d[3:], dtype=np.float64)),
                        ),
                        Duration(VELOCITY_COMMAND_DURATION_MS),
                        ee_dynamics,
                    )
                    _motion_started = True

                    def _run_ee(_m=_motion):
                        try:
                            robot.move(_m, asynchronous=True)
                        except Exception as _e:
                            if any(tok in str(_e) for tok in _RECOVERABLE_ERRORS):
                                try:
                                    robot.recover_from_errors()
                                except Exception:
                                    pass
                            logger.warning("move_ee_delta_async error: %s", _e)

                    threading.Thread(target=_run_ee, daemon=True).start()

                elif command == "stop_motion":
                    self.response_queue.put(
                        ("success", robot.move(make_prime_motion(), asynchronous=False))
                    )

                elif command == "shutdown":
                    try:
                        robot.move(make_prime_motion(), asynchronous=False)
                    except Exception:
                        pass
                    break

                else:
                    self.response_queue.put(("error", f"Unknown command: {command}"))

            except Exception as e:
                error_text = str(e)
                if any(token in error_text for token in _RECOVERABLE_ERRORS):
                    try:
                        robot.recover_from_errors()
                    except Exception:
                        pass
                self.response_queue.put(("error", error_text))


class MultiRobotWrapper:
    """Parent-side manager that dispatches commands to per-arm subprocesses."""

    def __init__(self):
        self.robots: dict[str, dict[str, Queue]] = {}
        self.processes: dict[str, Process] = {}

    def add_robot(
        self,
        name: str,
        server_ip: str,
        robot_ip: str,
        port: int,
        use_ee_delta: bool = False,
    ) -> None:
        if name in self.processes and self.processes[name].is_alive():
            raise ValueError(f"Robot '{name}' is already connected")

        command_queue: Queue = Queue()
        response_queue: Queue = Queue()
        worker = RobotProcess(server_ip, robot_ip, port, command_queue, response_queue, use_ee_delta)
        process = Process(target=worker.run, daemon=True)
        process.start()

        self.robots[name] = {"command_queue": command_queue, "response_queue": response_queue}
        self.processes[name] = process

    @property
    def num_processes(self) -> int:
        return sum(1 for p in self.processes.values() if p.is_alive())

    def _enqueue(self, robot_name: str, command: str, args: list, kwargs: dict) -> None:
        if robot_name not in self.robots:
            raise KeyError(f"Robot '{robot_name}' is not registered")
        process = self.processes.get(robot_name)
        if process is None or not process.is_alive():
            raise RuntimeError(f"Robot process '{robot_name}' is not alive")
        self.robots[robot_name]["command_queue"].put((command, args, kwargs))

    def _collect(self, robot_name: str, command: str, timeout_s: float):
        process = self.processes.get(robot_name)
        if process is None:
            raise RuntimeError(f"Robot process '{robot_name}' is not available")
        try:
            status, result = self.robots[robot_name]["response_queue"].get(timeout=timeout_s)
        except Empty as e:
            raise TimeoutError(
                f"Timed out waiting for '{command}' response from robot '{robot_name}'. "
                f"Process alive: {process.is_alive()}."
            ) from e
        if status == "error":
            raise Exception(result)
        return result

    def _request(self, robot_name: str, command: str, args: list, kwargs: dict, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S):
        self._enqueue(robot_name, command, args, kwargs)
        return self._collect(robot_name, command, timeout_s)

    def _request_many(self, requests: list[tuple[str, str, list, dict]], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> dict[str, Any]:
        for robot_name, command, args, kwargs in requests:
            self._enqueue(robot_name, command, args, kwargs)
        return {
            robot_name: self._collect(robot_name, command, timeout_s)
            for robot_name, command, _, _ in requests
        }

    def move_joint_velocity_batch(self, velocities_by_robot: dict[str, list], asynchronous: bool = False) -> dict[str, Any]:
        """Send joint velocities (rad/s) to several arms in parallel."""
        if asynchronous:
            for name, vel in velocities_by_robot.items():
                self._enqueue(name, "move_joint_velocity_async", [_validate_vector("move_joint_velocity", vel, NUM_JOINTS)], {})
            return {}
        requests = [
            (name, "move_joint_velocity", [_validate_vector("move_joint_velocity", vel, NUM_JOINTS), False], {})
            for name, vel in velocities_by_robot.items()
        ]
        return self._request_many(requests)

    def move_ee_delta_batch(self, positions_by_robot: dict[str, list], asynchronous: bool = False) -> dict[str, Any]:
        """Send EE twists to several arms in parallel."""
        if asynchronous:
            for name, pos in positions_by_robot.items():
                self._enqueue(name, "move_ee_delta_async", [_validate_vector("move_ee_delta position", pos, EE_DELTA_DIMS)], {})
            return {}
        requests = [
            (name, "move_ee_delta", [_validate_vector("move_ee_delta position", pos, EE_DELTA_DIMS), False], {})
            for name, pos in positions_by_robot.items()
        ]
        return self._request_many(requests)

    def stop_all_motion(self, timeout_s: float = SHUTDOWN_STOP_TIMEOUT_S) -> dict[str, Any]:
        requests = [
            (name, "stop_motion", [], {})
            for name, process in self.processes.items()
            if process.is_alive() and name in self.robots
        ]
        return self._request_many(requests, timeout_s=timeout_s) if requests else {}

    def current_kinematic_state(self, robot_name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> KinematicSnapshot:
        return self._request(robot_name, "current_kinematic_state", [], {}, timeout_s=timeout_s)

    def current_kinematic_state_batch(self, robot_names: list[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> dict[str, KinematicSnapshot]:
        requests = [(name, "current_kinematic_state", [], {}) for name in robot_names]
        return self._request_many(requests, timeout_s=timeout_s)

    def shutdown(self) -> None:
        try:
            self.stop_all_motion(timeout_s=SHUTDOWN_STOP_TIMEOUT_S)
        except Exception:
            pass

        for queues in self.robots.values():
            queues["command_queue"].put(("shutdown", [], {}))

        for process in self.processes.values():
            process.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)

        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=TERMINATE_JOIN_TIMEOUT_S)

        self.robots.clear()
        self.processes.clear()
