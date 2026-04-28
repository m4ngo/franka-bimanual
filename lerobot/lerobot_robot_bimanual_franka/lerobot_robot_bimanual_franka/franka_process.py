"""Subprocess-based Franka robot driver used by the bimanual plugin.

Each Franka arm runs in its own process (RobotProcess) and communicates with
the parent via two multiprocessing queues. MultiRobotWrapper is the
parent-side facade that dispatches commands to the right subprocess and
gathers their responses.
"""

from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, cast
import signal

import numpy as np
from numpy.typing import NDArray

# ---- Control parameters -----------------------------------------------------
# Duration attached to each streamed velocity command (ms). The control loop in
# the parent process must re-issue commands faster than this or the arm stalls.
# Position tracking / PD shaping happens in the parent (see BimanualFranka) so
# this subprocess can stay a thin pass-through to franky.
VELOCITY_COMMAND_DURATION_MS = 500
VELOCITY = 1.0 # m/s
ACCELERATION = 0.25
JERK = 1.0
TORQUE_THRESHOLD = 100.0 # Nm
FORCE_THRESHOLD = 200.0 # N
JOINT_STIFFNESS = [350.0, 350.0, 300.0, 500.0, 350.0, 150.0, 150.0]

# ---- Dimensions -------------------------------------------------------------
NUM_JOINTS = 7
EE_DELTA_DIMS = 6  # linear(3) + angular(3)

# ---- Timeouts ---------------------------------------------------------------
DEFAULT_REQUEST_TIMEOUT_S = 5.0
SHUTDOWN_STOP_TIMEOUT_S = 2.0
SHUTDOWN_JOIN_TIMEOUT_S = 5.0
TERMINATE_JOIN_TIMEOUT_S = 1.0

# Robot errors that are recoverable via ``recover_from_errors``.
_RECOVERABLE_ERRORS = (
    "UDP receive: Timeout",
    "communication_constrains_violation"
    )


def _validate_vector(name: str, values, expected_len: int) -> list[float]:
    """Validate that *values* is a numeric sequence of the requested length."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"{name} must be a list/tuple of length {expected_len}, got {type(values).__name__}"
        )
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
    ):
        self.server_ip = server_ip
        self.robot_ip = robot_ip
        self.port = port
        self.command_queue = command_queue
        self.response_queue = response_queue

    def run(self):
        """Main loop: initialise robot, then process commands until shutdown."""
        try:
            # The parent handles Ctrl+C and issues explicit shutdown commands.
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
                RelativeDynamicsFactor
            )

            robot = Robot(self.robot_ip)
            robot.recover_from_errors()
            robot.relative_dynamics_factor = RelativeDynamicsFactor(
                velocity=VELOCITY,
                acceleration=ACCELERATION,
                jerk=JERK
            )
            # robot.joint_velocity_limit.set([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
            # robot.joint_acceleration_limit.set([15.0, 7.5, 10.0, 12.5, 15.0, 15.0, 15.0])
            # robot.joint_jerk_limit.set([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])
            robot.set_collision_behavior(TORQUE_THRESHOLD, FORCE_THRESHOLD)  
            robot.set_joint_impedance(JOINT_STIFFNESS)
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
                    motion = JointVelocityMotion(
                        cast(Any, velocity), Duration(VELOCITY_COMMAND_DURATION_MS)
                    )
                    result = robot.move(motion, asynchronous=args[1])
                    self.response_queue.put(("success", result))

                elif command == "move_ee_delta":
                    delta = _validate_vector("move_ee_delta position", args[0], EE_DELTA_DIMS)
                    linear: NDArray[np.float64] = np.asarray(delta[:3], dtype=np.float64)
                    angular: NDArray[np.float64] = np.asarray(delta[3:], dtype=np.float64)
                    motion = CartesianVelocityMotion(
                        Twist(cast(Any, linear), cast(Any, angular)),
                        Duration(VELOCITY_COMMAND_DURATION_MS),
                    )
                    result = robot.move(motion, asynchronous=args[1])
                    self.response_queue.put(("success", result))

                elif command == "get_state":
                    self.response_queue.put(("success", robot.get_last_callback_data()))

                elif command == "current_joint_positions":
                    self.response_queue.put(("success", robot.current_joint_positions))

                elif command == "current_kinematic_state":
                    # All kinematic state needed for one parent-side control
                    # tick (PD + safety overlays), captured from the same
                    # snapshot of robot.state for consistency.
                    state = robot.state
                    q = np.asarray(state.q, dtype=np.float64)
                    dq = np.asarray(state.dq, dtype=np.float64)
                    ee_translation = np.asarray(
                        state.O_T_EE.translation, dtype=np.float64
                    ).flatten()
                    jacobian = np.asarray(
                        robot.model.zero_jacobian(Frame.EndEffector, state),
                        dtype=np.float64,
                    )
                    self.response_queue.put(
                        ("success", (q, dq, ee_translation, jacobian))
                    )

                elif command == "join_motion":
                    timeout = args[0] if args else None
                    self.response_queue.put(("success", robot.join_motion(timeout=timeout)))

                elif command == "stop_motion":
                    zero = np.zeros(NUM_JOINTS, dtype=np.float64)
                    motion = JointVelocityMotion(
                        cast(Any, zero), Duration(VELOCITY_COMMAND_DURATION_MS)
                    )
                    self.response_queue.put(("success", robot.move(motion, asynchronous=False)))

                elif command == "shutdown":
                    try:
                        zero = np.zeros(NUM_JOINTS, dtype=np.float64)
                        robot.move(
                            JointVelocityMotion(
                                cast(Any, zero), Duration(VELOCITY_COMMAND_DURATION_MS)
                            ),
                            asynchronous=False,
                        )
                    except Exception:
                        # A failed final-stop should not block shutdown.
                        pass
                    break

                else:
                    self.response_queue.put(("error", f"Unknown command: {command}"))

            except Exception as e:
                error_text = str(e)
                # Attempt in-process recovery for known transient faults.
                if any(token in error_text for token in _RECOVERABLE_ERRORS):
                    try:
                        robot.recover_from_errors()
                    except Exception:
                        # Keep surfacing the original exception even if recovery fails.
                        pass
                self.response_queue.put(("error", error_text))


class MultiRobotWrapper:
    """Parent-side manager that dispatches commands to per-arm subprocesses."""

    def __init__(self):
        # robot_name -> {"command_queue": Queue, "response_queue": Queue}
        self.robots: dict[str, dict[str, Queue]] = {}
        self.processes: dict[str, Process] = {}

    def add_robot(self, name: str, server_ip: str, robot_ip: str, port: int) -> None:
        """Spawn a subprocess for a new robot connection."""
        if name in self.processes and self.processes[name].is_alive():
            raise ValueError(f"Robot '{name}' is already connected")

        command_queue: Queue = Queue()
        response_queue: Queue = Queue()

        worker = RobotProcess(server_ip, robot_ip, port, command_queue, response_queue)
        process = Process(target=worker.run, daemon=True)
        process.start()

        self.robots[name] = {"command_queue": command_queue, "response_queue": response_queue}
        self.processes[name] = process

    @property
    def num_processes(self) -> int:
        """Count of subprocesses that are currently alive."""
        return sum(1 for p in self.processes.values() if p.is_alive())

    # --- Request plumbing ----------------------------------------------------

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

    def _request(
        self,
        robot_name: str,
        command: str,
        args: list,
        kwargs: dict,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ):
        self._enqueue(robot_name, command, args, kwargs)
        return self._collect(robot_name, command, timeout_s)

    def _request_many(
        self,
        requests: list[tuple[str, str, list, dict]],
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> dict[str, Any]:
        # Enqueue every command first so subprocesses run in parallel, then
        # harvest the responses.
        for robot_name, command, args, kwargs in requests:
            self._enqueue(robot_name, command, args, kwargs)
        return {
            robot_name: self._collect(robot_name, command, timeout_s)
            for robot_name, command, _, _ in requests
        }

    # --- Motion commands -----------------------------------------------------

    def move_joint_velocity(
        self, robot_name: str, velocity: list, asynchronous: bool = False
    ):
        """Stream a 7-DoF joint velocity (rad/s) to a single arm.

        The subprocess forwards the velocity to franky as-is; PD shaping and
        safety clamping live in the parent process.
        """
        velocity = _validate_vector("move_joint_velocity", velocity, NUM_JOINTS)
        return self._request(
            robot_name, "move_joint_velocity", [velocity, asynchronous], {}
        )

    def move_ee_delta(self, robot_name: str, position: list, asynchronous: bool = False):
        """Command an end-effector twist (linear + angular) for a single arm."""
        position = _validate_vector("move_ee_delta position", position, EE_DELTA_DIMS)
        return self._request(robot_name, "move_ee_delta", [position, asynchronous], {})

    def move_joint_velocity_batch(
        self, velocities_by_robot: dict[str, list], asynchronous: bool = False
    ) -> dict[str, Any]:
        """Send joint velocities (rad/s) to several arms in parallel."""
        requests = [
            (
                name,
                "move_joint_velocity",
                [_validate_vector("move_joint_velocity", vel, NUM_JOINTS), asynchronous],
                {},
            )
            for name, vel in velocities_by_robot.items()
        ]
        return self._request_many(requests)

    def move_ee_delta_batch(
        self, positions_by_robot: dict[str, list], asynchronous: bool = False
    ) -> dict[str, Any]:
        """Send EE twists to several arms in parallel."""
        requests = [
            (
                name,
                "move_ee_delta",
                [_validate_vector("move_ee_delta position", pos, EE_DELTA_DIMS), asynchronous],
                {},
            )
            for name, pos in positions_by_robot.items()
        ]
        return self._request_many(requests)

    def stop_motion(self, robot_name: str, timeout_s: float = SHUTDOWN_STOP_TIMEOUT_S):
        return self._request(robot_name, "stop_motion", [], {}, timeout_s=timeout_s)

    def stop_all_motion(self, timeout_s: float = SHUTDOWN_STOP_TIMEOUT_S) -> dict[str, Any]:
        """Issue stop_motion to every live arm."""
        requests = [
            (name, "stop_motion", [], {})
            for name, process in self.processes.items()
            if process.is_alive() and name in self.robots
        ]
        return self._request_many(requests, timeout_s=timeout_s) if requests else {}

    # --- State queries -------------------------------------------------------

    def get_robot_state(self, robot_name: str):
        return self._request(robot_name, "get_state", [], {})

    def current_joint_positions(
        self, robot_name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ):
        return self._request(robot_name, "current_joint_positions", [], {}, timeout_s=timeout_s)

    def current_kinematic_state(
        self, robot_name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Return (q, dq, ee_translation, jacobian) for one arm.

        - q, dq: 7-vector joint positions and velocities (rad, rad/s).
        - ee_translation: 3-vector EE position in the arm's base frame (m).
        - jacobian: 6x7 base-frame Jacobian; rows [0:3] linear, [3:6] angular.

        All four arrays are read from the same robot.state snapshot so they
        are mutually consistent for one parent-side control tick.
        """
        return self._request(
            robot_name, "current_kinematic_state", [], {}, timeout_s=timeout_s
        )

    def current_kinematic_state_batch(
        self,
        robot_names: list[str],
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> dict[
        str,
        tuple[
            NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
        ],
    ]:
        """Query (q, dq, ee_translation, jacobian) for several arms in parallel."""
        requests = [(name, "current_kinematic_state", [], {}) for name in robot_names]
        return self._request_many(requests, timeout_s=timeout_s)

    def join_motion(self, robot_name: str, timeout: float = 0.0):
        # Grant extra budget over the in-robot timeout to allow round-trip.
        return self._request(
            robot_name, "join_motion", [timeout], {}, timeout_s=max(DEFAULT_REQUEST_TIMEOUT_S, timeout + 1.0)
        )

    # --- Lifecycle -----------------------------------------------------------

    def shutdown(self) -> None:
        """Stop motion, signal shutdown, then join (and terminate) workers."""
        try:
            self.stop_all_motion(timeout_s=SHUTDOWN_STOP_TIMEOUT_S)
        except Exception:
            # Continue shutdown even if stopping one robot fails.
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
