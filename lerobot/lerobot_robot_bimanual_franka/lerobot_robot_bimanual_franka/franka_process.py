from multiprocessing import Queue, Process  
from typing import Any, cast  
from queue import Empty
import signal

import numpy as np
from numpy.typing import NDArray


# PD joint-velocity controller settings for smooth position tracking.
JOINT_POSITION_DEADBAND_RAD = 0.00
JOINT_PD_VELOCITY_MAX = 1.0
JOINT_PD_KP = 1.5
JOINT_PD_KD = 0.2
VELOCITY_COMMAND_DURATION_MS = 100


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
    def __init__(self, server_ip: str, robot_ip: str, port: int, command_queue: Queue, response_queue: Queue):  
        self.server_ip = server_ip  
        self.robot_ip = robot_ip  
        self.port = port
        self.command_queue = command_queue  
        self.response_queue = response_queue  
          
    def run(self):
        try:
            # Parent handles Ctrl+C and sends explicit shutdown commands.
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            from net_franky import setup_net_franky  
            setup_net_franky(self.server_ip, self.port)  
              
            from net_franky.franky import Robot, JointVelocityMotion, CartesianVelocityMotion, Twist, Duration 
              
            robot = Robot(self.robot_ip)  
            robot.recover_from_errors()  
            robot.relative_dynamics_factor = 0.2
        except Exception as e:
            self.response_queue.put(("error", f"Failed to initialize robot: {e}"))
            return  
          
        while True:  
            try:  
                command, args, kwargs = self.command_queue.get()  
                  
                if command == "move_joints":  
                    joint_target = _validate_vector("move_joints position", args[0], 7)
                    target: NDArray[np.float64] = np.asarray(joint_target, dtype=np.float64)

                    current_joint_positions: NDArray[np.float64] = np.asarray(robot.current_joint_positions, dtype=np.float64)
                    current_joint_velocities: NDArray[np.float64] = np.asarray(robot.current_joint_velocities, dtype=np.float64)

                    joint_error = target - current_joint_positions
                    if float(np.max(np.abs(joint_error))) <= JOINT_POSITION_DEADBAND_RAD:
                        self.response_queue.put(("success", False))
                        continue

                    joint_velocity = (JOINT_PD_KP * joint_error) - (JOINT_PD_KD * current_joint_velocities)

                    norm = np.linalg.norm(joint_velocity)
                    if norm > JOINT_PD_VELOCITY_MAX:
                        joint_velocity *= JOINT_PD_VELOCITY_MAX / norm

                    motion = JointVelocityMotion(cast(Any, joint_velocity), Duration(VELOCITY_COMMAND_DURATION_MS))
                    result = robot.move(motion, asynchronous=args[1])
                    self.response_queue.put(("success", result))
                  
                elif command == "move_ee_delta":
                    ee_delta = _validate_vector("move_ee_delta position", args[0], 6)
                    linear: NDArray[np.float64] = np.asarray(ee_delta[:3], dtype=np.float64)
                    angular: NDArray[np.float64] = np.asarray(ee_delta[3:], dtype=np.float64)
                    motion = CartesianVelocityMotion(
                        Twist(cast(Any, linear), cast(Any, angular)),
                        Duration(VELOCITY_COMMAND_DURATION_MS),
                    )
                    result = robot.move(motion, asynchronous=args[1])
                    self.response_queue.put(("success", result))
                      
                elif command == "get_state":  
                    state = robot.get_last_callback_data()
                    self.response_queue.put(("success", state))  

                elif command == "current_joint_positions":
                    joints = robot.current_joint_positions
                    self.response_queue.put(("success", joints))  
                      
                elif command == "join_motion":  
                    result = robot.join_motion(timeout=args[0] if args else None)
                    self.response_queue.put(("success", result))  

                elif command == "stop_motion":
                    zero_velocity: NDArray[np.float64] = np.zeros(7, dtype=np.float64)
                    motion = JointVelocityMotion(cast(Any, zero_velocity), Duration(VELOCITY_COMMAND_DURATION_MS))
                    result = robot.move(motion, asynchronous=False)
                    self.response_queue.put(("success", result))
                      
                elif command == "shutdown":  
                    try:
                        zero_velocity: NDArray[np.float64] = np.zeros(7, dtype=np.float64)
                        robot.move(
                            JointVelocityMotion(cast(Any, zero_velocity), Duration(VELOCITY_COMMAND_DURATION_MS)),
                            asynchronous=False,
                        )
                    except Exception:
                        # Shutdown should continue even if a final stop command fails.
                        pass
                    break  
                      
                else:  
                    self.response_queue.put(("error", f"Unknown command: {command}"))  
                      
            except Exception as e:
                error_text = str(e)
                if "UDP receive: Timeout" in error_text or "communication_constrains_violation" in error_text:
                    try:
                        robot.recover_from_errors()
                    except Exception:
                        # Keep surfacing the original exception even if recovery fails.
                        pass
                self.response_queue.put(("error", error_text))

class MultiRobotWrapper:  
    def __init__(self):  
        self.robots = {}  
        self.processes = {}  
          
    def add_robot(self, name: str, server_ip: str, robot_ip: str, port: int):  
        """Add a robot connection"""  
        if name in self.processes and self.processes[name].is_alive():
            raise ValueError(f"Robot '{name}' is already connected")

        command_queue = Queue()  
        response_queue = Queue()  
          
        robot_proc = RobotProcess(server_ip, robot_ip, port, command_queue, response_queue)  
        process = Process(target=robot_proc.run)
        process.start()  
          
        self.robots[name] = {  
            'command_queue': command_queue,  
            'response_queue': response_queue  
        }  
        self.processes[name] = process  

    @property
    def num_processes(self) -> int:
        return sum(1 for process in self.processes.values() if process.is_alive())

    def _enqueue(self, robot_name: str, command: str, args: list, kwargs: dict):
        if robot_name not in self.robots:
            raise KeyError(f"Robot '{robot_name}' is not registered")

        process = self.processes.get(robot_name)
        if process is None or not process.is_alive():
            raise RuntimeError(f"Robot process '{robot_name}' is not alive")

        queues = self.robots[robot_name]
        queues['command_queue'].put((command, args, kwargs))

    def _collect(self, robot_name: str, command: str, timeout_s: float):
        process = self.processes.get(robot_name)
        if process is None:
            raise RuntimeError(f"Robot process '{robot_name}' is not available")

        queues = self.robots[robot_name]
        try:
            status, result = queues['response_queue'].get(timeout=timeout_s)
        except Empty as e:
            alive = process.is_alive()
            raise TimeoutError(
                f"Timed out waiting for '{command}' response from robot '{robot_name}'. Process alive: {alive}."
            ) from e

        if status == "error":
            raise Exception(result)

        return result

    def _request(self, robot_name: str, command: str, args: list, kwargs: dict, timeout_s: float = 5.0):
        self._enqueue(robot_name, command, args, kwargs)
        return self._collect(robot_name, command, timeout_s)

    def _request_many(
        self,
        requests: list[tuple[str, str, list, dict]],
        timeout_s: float = 5.0,
    ) -> dict[str, Any]:
        for robot_name, command, args, kwargs in requests:
            self._enqueue(robot_name, command, args, kwargs)

        results: dict[str, Any] = {}
        for robot_name, command, _, _ in requests:
            results[robot_name] = self._collect(robot_name, command, timeout_s)

        return results
          
    def move_joints(self, robot_name: str, position: list, asynchronous: bool = False):  
        """Move robot to cartesian position"""  
        position = _validate_vector("move_joints position", position, 7)
        return self._request(robot_name, "move_joints", [position, asynchronous], {}, timeout_s=5.0)
    
    def move_ee_delta(self, robot_name: str, position: list, asynchronous: bool = False):
        """Move robot to cartesian position"""  
        position = _validate_vector("move_ee_delta position", position, 6)
        return self._request(robot_name, "move_ee_delta", [position, asynchronous], {}, timeout_s=5.0)

    def move_joints_batch(self, positions_by_robot: dict[str, list], asynchronous: bool = False):
        requests: list[tuple[str, str, list, dict]] = []
        for robot_name, position in positions_by_robot.items():
            validated_position = _validate_vector("move_joints position", position, 7)
            requests.append((robot_name, "move_joints", [validated_position, asynchronous], {}))
        return self._request_many(requests, timeout_s=5.0)

    def move_ee_delta_batch(self, positions_by_robot: dict[str, list], asynchronous: bool = False):
        requests: list[tuple[str, str, list, dict]] = []
        for robot_name, position in positions_by_robot.items():
            validated_position = _validate_vector("move_ee_delta position", position, 6)
            requests.append((robot_name, "move_ee_delta", [validated_position, asynchronous], {}))
        return self._request_many(requests, timeout_s=5.0)

    def stop_motion(self, robot_name: str, timeout_s: float = 2.0):
        return self._request(robot_name, "stop_motion", [], {}, timeout_s=timeout_s)

    def stop_all_motion(self, timeout_s: float = 2.0) -> dict[str, Any]:
        requests: list[tuple[str, str, list, dict]] = []
        for robot_name, process in self.processes.items():
            if process.is_alive() and robot_name in self.robots:
                requests.append((robot_name, "stop_motion", [], {}))
        if not requests:
            return {}
        return self._request_many(requests, timeout_s=timeout_s)
          
    def get_robot_state(self, robot_name: str):  
        """Get current robot state"""  
        return self._request(robot_name, "get_state", [], {}, timeout_s=5.0)
    
    def current_joint_positions(self, robot_name: str, timeout_s: float = 5.0):
        """Get current joint positions"""  
        return self._request(robot_name, "current_joint_positions", [], {}, timeout_s=timeout_s)
          
    def join_motion(self, robot_name: str, timeout: float = 0.0):  
        """Wait for motion to complete"""  
        return self._request(robot_name, "join_motion", [timeout], {}, timeout_s=max(5.0, timeout + 1.0))
          
    def shutdown(self):  
        """Shutdown all robot processes"""  
        try:
            self.stop_all_motion(timeout_s=2.0)
        except Exception:
            # Continue shutdown even if stopping fails for one robot.
            pass

        for name, queues in self.robots.items():  
            queues['command_queue'].put(("shutdown", [], {}))  
              
        for name, process in self.processes.items():  
            process.join(timeout=5)

        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        
        self.robots.clear()
        self.processes.clear()