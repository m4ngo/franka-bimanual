from multiprocessing import Queue, Process  
from typing import Any, cast  
from queue import Empty

import numpy as np
from numpy.typing import NDArray


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
            from net_franky import setup_net_franky  
            setup_net_franky(self.server_ip, self.port)  
              
            from net_franky.franky import Robot, JointMotion, CartesianVelocityMotion, ReferenceType, Twist 
              
            robot = Robot(self.robot_ip)  
            robot.recover_from_errors()  
            robot.relative_dynamics_factor = 0.1
        except Exception as e:
            self.response_queue.put(("error", f"Failed to initialize robot: {e}"))
            return  
          
        while True:  
            try:  
                command, args, kwargs = self.command_queue.get()  
                  
                if command == "move_joints":  
                    joint_target = _validate_vector("move_joints position", args[0], 7)
                    motion = JointMotion(joint_target, ReferenceType.Absolute)  
                    result = robot.move(motion, asynchronous=args[1])  
                    self.response_queue.put(("success", result))
                  
                elif command == "move_ee_delta":  
                    ee_delta = _validate_vector("move_ee_delta position", args[0], 6)
                    linear: NDArray[np.float64] = np.asarray(ee_delta[:3], dtype=np.float64)
                    angular: NDArray[np.float64] = np.asarray(ee_delta[3:], dtype=np.float64)
                    motion = CartesianVelocityMotion(Twist(cast(Any, linear), cast(Any, angular)))
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
                      
                elif command == "shutdown":  
                    break  
                      
                else:  
                    self.response_queue.put(("error", f"Unknown command: {command}"))  
                      
            except Exception as e:  
                self.response_queue.put(("error", str(e)))  

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

    def _request(self, robot_name: str, command: str, args: list, kwargs: dict, timeout_s: float = 5.0):
        if robot_name not in self.robots:
            raise KeyError(f"Robot '{robot_name}' is not registered")

        process = self.processes.get(robot_name)
        if process is None or not process.is_alive():
            raise RuntimeError(f"Robot process '{robot_name}' is not alive")

        queues = self.robots[robot_name]
        queues['command_queue'].put((command, args, kwargs))

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
          
    def move_joints(self, robot_name: str, position: list, asynchronous: bool = False):  
        """Move robot to cartesian position"""  
        position = _validate_vector("move_joints position", position, 7)
        return self._request(robot_name, "move_joints", [position, asynchronous], {}, timeout_s=5.0)
    
    def move_ee_delta(self, robot_name: str, position: list, asynchronous: bool = False):
        """Move robot to cartesian position"""  
        position = _validate_vector("move_ee_delta position", position, 6)
        return self._request(robot_name, "move_ee_delta", [position, asynchronous], {}, timeout_s=5.0)
          
    def get_robot_state(self, robot_name: str):  
        """Get current robot state"""  
        return self._request(robot_name, "get_state", [], {}, timeout_s=5.0)
    
    def current_joint_positions(self, robot_name: str):
        """Get current joint positions"""  
        return self._request(robot_name, "current_joint_positions", [], {}, timeout_s=5.0)
          
    def join_motion(self, robot_name: str, timeout: float = 0.0):  
        """Wait for motion to complete"""  
        return self._request(robot_name, "join_motion", [timeout], {}, timeout_s=max(5.0, timeout + 1.0))
          
    def shutdown(self):  
        """Shutdown all robot processes"""  
        for name, queues in self.robots.items():  
            queues['command_queue'].put(("shutdown", [], {}))  
              
        for name, process in self.processes.items():  
            process.join(timeout=5)
        
        self.robots.clear()
        self.processes.clear()