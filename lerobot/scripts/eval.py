"""Policy evaluation script with rollout tracking.

Runs a trained policy on the robot for multiple rollouts, collecting success/failure
metrics and timing data. Results are saved to JSON for analysis.
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pprint import pformat
import numpy as np
import time
import os
import json
import sys
import termios
from pathlib import Path

from lerobot.robots import RobotConfig, make_robot_from_config, Robot
from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from lerobot.utils.control_utils import is_headless
from lerobot.utils.robot_utils import precise_sleep

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

# Ensure third-party devices are discoverable by lerobot
from lerobot_robot_bimanual_franka import BimanualFrankaConfig  # noqa: F401
from lerobot_teleoperator_gello import GelloConfig  # noqa: F401

logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    task: str # e.g. "task1"
    model_type: str # e.g. "lora" or "fpft"
    total_steps: int
    timeout: int = 90 # Seconds before rollout counts as failed
    home_pose: list[float] = field(default_factory=lambda: [0, -1.57, 1.57, -1.57, -1.57, -1.57])
    num_rollouts: int = 5

@dataclass
class InferenceConfig:
    ip: str
    port: int
    prompt: str
    eval: EvalConfig
    arm_prefix: str = "l"
    robot: RobotConfig = field(
        default_factory=lambda: BimanualFrankaConfig(
            l_server_ip="192.168.3.11",
            l_robot_ip="192.168.200.2",
            l_gripper_ip="192.168.2.21",
            l_port=18813,
            r_server_ip="192.168.3.10",
            r_robot_ip="192.168.201.10",
            r_gripper_ip="192.168.2.20",
            r_port=18812,
            use_ee_delta=False,
        )
    )
    fps: int = 60


def _build_bimanual_action(obs: dict, action: np.ndarray, arm_prefix: str) -> dict[str, float]:
    action_dict = {
        f"l_joint_{i}": float(obs[f"l_joint_{i}"]) for i in range(1, 8)
    } | {
        f"r_joint_{i}": float(obs[f"r_joint_{i}"]) for i in range(1, 8)
    }
    action_dict["l_gripper"] = float(obs["l_gripper"])
    action_dict["r_gripper"] = float(obs["r_gripper"])

    action_dict[f"{arm_prefix}_joint_1"] = float(action[0])
    action_dict[f"{arm_prefix}_joint_2"] = float(action[1])
    action_dict[f"{arm_prefix}_joint_3"] = float(action[2])
    action_dict[f"{arm_prefix}_joint_4"] = float(action[3])
    action_dict[f"{arm_prefix}_joint_5"] = float(action[4])
    action_dict[f"{arm_prefix}_joint_6"] = float(action[5])
    action_dict[f"{arm_prefix}_gripper"] = float(action[6])
    return action_dict

@dataclass
class RolloutResult:
    timestamp: str
    success: bool
    duration: float
    steps_completed: int
    total_steps: int
    score: float

    def to_dict(self):
        return asdict(self)

def try_load_rollout_results_from_file(eval_config: EvalConfig) -> list[RolloutResult]:
    results_file = f"assets/rollouts/rollout_results_{eval_config.task}_{eval_config.model_type}.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                data = json.load(f)
                return [RolloutResult(**d) for d in data]
            except json.JSONDecodeError:
                return []
    return []

def save_rollout_results_to_file(eval_config: EvalConfig, rollout_results: list[RolloutResult]):
    results_file = f"assets/rollouts/rollout_results_{eval_config.task}_{eval_config.model_type}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in rollout_results], f, indent=2)

def flush_stdin():
    """Flushes the standard input buffer to remove stray keypresses."""
    try:
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except Exception:
        pass

def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (1, 2, right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping evaluation or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    The keys are:
    - Escape: Stop evaluation
    - up: Rollout was successful, go to home pose and wait for right arrow to continue
    - down: Rollout was unsuccessful, go to home pose and wait for right arrow to continue
    - Right arrow: Continue to next rollout
    - Left arrow: Rerecord the current rollout, got to home pose and wait for right arrow to continue

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    events = {}
    events["stop_evaluation"] = False
    events["next_rollout"] = False
    events["rerecord_rollout"] = False
    events["success_rollout"] = False
    events["failure_rollout"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                logger.info("Escape key pressed. Stopping ...")
                events["stop_evaluation"] = True
            elif key == keyboard.Key.right:
                events["next_rollout"] = True
            elif key == keyboard.Key.left:
                events["rerecord_rollout"] = True
            elif key == keyboard.Key.up:
                events["success_rollout"] = True
            elif key == keyboard.Key.down:
                events["failure_rollout"] = True
        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events

def move_to_home(robot: Robot, home_pose: list[float], arm_prefix: str, duration: float = 3.0, fps: int = 60):
    logger.info("Moving to home pose...")
    
    # Get current state
    obs = robot.get_observation()
    current_joints = np.array([obs[f"{arm_prefix}_joint_{i}"] for i in range(1, 7)])
    target_joints = np.array(home_pose)
    
    # Calculate steps
    steps = int(duration * fps)
    
    for i in range(steps):
        # Linear interpolation (LERP)
        alpha = (i + 1) / steps
        interpolated_joints = current_joints + alpha * (target_joints - current_joints)
        
        single_arm_action = np.zeros(7, dtype=float)
        single_arm_action[:6] = interpolated_joints
        single_arm_action[6] = 0.0
        action_dict = _build_bimanual_action(obs, single_arm_action, arm_prefix)
        
        robot.send_action(action_dict)
        precise_sleep(1 / fps)
        
    time.sleep(0.5) # Settle time


def evaluation_loop(client, robot: Robot, events: dict, inference_config: InferenceConfig):
    eval_config = inference_config.eval
    fps = inference_config.fps
    
    rollout_results = try_load_rollout_results_from_file(eval_config)
    logger.info(f"Loaded {len(rollout_results)} rollout results from file.")

    move_to_home(robot, eval_config.home_pose, inference_config.arm_prefix)
    
    # Ensure we continue from where we left off
    while len(rollout_results) < eval_config.num_rollouts:
        rollout_idx = len(rollout_results)
        logger.info(f"=== Starting Rollout {rollout_idx + 1}/{eval_config.num_rollouts} ===")
        
        # Reset events
        for k in events:
            events[k] = False
        
        logger.info("Rollout started. Press UP for Success, DOWN for Failure, LEFT to Retry, RIGHT to Skip/Next.")
        
        action_queue = deque([])
        rollout_start_wall = time.time()
        robot_active_time = 0.0
        
        stop_rollout = False
        rollout_success = False
        
        while not stop_rollout:
            loop_start = time.perf_counter()
            
            # 1. Check Global Stop
            if events["stop_evaluation"]:
                logger.info("Stop evaluation requested.")
                return

            # 2. Check Rollout Outcomes
            if events["success_rollout"]:
                rollout_success = True
                stop_rollout = True
                logger.info("User marked SUCCESS.")
                break
            if events["failure_rollout"]:
                rollout_success = False
                stop_rollout = True
                logger.info("User marked FAILURE.")
                break
            if events["rerecord_rollout"]:
                logger.info("User requested RETRY. Discarding current rollout.")
                stop_rollout = True
                rollout_success = None # Signal to retry
                break
            if events["next_rollout"]:
                # mid-rollout -> ignore
                pass
                
            # 3. Check Timeout
            if time.time() - rollout_start_wall > eval_config.timeout:
                logger.info("Timeout reached. Marking as FAILURE.")
                rollout_success = False
                stop_rollout = True
                break
                
            # 4. Inference
            inference_duration = 0.0
            if len(action_queue) == 0:
                obs = robot.get_observation()
                
                # Mapping per remote_pi_inference.py
                obs_dict = {
                    "observation/joint_position": [
                        obs[f"{inference_config.arm_prefix}_joint_{i}"] for i in range(1, 7)
                    ],
                    "observation/gripper_position": obs[f"{inference_config.arm_prefix}_gripper"],
                    "prompt": inference_config.prompt,
                }
                
                t_infer_start = time.perf_counter()
                try:
                    result = client.infer(obs_dict)
                    action_chunk = result["actions"]
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    # Short sleep to avoid busy loop on error
                    time.sleep(0.1)
                    continue
                t_infer_end = time.perf_counter()
                inference_duration = t_infer_end - t_infer_start
                
                if not isinstance(action_chunk, np.ndarray):
                    action_chunk = np.array(action_chunk)
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk.reshape(1, -1)
                action_queue.extend(action_chunk)

            # 5. Execute Action
            if action_queue:
                action = action_queue.popleft()
                action_dict = _build_bimanual_action(obs, action, inference_config.arm_prefix)
                robot.send_action(action_dict)
                
                # Pacing
                loop_end = time.perf_counter()
                dt_s = loop_end - loop_start
                precise_sleep(1 / fps - dt_s)
                
                # Calculate time for this step
                step_end = time.perf_counter()
                step_total = step_end - loop_start
                
                # Accumulate active time: Total step time MINUS the time we waited for inference
                robot_active_time += (step_total - inference_duration)
                
            else:
                # No actions available, just wait
                precise_sleep(1 / fps)

        # Rollout Loop Ended
        
        if rollout_success is None:
             # Retry requested
             events["rerecord_rollout"] = False
        else: 
            total_steps = eval_config.total_steps
            if not rollout_success:
                # Flush stdin before asking for input to clear any buffered keypresses (like arrow keys)
                flush_stdin()
                
                # Wait for user to input the number of completed steps
                steps_completed = 0
                while steps_completed < total_steps - 1: # We know that the last step must be unsuccessful
                    response = input(f"Did it pass step '{steps_completed + 1}'? (y/n): ").strip()
                    if response.lower() != 'n':
                        steps_completed += 1
                    else:
                        break
                score = steps_completed / total_steps
            else:
                # Success -> full score
                score = 1.0
                steps_completed = total_steps
                
            # Save Result
            res = RolloutResult(
                timestamp=time.time(),
                success=rollout_success,
                duration=robot_active_time,
                steps_completed=steps_completed,
                total_steps=total_steps,
                score=score,
            )
            rollout_results.append(res)
            save_rollout_results_to_file(eval_config, rollout_results)
            
            logger.info(f"Rollout {rollout_idx + 1} Result: Success={rollout_success}, Time={robot_active_time:.3f}s")
        
        # Move to home and wait for user input
        move_to_home(robot, eval_config.home_pose, inference_config.arm_prefix)

        # Inter-rollout Wait
        logger.info("Waiting for user input to proceed to next rollout...")
        logger.info("  [RIGHT ARROW] -> Next Rollout")
        logger.info("  [LEFT ARROW]  -> Retry this rollout")
        logger.info("  [ESC]         -> Stop Evaluation")
        
        # Reset events before waiting
        for k in events: events[k] = False
        
        while True:
            if events["next_rollout"]:
                break # Go to outer loop
            if events["rerecord_rollout"]:
                # Remove the just-added result and retry
                rollout_results.pop()
                save_rollout_results_to_file(eval_config, rollout_results)
                break # Go to outer loop (len is now same as before)
            if events["stop_evaluation"]:
                return
            time.sleep(0.05)

@parser.wrap()
def run_evaluation(cfg: InferenceConfig):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    client = None
    listener = None
    
    try:
        client = WebsocketClientPolicy(
            host=cfg.ip,
            port=cfg.port,
            api_key=None,
        )
        
        listener, events = init_keyboard_listener()
        
        evaluation_loop(client, robot, events, cfg)
        
    except KeyboardInterrupt:
        logger.info("Inference stopped by user.")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise e
    finally:
        if client:
            logger.info("Closing client connection...")
            client.close()

        logger.info("Disconnecting robot...")
        robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()

def main():
    register_third_party_plugins()
    run_evaluation()

if __name__ == "__main__":
    main()
