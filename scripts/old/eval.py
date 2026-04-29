"""Policy evaluation script with rollout tracking.

Runs a trained policy against the bimanual Franka for a configurable number of
rollouts, letting the operator mark success/failure at runtime and persisting
per-rollout metrics to JSON for later analysis.
"""

import json
import logging
import os
import sys
import termios
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.utils.control_utils import is_headless
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

# Ensure third-party devices are discoverable by lerobot's config parser.
from lerobot_robot_bimanual_franka import BimanualFrankaConfig  # noqa: F401
from lerobot_teleoperator_gello import GelloConfig  # noqa: F401

logger = logging.getLogger(__name__)

# ---- Constants --------------------------------------------------------------

# Number of joints used by the evaluated policy (currently libero-style 6-DOF,
# even though the FR3 arm itself is 7-DOF).
POLICY_NUM_JOINTS = 6
# Number of joints reported on each arm of the bimanual follower robot.
ROBOT_NUM_JOINTS = 7

# Duration spent traversing from the current pose to the home pose.
HOME_MOVE_DURATION_S = 3.0
# Settling pause after reaching home.
HOME_SETTLE_S = 0.5
# Short backoff after an inference error before retrying.
INFERENCE_ERROR_BACKOFF_S = 0.1
# Polling interval while waiting for the operator's decision between rollouts.
DECISION_POLL_S = 0.05

# Location (relative to CWD) used to persist rollout results per task/model.
ROLLOUTS_DIR = "assets/rollouts"


# ---- Config dataclasses -----------------------------------------------------

@dataclass
class EvalConfig:
    task: str  # e.g. "task1"
    model_type: str  # e.g. "lora" or "fpft"
    total_steps: int
    # Seconds before a rollout is automatically counted as failed.
    timeout: int = 90
    # Target joint angles used as the per-rollout home pose.
    home_pose: list[float] = field(
        default_factory=lambda: [0, -1.57, 1.57, -1.57, -1.57, -1.57]
    )
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


# ---- Rollout results persistence -------------------------------------------

def _results_path(cfg: EvalConfig) -> str:
    return f"{ROLLOUTS_DIR}/rollout_results_{cfg.task}_{cfg.model_type}.json"


def try_load_rollout_results_from_file(eval_config: EvalConfig) -> list[RolloutResult]:
    path = _results_path(eval_config)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return [RolloutResult(**d) for d in json.load(f)]
        except json.JSONDecodeError:
            return []


def save_rollout_results_to_file(eval_config: EvalConfig, rollout_results: list[RolloutResult]):
    path = _results_path(eval_config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in rollout_results], f, indent=2)


# ---- Action construction ----------------------------------------------------

def _build_bimanual_action(obs: dict, action: np.ndarray, arm_prefix: str) -> dict[str, float]:
    """Construct a full bimanual action where non-controlled arm mirrors *obs*."""
    action_dict: dict[str, float] = {}
    for side in ("l", "r"):
        for i in range(1, ROBOT_NUM_JOINTS + 1):
            action_dict[f"{side}_joint_{i}"] = float(obs[f"{side}_joint_{i}"])
        action_dict[f"{side}_gripper"] = float(obs[f"{side}_gripper"])

    # Overlay the policy's output onto the controlled arm (first N joints + gripper).
    for i in range(POLICY_NUM_JOINTS):
        action_dict[f"{arm_prefix}_joint_{i + 1}"] = float(action[i])
    action_dict[f"{arm_prefix}_gripper"] = float(action[POLICY_NUM_JOINTS])
    return action_dict


# ---- Keyboard / stdin helpers ----------------------------------------------

def flush_stdin() -> None:
    """Discard buffered stdin so stray keypresses do not leak into prompts."""
    try:
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except Exception:
        pass


def init_keyboard_listener():
    """
    Start a non-blocking keyboard listener that sets event flags during rollout.

    Key bindings:
        Escape      - stop evaluation
        Up arrow    - mark current rollout a success
        Down arrow  - mark current rollout a failure
        Right arrow - advance to the next rollout
        Left arrow  - retry / rerecord the current rollout

    Returns ``(listener, events)``; ``listener`` is ``None`` in headless envs.
    """
    events = {
        "stop_evaluation": False,
        "next_rollout": False,
        "rerecord_rollout": False,
        "success_rollout": False,
        "failure_rollout": False,
    }

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        return None, events

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


# ---- Home pose motion -------------------------------------------------------

def move_to_home(
    robot: Robot,
    home_pose: list[float],
    arm_prefix: str,
    duration: float = HOME_MOVE_DURATION_S,
    fps: int = 60,
) -> None:
    """Linearly interpolate the controlled arm to the home pose at *fps*."""
    logger.info("Moving to home pose...")

    obs = robot.get_observation()
    current_joints = np.array(
        [obs[f"{arm_prefix}_joint_{i}"] for i in range(1, POLICY_NUM_JOINTS + 1)]
    )
    target_joints = np.array(home_pose)

    steps = int(duration * fps)
    for i in range(steps):
        alpha = (i + 1) / steps
        interpolated = current_joints + alpha * (target_joints - current_joints)

        # Policy action layout: [joint_1..N, gripper].
        policy_action = np.zeros(POLICY_NUM_JOINTS + 1, dtype=float)
        policy_action[:POLICY_NUM_JOINTS] = interpolated
        # Gripper is held at 0 while homing.

        robot.send_action(_build_bimanual_action(obs, policy_action, arm_prefix))
        precise_sleep(1 / fps)

    time.sleep(HOME_SETTLE_S)


# ---- Evaluation loop --------------------------------------------------------

def _reset_events(events: dict) -> None:
    for k in events:
        events[k] = False


def _prompt_steps_completed(total_steps: int) -> int:
    """Ask operator how many task steps were completed prior to failure."""
    # The final step cannot have been completed since the rollout failed.
    steps_completed = 0
    while steps_completed < total_steps - 1:
        response = input(f"Did it pass step '{steps_completed + 1}'? (y/n): ").strip()
        if response.lower() == "n":
            break
        steps_completed += 1
    return steps_completed


def _run_single_rollout(
    client,
    robot: Robot,
    events: dict,
    inference_config: InferenceConfig,
) -> tuple[bool | None, float]:
    """
    Execute one rollout until the user or timeout terminates it.

    Returns ``(success, robot_active_time)`` where ``success`` is ``None`` when
    the operator requested a retry and the rollout should be discarded.
    """
    eval_config = inference_config.eval
    fps = inference_config.fps

    action_queue: deque = deque([])
    rollout_start_wall = time.time()
    robot_active_time = 0.0

    rollout_success: bool | None = False
    stop_rollout = False
    obs = None

    while not stop_rollout:
        loop_start = time.perf_counter()

        if events["stop_evaluation"]:
            logger.info("Stop evaluation requested.")
            # Propagate as success=False so the outer loop's stop handling runs.
            return False, robot_active_time

        if events["success_rollout"]:
            logger.info("User marked SUCCESS.")
            return True, robot_active_time
        if events["failure_rollout"]:
            logger.info("User marked FAILURE.")
            return False, robot_active_time
        if events["rerecord_rollout"]:
            logger.info("User requested RETRY. Discarding current rollout.")
            return None, robot_active_time

        # Hard timeout -> count as failure.
        if time.time() - rollout_start_wall > eval_config.timeout:
            logger.info("Timeout reached. Marking as FAILURE.")
            return False, robot_active_time

        # Fetch a new action chunk if the queue is empty.
        inference_duration = 0.0
        if not action_queue:
            obs = robot.get_observation()

            obs_dict = {
                "observation/joint_position": [
                    obs[f"{inference_config.arm_prefix}_joint_{i}"]
                    for i in range(1, POLICY_NUM_JOINTS + 1)
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
                time.sleep(INFERENCE_ERROR_BACKOFF_S)
                continue
            inference_duration = time.perf_counter() - t_infer_start

            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)
            action_queue.extend(action_chunk)

        if action_queue:
            action = action_queue.popleft()
            robot.send_action(_build_bimanual_action(obs, action, inference_config.arm_prefix))

            # Pacing: sleep to maintain fps, excluding inference wait.
            dt_s = time.perf_counter() - loop_start
            precise_sleep(1 / fps - dt_s)

            step_total = time.perf_counter() - loop_start
            # Accumulate "active" time - total minus time spent blocked on inference.
            robot_active_time += step_total - inference_duration
        else:
            precise_sleep(1 / fps)

    # Unreachable; the loop either returns directly or the timeout branch fires.
    return rollout_success, robot_active_time


def evaluation_loop(client, robot: Robot, events: dict, inference_config: InferenceConfig):
    eval_config = inference_config.eval

    rollout_results = try_load_rollout_results_from_file(eval_config)
    logger.info(f"Loaded {len(rollout_results)} rollout results from file.")

    move_to_home(robot, eval_config.home_pose, inference_config.arm_prefix)

    # Continue from wherever we left off in a previous session.
    while len(rollout_results) < eval_config.num_rollouts:
        rollout_idx = len(rollout_results)
        logger.info(f"=== Starting Rollout {rollout_idx + 1}/{eval_config.num_rollouts} ===")
        _reset_events(events)
        logger.info(
            "Rollout started. Press UP for Success, DOWN for Failure, LEFT to Retry, RIGHT to Skip/Next."
        )

        rollout_success, robot_active_time = _run_single_rollout(
            client, robot, events, inference_config
        )

        # An evaluation stop was requested mid-rollout.
        if events["stop_evaluation"]:
            return

        if rollout_success is None:
            # Retry: discard this rollout entirely and try again.
            events["rerecord_rollout"] = False
        else:
            total_steps = eval_config.total_steps
            if rollout_success:
                score = 1.0
                steps_completed = total_steps
            else:
                # Clear arrow-key presses before prompting for keyboard input.
                flush_stdin()
                steps_completed = _prompt_steps_completed(total_steps)
                score = steps_completed / total_steps

            rollout_results.append(
                RolloutResult(
                    timestamp=time.time(),
                    success=rollout_success,
                    duration=robot_active_time,
                    steps_completed=steps_completed,
                    total_steps=total_steps,
                    score=score,
                )
            )
            save_rollout_results_to_file(eval_config, rollout_results)
            logger.info(
                f"Rollout {rollout_idx + 1} Result: Success={rollout_success}, Time={robot_active_time:.3f}s"
            )

        move_to_home(robot, eval_config.home_pose, inference_config.arm_prefix)

        logger.info("Waiting for user input to proceed to next rollout...")
        logger.info("  [RIGHT ARROW] -> Next Rollout")
        logger.info("  [LEFT ARROW]  -> Retry this rollout")
        logger.info("  [ESC]         -> Stop Evaluation")

        _reset_events(events)
        while True:
            if events["next_rollout"]:
                break
            if events["rerecord_rollout"]:
                # Remove the just-added result and retry at the same index.
                rollout_results.pop()
                save_rollout_results_to_file(eval_config, rollout_results)
                break
            if events["stop_evaluation"]:
                return
            time.sleep(DECISION_POLL_S)


# ---- Entry points -----------------------------------------------------------

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
        client = WebsocketClientPolicy(host=cfg.ip, port=cfg.port, api_key=None)
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
