#!/usr/bin/env python3
"""The LIBERO half of a sim baseline rollout: the stock env, the observation the
converted datasets were built from, and the inverse of their action space.

Runs in `multi-fast/.venv`, the only interpreter here with robosuite and libero.
The policy stays in its own venv behind ZMQ (baselines/zmq_client.py), so
nothing in this module imports torch or either upstream.

Module level is numpy + cv2 only, so `libero_bridge/dataset.py` can import the
camera map, `policy_frame` and the task table from the workspace venv. A
training frame and a rollout frame have to come out of the same code; a
converter-only definition is how the two drift and the policy is quietly
evaluated off-distribution.

The plant is built STOCK -- no plant overrides, no gripper overrides -- because
`regenerate_libero_dataset.py` recorded the demonstrations under exactly that
model. Evaluating a fitted plant here would measure a sim2sim gap rather than
the policies. The controller is stock too unless a backend names its own
execution controller (`SimTask(controller=...)`): SAIL's method is to replace
the teleoperation controller with a stiff one at rollout, and its reached-pose
targets are what make that valid.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

_MULTI_FAST = Path(__file__).resolve().parent.parent.parent / "multi-fast"
# Converted tasks are filed by index, not name: `libero_90/task_9`.
TAG_PREFIX = "task_"

# LIBERO's action layout: [dpos(3), drotvec(3), gripper(1)], normalised to
# [-1, 1] against robosuite's osc_pose.json output_max. Gripper is +-1.
GRIPPER = 6
POSE_DIM = 7
# Both upstreams' benchmark resolution (SAIL's and B-Spline's crop is 76 of 84).
DEFAULT_IMAGE_SIZE = (84, 84)
# What regenerate_libero_dataset.py rendered at; the converter downscaled from
# it, so rolling out at anything else resamples differently.
RENDER_RESOLUTION = 256
# LIBERO -> robosuite's own observation names, which is what SAIL's robomimic
# template already lists under observation.modalities.obs.
CAMERAS = (("agentview_rgb", "agentview_image"),
           ("eye_in_hand_rgb", "robot0_eye_in_hand_image"))
# regenerate_libero_dataset.py's own settle before it records anything.
SETTLE_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def policy_frame(frame: np.ndarray, image_size, flip: bool = True) -> np.ndarray:
    """One rendered HWC uint8 frame -> the frame the converter stored.

    LIBERO's offscreen renderer returns these upside down and multi-fast's
    LIBEROObservationWrapper rotates them 180 before any policy sees them.
    """
    if flip:
        # Contiguous immediately: cv2 will not take a negative-stride view.
        frame = np.ascontiguousarray(frame[::-1, ::-1])
    if image_size is not None and tuple(frame.shape[:2]) != tuple(image_size[::-1]):
        frame = cv2.resize(frame, tuple(image_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _multi_fast_on_path() -> None:
    if str(_MULTI_FAST) not in sys.path:
        sys.path.insert(0, str(_MULTI_FAST))


def max_steps(suite: str) -> int:
    """Upstream's per-suite episode budget, from multi-fast's own table."""
    _multi_fast_on_path()
    from utils.envs.libero import LIBERO_MAX_STEPS
    if suite not in LIBERO_MAX_STEPS:
        raise SystemExit(f"unknown suite {suite!r}; known: {sorted(LIBERO_MAX_STEPS)}")
    return int(LIBERO_MAX_STEPS[suite])


def _libero_on_path() -> None:
    # multi-fast/.venv installs libero; the workspace venv, where the converter
    # runs, does not. The task table needs only torch and yaml.
    if importlib.util.find_spec("libero") is None:
        sys.path.insert(0, str(_MULTI_FAST / "LIBERO"))


def suites() -> tuple[str, ...]:
    _libero_on_path()
    from libero.libero import benchmark
    return tuple(benchmark.get_benchmark_dict())


@functools.cache
def task_names(suite: str) -> tuple[str, ...]:
    """The suite's task names in LIBERO's own order, which is what an index means."""
    if suite not in suites():
        raise SystemExit(f"unknown suite {suite!r}; known: {sorted(suites())}")
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[suite]()
    return tuple(ts.get_task(i).name for i in range(ts.n_tasks))


def task_tag(index: int) -> str:
    """`task_9`: what task 9's converted files, policies and rollouts are filed under."""
    return f"{TAG_PREFIX}{int(index)}"


def tag_index(text: str | int) -> int | None:
    """9 from `9` or `task_9`; None for anything else, such as a task name."""
    digits = str(text).removeprefix(TAG_PREFIX)
    return int(digits) if digits.isdigit() else None


def resolve_task(suite: str, task: str | int) -> int:
    """Task index from an index (`9`), its tag (`task_9`), a LIBERO task name, or
    a `<task>_demo` stem."""
    names = task_names(suite)
    index = tag_index(task)
    if index is not None:
        if not 0 <= index < len(names):
            raise SystemExit(f"task index {index} is outside {suite} (0..{len(names) - 1})")
        return index
    name = str(task).removesuffix(".hdf5").removesuffix("_demo")
    if name not in names:
        raise SystemExit(f"{name!r} is not a task of {suite}")
    return names.index(name)


def osc_config(kp=None, damping_ratio=None) -> dict:
    """Gains to merge over robosuite's osc_pose.json; None keeps the stock value.

    The per-step limit (output_max) is LIBERO's action space, so it is never
    overridden.
    """
    out = {}
    if kp is not None:
        out["kp"] = float(kp)
    if damping_ratio is not None:
        out["damping_ratio"] = float(damping_ratio)
    return out


def osc_report(controller) -> dict:
    """What the live OSC actually runs, for the run manifest."""
    kp = np.asarray(controller.kp, dtype=np.float64)
    kd = np.asarray(controller.kd, dtype=np.float64)
    return {"kp": kp.tolist(),
            "damping_ratio": (kd / (2.0 * np.sqrt(kp))).round(6).tolist(),
            "output_max": np.asarray(controller.output_max, dtype=np.float64).tolist()}


class SimTask:
    """One LIBERO task's stock env, its init states, and the two conversions
    between the env and the action space the converted datasets are in."""

    def __init__(self, suite: str, task: str | int, *, seed: int = 0,
                 resolution: int = RENDER_RESOLUTION,
                 image_size=DEFAULT_IMAGE_SIZE, flip_images: bool = True,
                 controller: dict | None = None) -> None:
        _multi_fast_on_path()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite = suite
        self.image_size = image_size
        self.flip_images = flip_images
        self.max_steps = max_steps(suite)

        task_suite = benchmark.get_benchmark_dict()[suite]()
        self.task_index = resolve_task(suite, task)
        self.task = task_suite.get_task(self.task_index)
        self.name = self.task.name
        self.language = self.task.language
        self.init_states = task_suite.get_task_init_states(self.task_index)

        bddl = (Path(get_libero_path("bddl_files"))
                / self.task.problem_folder / self.task.bddl_file)
        # Stock plant and gripper. `controller` takes osc_config's arguments;
        # hard resets rebuild the robot from the same merged config.
        # ignore_done: the rollout owns the episode budget; robosuite would refuse step 1001.
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                      camera_heights=resolution,
                                      camera_widths=resolution,
                                      controller_configs=osc_config(**(controller or {})),
                                      ignore_done=True)
        self.env.seed(seed)
        self._site_to_body = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def controller(self):
        """Live OSC controller behind LIBERO's wrappers. Valid only post-reset --
        robosuite rebuilds it on each hard reset."""
        return self.env.env.robots[0].controller

    @property
    def n_init_states(self) -> int:
        return int(len(self.init_states))

    def start(self, episode: int, state=None) -> dict:
        """Reset onto init state `episode` and let the objects settle.

        Returns the observation the first action is chosen from -- the same one
        regenerate_libero_dataset.py recorded as step 0. `state` overrides the
        suite's init state with a raw mujoco state, which is what lets a
        recorded demo be replayed through this same path.
        """
        from utils.base_policy_utils import estimate_site_to_body

        self.env.reset()
        if state is None:
            state = self.init_states[episode % self.n_init_states]
        obs = self.env.set_init_state(state)
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self.env.step(DUMMY_ACTION)
        # Constant grip-site -> wrist-body transform, estimated once per episode
        # at a matched reading, exactly as the relabeler does.
        self._site_to_body = estimate_site_to_body(self.controller, obs["robot0_eef_quat"])
        return obs

    def step(self, action, dt: float | None = None):
        """One env step; `dt` sets this step's control period, as SAIL's patched robosuite does."""
        if dt is None:
            return self.env.step(np.asarray(action, dtype=np.float64).tolist())
        env = self.env.env
        env.control_timestep = dt
        try:
            _, reward, done, info = self.env.step(np.asarray(action, dtype=np.float64).tolist())
        finally:
            env.control_timestep = 1.0 / env.control_freq
        # Observables refresh at the env's own rate; read them fresh after a shorter step.
        return env._get_observations(force_update=True), reward, done, info

    def wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """(force N, torque Nm) at the gripper's wrist sensor, read after a step as
        multi-fast's LIBEROEvalWrapper does for eval_fast.py."""
        robot = self.env.env.robots[0]
        return (np.array(robot.ee_force, dtype=np.float64),
                np.array(robot.ee_torque, dtype=np.float64))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    # -- conversions -------------------------------------------------------

    def observe(self, raw: dict, shapes: dict) -> dict:
        """Env observation -> exactly the keys the checkpoint declares.

        Both converted files use robosuite's own observation names, so one
        builder serves SAIL and B-Spline. Sizes come from the checkpoint rather
        than from `image_size` because nothing else in the stack would catch a
        checkpoint trained at a different resolution.
        """
        out = {}
        for key, shape in shapes.items():
            if key.endswith("_image"):
                if key not in raw:
                    raise KeyError(f"the checkpoint wants {key!r}; the env renders "
                                   f"{sorted(k for k in raw if k.endswith('_image'))}")
                out[key] = policy_frame(raw[key], (int(shape[-1]), int(shape[-2])),
                                        self.flip_images)
            else:
                if key not in raw:
                    raise KeyError(f"the checkpoint wants {key!r}, which this env "
                                   f"does not observe")
                out[key] = np.asarray(raw[key], dtype=np.float32).reshape(-1)
        return out

    def action(self, pos, rotvec, gripper: float) -> np.ndarray:
        """Absolute pose target -> the normalised OSC delta that lands on it.

        `target_slot_to_delta` is multi-fast's own inverse of the relabeler that
        wrote `goal_pos`/`goal_ori`, so the executor and the converter cannot
        drift. It re-reads the controller, which is why it must be called with
        the arm at the state the action is for.
        """
        from utils.base_policy_utils import target_slot_to_delta

        if self._site_to_body is None:
            raise RuntimeError("start() must run before action(): the tool transform "
                               "is estimated per episode")
        slot = np.concatenate([np.asarray(pos, dtype=np.float64).reshape(3),
                               np.asarray(rotvec, dtype=np.float64).reshape(3),
                               [float(gripper)]])
        return target_slot_to_delta(slot, self.controller, 0,
                                    site_to_body=self._site_to_body)
