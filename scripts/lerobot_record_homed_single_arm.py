"""Homed single-arm Franka recording/rollout orchestrator.

This is the right-arm-only counterpart to ``lerobot_record_homed.py``.
It preserves the same record-loop behavior, but builds the single-arm robot
wrapper and only homes/drives the right arm.

The saved pose format is still compatible with ``home_poses/home_pose.json``;
only ``r_q`` and ``gripper`` are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import sys
import termios
import threading
import tty
from pathlib import Path
import time

import franka_config as fc
import numpy as np

POSES_DIR = fc.home_poses_dir()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging, log_say

# Importing the plugin packages triggers their @register_subclass decorators.
from lerobot_robot_bimanual_franka import ControlMode, SingleArmFranka, SingleArmFrankaConfig
from lerobot_teleoperator_gello import GelloConfig, GelloEEConfig
from lerobot_teleoperator_spacemouse import SpaceMouseConfig

logger = logging.getLogger(__name__)


class _StdinKeyboardThread:
    """Raw-stdin keyboard reader that updates a LeRobot events dict.

    pynput (used by init_keyboard_listener) requires an X11/Wayland display and
    silently fails over SSH without X11 forwarding.  This thread reads escape
    sequences directly from stdin so that right-arrow, left-arrow, and Escape
    work in any terminal, including SSH sessions.

    Start it just before record_loop and stop it immediately after so that the
    raw-mode terminal setting does not interfere with input() calls between episodes.
    """

    def __init__(self, events: dict) -> None:
        self._events = events
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="stdin-kb")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop_evt.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                time.sleep(0.02)  # let multi-byte sequences fully arrive
                data = os.read(fd, 16)
                if b"\x03" in data:                                          # Ctrl-C
                    self._events["stop_recording"] = True
                    self._events["exit_early"] = True
                elif data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):  # right arrow
                    print("\r\nRight arrow: ending episode early...", flush=True)
                    self._events["exit_early"] = True
                elif data.startswith(b"\x1b[D") or data.startswith(b"\x1bOD"):  # left arrow
                    print("\r\nLeft arrow: ending episode and re-recording...", flush=True)
                    self._events["rerecord_episode"] = True
                    self._events["exit_early"] = True
                elif data == b"\x1b":                                        # Escape
                    print("\r\nEscape: stopping recording...", flush=True)
                    self._events["stop_recording"] = True
                    self._events["exit_early"] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Which physical arm the `r_` keys drive comes from config/rig.yaml. The leader
# device is a SEPARATE question — which SpaceMouse/GELLO the operator holds —
# so it comes from the profile's teleop_device, never from the arm. Both
# SpaceMice enumerate as identical HID devices, so picking the wrong one
# connects successfully and then does nothing.
_PROFILE = "single_arm_franka"
_ARM_KEY = fc.profile(_PROFILE).depth_center_arm
_TELEOP_DEVICE = fc.profile(_PROFILE).teleop_device


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _build_robot(control_mode: ControlMode, depth: bool = True, noise: bool = False) -> SingleArmFranka:
    cfg = SingleArmFrankaConfig(
        control_mode=control_mode,
        depth=depth,
        use_noise=noise,
    )
    return make_robot_from_config(cfg)


def _build_teleop(mode: str, teleop_id: str, device: str | None = None):
    device = device or _TELEOP_DEVICE
    if device is None:
        raise ValueError(
            f"rig profile {_PROFILE!r} has no teleop_device; set one in config/rig.yaml "
            "or pass --teleop-device."
        )
    gello_port = fc.teleop(f"gello.devices.{device}.port")
    spacemouse_path = fc.teleop(f"spacemouse.devices.{device}.hidraw_path")
    if mode == "gello":
        cfg = GelloConfig(id=teleop_id, side=_ARM_KEY, port=gello_port, use_noise=True)
    elif mode == "gello_ee":
        cfg = GelloEEConfig(id=teleop_id, side=_ARM_KEY, port=gello_port, use_noise=True)
    elif mode == "spacemouse":
        # translation/rotation scale come from config/teleop.yaml defaults.
        cfg = SpaceMouseConfig(
            id=teleop_id,
            hidraw_path=spacemouse_path,
            prefix=f"{_ARM_KEY}_",
            use_delta=True,
        )
    else:
        raise ValueError(f"Unsupported --teleop-mode: {mode!r}. Use 'gello', 'gello_ee', or 'spacemouse'.")
    return make_teleoperator_from_config(cfg)


def _build_dataset(args, robot, teleop_proc, robot_obs_proc) -> LeRobotDataset:
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    n_cams = len(getattr(robot, "cameras", {}))
    common = dict(
        batch_encoding_size=1,
        vcodec="auto",
        streaming_encoding=True,
        encoder_queue_maxsize=8,
        encoder_threads=2,
    )

    if args.resume:
        dataset = LeRobotDataset.resume(
            args.repo_id,
            root=args.output_dir,
            image_writer_processes=0 if n_cams == 0 else 0,
            image_writer_threads=4 * n_cams if n_cams > 0 else 0,
            **common,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
    else:
        sanity_check_dataset_name(args.repo_id, args._policy_cfg)
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            root=args.output_dir,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * n_cams,
            **common,
        )
    return dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, required=True)
    p.add_argument("--task", required=True, help="single_task description")
    p.add_argument("--policy", default=None, help="HF repo for a pretrained policy; omit for teleop recording")
    p.add_argument("--control-mode", default=fc.profile(_PROFILE).control_mode,
                   choices=("JOINT_POS", "EE_POS", "EE_DELTA"), help="Robot control mode")
    p.add_argument("--depth", type=_str2bool, default=True,
                   help="Enable depth point-cloud observations (default: true)")
    p.add_argument(
        "--teleop-mode",
        default="gello_ee",
        choices=("gello", "gello_ee", "spacemouse"),
        help="Teleop type (ignored when --policy is set)",
    )
    p.add_argument("--teleop-device", default=_TELEOP_DEVICE,
                   help="Which physical leader the operator holds (teleop.yaml devices). "
                        "Defaults to the rig profile's teleop_device.")
    p.add_argument("--teleop-id", default="homed_single_arm_teleop")
    p.add_argument("--fps", type=int, default=fc.control_fps())
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--reset-time-s", type=float, default=5.0, help="Time between episodes for manual reset")
    p.add_argument("--resume", type=_str2bool, default=False)
    p.add_argument("--push-to-hub", type=_str2bool, default=True)
    p.add_argument("--play-sounds", type=_str2bool, default=False)

    p.add_argument(
        "--home-pose-name",
        default=fc.default_home_pose_name(),
        help=f"Name of a saved pose in {POSES_DIR} (overrides --home-q-right and --home-gripper)",
    )
    p.add_argument(
        "--home-q-right",
        nargs=7,
        type=float,
        default=None,
        help="7 joint angles (rad) overriding the saved home pose",
    )
    p.add_argument(
        "--home-gripper",
        type=float,
        default=fc.control("homing.gripper_norm"),
        help="Normalized gripper at home (0=closed, 1=open)",
    )
    p.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    p.add_argument(
        "--home-tol-rad",
        type=float,
        default=fc.control("homing.tol_rad"),
        help="Joint homing: max per-joint error (rad). EE homing: default rot tolerance if --home-tol-rot-rad unset.",
    )
    p.add_argument(
        "--home-fps",
        type=int,
        default=None,
        help="Homing control rate (Hz). Default: control.rates.home_fps in config/control.yaml.",
    )
    p.add_argument("--noise", type=bool, default=False, help="Whether to add noise to actions or not")

    args = p.parse_args()
    init_logging()

    args._policy_cfg = None
    if args.policy:
        args._policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
        args._policy_cfg.pretrained_path = args.policy

    robot = _build_robot(control_mode=ControlMode(args.control_mode), depth=args.depth, noise=args.noise)
    teleop = None if args.policy else _build_teleop(args.teleop_mode, args.teleop_id, args.teleop_device)

    teleop_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    dataset = _build_dataset(args, robot, teleop_proc, robot_obs_proc)

    policy = preprocessor = postprocessor = None
    if args._policy_cfg is not None:
        policy = make_policy(args._policy_cfg, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=args._policy_cfg,
            pretrained_path=args.policy,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": args._policy_cfg.device}},
        )

    if args.home_q_right is not None:
        home_q_r = np.asarray(args.home_q_right, dtype=np.float64)
    else:
        pose = fc.load_home_pose(args.home_pose_name)
        home_q_r = fc.home_q(args.home_pose_name, key=_ARM_KEY)
        args.home_gripper = float(pose.get("gripper", args.home_gripper))
        logger.info("Loaded home pose %r from %s", args.home_pose_name,
                    POSES_DIR / f"{args.home_pose_name}.json")

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()
    stdin_kb = _StdinKeyboardThread(events)

    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < args.num_episodes and not events["stop_recording"]:
                log_say(f"Homing arm before episode {dataset.num_episodes}", args.play_sounds)
                home_kw: dict = dict(
                    gripper_norm=args.home_gripper,
                    max_time_s=args.home_max_time_s,
                    tol_rad=args.home_tol_rad,
                    fps=args.fps,
                    home_fps=args.home_fps,
                )
                ok = robot.home(home_q_left=None, home_q_right=home_q_r, **home_kw)
                if not ok:
                    logger.warning("Homing did not converge before episode %d; proceeding anyway", dataset.num_episodes)

                log_say(f"Recording episode {dataset.num_episodes}", args.play_sounds)
                events["exit_early"] = False
                stdin_kb.start()
                try:
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=args.fps,
                        teleop_action_processor=teleop_proc,
                        robot_action_processor=robot_action_proc,
                        robot_observation_processor=robot_obs_proc,
                        teleop=teleop,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        dataset=dataset,
                        control_time_s=args.episode_time_s,
                        single_task=args.task,
                    )
                finally:
                    stdin_kb.stop()

                if not events["stop_recording"] and (
                    (recorded < args.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", args.play_sounds)
                    # if args.policy is not None:
                    #     stdin_kb.start()
                    #     try:
                    #         record_loop(
                    #             robot=robot,
                    #             events=events,
                    #             fps=args.fps,
                    #             teleop_action_processor=teleop_proc,
                    #             robot_action_processor=robot_action_proc,
                    #             robot_observation_processor=robot_obs_proc,
                    #             teleop=teleop,
                    #             control_time_s=args.reset_time_s,
                    #             single_task=args.task,
                    #         )
                    #     finally:
                    #         stdin_kb.stop()
                    # else:
                    input("Press Enter when ready to start the next episode...")

                if events["rerecord_episode"]:
                    log_say("Re-record episode", args.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1
    finally:
        log_say("Stop recording", args.play_sounds, blocking=True)
        if dataset is not None:
            dataset.finalize()
            if args.push_to_hub:
                try:
                    dataset.push_to_hub()
                except Exception:
                    logger.exception("push_to_hub failed; dataset is still on disk at %s", Path(args.output_dir).resolve())
        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()


if __name__ == "__main__":
    main()