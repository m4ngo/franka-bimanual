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
from pathlib import Path

import numpy as np

POSES_DIR = Path(__file__).resolve().parent.parent / "home_poses"

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
from lerobot_robot_bimanual_franka import SingleArmFranka, SingleArmFrankaConfig
from lerobot_teleoperator_gello import GelloConfig, GelloEEConfig

logger = logging.getLogger(__name__)


_R_SERVER_IP, _R_ROBOT_IP, _R_GRIPPER_IP, _R_PORT = "192.168.3.10", "192.168.201.10", "192.168.2.20", 18812
_R_GELLO_PORT = "/dev/ttyUSB0"


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _build_robot(use_ee_pos: bool) -> SingleArmFranka:
    cfg = SingleArmFrankaConfig(
        r_server_ip=_R_SERVER_IP,
        r_robot_ip=_R_ROBOT_IP,
        r_gripper_ip=_R_GRIPPER_IP,
        r_port=_R_PORT,
        use_ee_pos=use_ee_pos,
    )
    return make_robot_from_config(cfg)


def _build_teleop(mode: str, teleop_id: str):
    if mode == "gello":
        cfg = GelloConfig(id=teleop_id, side="r", port=_R_GELLO_PORT)
    elif mode == "gello_ee":
        cfg = GelloEEConfig(id=teleop_id, side="r", port=_R_GELLO_PORT)
    else:
        raise ValueError(f"Unsupported --teleop-mode: {mode!r}. Use 'gello' or 'gello_ee'.")
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
    p.add_argument("--use-ee-pos", type=_str2bool, required=True)
    p.add_argument(
        "--teleop-mode",
        default="gello_ee",
        choices=("gello", "gello_ee"),
        help="Teleop type (ignored when --policy is set)",
    )
    p.add_argument("--teleop-id", default="homed_single_arm_teleop")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--reset-time-s", type=float, default=3.0, help="Time between episodes for manual reset")
    p.add_argument("--resume", type=_str2bool, default=False)
    p.add_argument("--push-to-hub", type=_str2bool, default=True)
    p.add_argument("--play-sounds", type=_str2bool, default=False)

    p.add_argument(
        "--home-pose-name",
        default=None,
        help=f"Name of a saved pose in {POSES_DIR} (overrides --home-q-right and --home-gripper)",
    )
    p.add_argument(
        "--home-q-right",
        nargs=7,
        type=float,
        default=[-0.28223089288736675, -0.5594522989991991, -0.4191884798561259, -1.82212661700904, 0.06416041394704838, 1.5246974433097138, -0.7569427650529224],
    )
    p.add_argument(
        "--home-gripper",
        type=float,
        default=1.0,
        help="Normalized gripper at home (0=closed, 1=open)",
    )
    p.add_argument("--home-max-time-s", type=float, default=3.0)
    p.add_argument(
        "--home-tol-rad",
        type=float,
        default=0.05,
        help="Joint homing: max per-joint error (rad). EE homing: default rot tolerance if --home-tol-rot-rad unset.",
    )
    p.add_argument(
        "--home-fps",
        type=int,
        default=None,
        help="Homing control rate (Hz). Default: max(--fps, 60) in EE mode, else --fps.",
    )
    p.add_argument("--home-tol-m", type=float, default=0.03, help="EE homing only: max position error (m)")
    p.add_argument(
        "--home-tol-rot-rad",
        type=float,
        default=None,
        help="EE homing only: max axis-angle error (rad); defaults to --home-tol-rad",
    )

    args = p.parse_args()
    init_logging()

    args._policy_cfg = None
    if args.policy:
        args._policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
        args._policy_cfg.pretrained_path = args.policy

    robot = _build_robot(use_ee_pos=args.use_ee_pos)
    teleop = None if args.policy else _build_teleop(args.teleop_mode, args.teleop_id)

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

    if args.home_pose_name:
        pose_path = POSES_DIR / f"{args.home_pose_name}.json"
        pose = json.loads(pose_path.read_text())
        home_q_r = np.asarray(pose["r_q"], dtype=np.float64)
        args.home_gripper = float(pose.get("gripper", args.home_gripper))
        logger.info("Loaded home pose %r from %s", args.home_pose_name, pose_path)
    else:
        home_q_r = np.asarray(args.home_q_right, dtype=np.float64)

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

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
                if args.use_ee_pos:
                    home_kw["tol_pos_m"] = args.home_tol_m
                    home_kw["tol_rot_rad"] = args.home_tol_rot_rad
                ok = robot.home(home_q_left=None, home_q_right=home_q_r, **home_kw)
                if not ok:
                    logger.warning("Homing did not converge before episode %d; proceeding anyway", dataset.num_episodes)

                log_say(f"Recording episode {dataset.num_episodes}", args.play_sounds)
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

                if not events["stop_recording"] and (
                    (recorded < args.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", args.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=args.fps,
                        teleop_action_processor=teleop_proc,
                        robot_action_processor=robot_action_proc,
                        robot_observation_processor=robot_obs_proc,
                        teleop=teleop,
                        control_time_s=args.reset_time_s,
                        single_task=args.task,
                    )

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