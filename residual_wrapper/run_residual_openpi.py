"""Entry point for running and recording residual-policy episodes on the
Franka, using OpenPI's pi05_libero as the base policy.

This is a thin variant of run_residual.py: it reuses that module's
_run_episode loop, dataset building, home-pose handling, and viz saving
verbatim, swapping only how the base policy is constructed (OpenPIBasePolicy,
a remote websocket-client-backed BasePolicy-compatible wrapper, instead of
the local lerobot-checkpoint BasePolicy). See openpi_policy_wrapper.py for
the FK-based joint-delta -> EE-delta bridging that makes this substitution
possible without touching _run_episode.

Prerequisite: a pi05_libero policy server must already be running, e.g.

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

(gs:// paths are downloaded automatically by openpi's `download.maybe_download`
the first time the server config is resolved.) Start this on a machine with a
GPU, then point --openpi-host/--openpi-port at it from wherever this script
runs (they can be the same machine).
"""

import logging
import os
import sys
import time
from pathlib import Path

import franka_config as fc
import numpy as np

import env_wrapper
# Reuse the entire episode loop, dataset builder, and CLI plumbing from the
# existing residual entry point -- only the base-policy construction differs.
from run_residual import (
    _build_dataset,
    _run_episode,
    _save_viz,
    _str2bool,
    _wait_for_right_arrow,
    _default_home_q,
    _POSES_DIR,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from openpi_policy_wrapper import OpenPIBasePolicy
from policy_wrapper import ResidualPolicy

logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-host", default="localhost",
                        help="Host of the running pi05_libero policy server "
                             "(scripts/serve_policy.py)")
    parser.add_argument("--openpi-port", type=int, default=8000,
                        help="Port of the running pi05_libero policy server")
    parser.add_argument("--prompt", required=True,
                        help="Language instruction sent to pi05_libero with "
                             "every observation, e.g. 'pick up the black bowl'")
    parser.add_argument("--image-key", default="cam_2",
                        help="obs dict key holding the scene/exterior camera "
                             "image fed to pi05_libero as observation/image")
    parser.add_argument("--wrist-image-key", default="cam_5",
                        #choices=("cam_3_wrist", "cam_4_wrist"),
                        help="obs dict key holding the wrist camera image fed "
                             "to pi05_libero as observation/wrist_image "
                             "(rig has two wrist cams; pick one — pi05_libero "
                             "takes a single wrist view)")
    parser.add_argument("--base-kp", type=float, default=0.0,
                        help="Gain forwarded on the base policy's own action "
                             "channel (OpenPI has no notion of gains)")
    parser.add_argument("--base-kd", type=float, default=0.00,
                        help="Gain forwarded on the base policy's own action "
                             "channel (OpenPI has no notion of gains)")
    parser.add_argument(
        "--residual-policy",
        default=str(Path(__file__).resolve().parent.parent / "best.pt"),
        help="Path to residual policy checkpoint (best.pt)",
    )
    parser.add_argument("--save-videos", action="store_true",
                        help="Write one time-aligned mp4 per camera into --viz-dir")
    parser.add_argument("--video-cams", nargs="+", default=None,
                        help="Camera obs keys to record (default: all connected cameras)")
    parser.add_argument("--dump-obs-dir", default=None,
                        help="Dump every residual_obs bundle (+ residual output) to npz")
    parser.add_argument("--no-residual", action="store_true",
                        help="Disable the residual policy; run base (pi05_libero) policy only")
    parser.add_argument("--proprio-frame", choices=("robot", "world"), default="world",
                        help="Frame for the residual proprio pose")
    parser.add_argument("--raw-proprio", action="store_true",
                        help="A/B control: skip the sim-convention proprio correction")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda/cpu) for the residual policy")
    parser.add_argument("--home-pose-name", default=fc.default_home_pose_name(),
                        help=f"Name of a saved pose JSON in {_POSES_DIR} (overrides --home-q)")
    parser.add_argument("--home-q", nargs=7, type=float, default=None,
                        help="7 joint angles (rad) overriding the saved home pose")
    parser.add_argument("--home-gripper", type=float, default=fc.control("homing.gripper_norm"))
    parser.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    parser.add_argument("--home-tol-rad", type=float, default=fc.control("homing.tol_rad"))

    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=fc.control_fps())
    parser.add_argument("--push-to-hub", type=_str2bool, default=True)
    parser.add_argument("--resume", type=_str2bool, default=False)
    parser.add_argument("--viz-dir", default=None)
    parser.add_argument("--viz-stride", type=int, default=1)

    args = parser.parse_args()

    if args.repo_id and not args.output_dir:
        parser.error("--output-dir is required when --repo-id is set")
    if args.save_videos and not args.viz_dir:
        parser.error("--viz-dir is required when --save-videos is set")
    if args.repo_id and not args.task:
        parser.error("--task is required when --repo-id is set")

    logging.basicConfig(level=logging.INFO, force=True)

    if args.home_q is not None:
        home_q = np.asarray(args.home_q, dtype=np.float64)
        home_gripper = args.home_gripper
    else:
        pose = fc.load_home_pose(args.home_pose_name)
        home_q = _default_home_q(args.home_pose_name)
        home_gripper = float(pose.get("gripper", args.home_gripper))

    print("attempting connection to robot...")
    controller = env_wrapper.start_controller()
    print("robot initialized!")

    print(f"attempting to connect to pi05_libero server at {args.openpi_host}:{args.openpi_port}")
    base_policy = OpenPIBasePolicy(
        host=args.openpi_host,
        port=args.openpi_port,
        prompt=args.prompt,
        image_key=args.image_key,
        wrist_image_key=args.wrist_image_key,
        default_kp=args.base_kp,
        default_kd=args.base_kd,
    )
    print("base policy (pi05_libero, remote) connected!")

    residual: "ResidualPolicy | None" = None
    if args.no_residual:
        print("residual policy disabled (--no-residual)")
    else:
        print(f"attempting to start residual policy: {args.residual_policy}")
        residual = ResidualPolicy(args.residual_policy, device=args.device)
        print("residual policy started")

    dump_root = None
    if args.dump_obs_dir:
        if residual is None:
            print("--dump-obs-dir ignored: residual_obs only exists with a residual policy")
        else:
            import hashlib
            import json as _json
            dump_root = Path(args.dump_obs_dir).expanduser() / time.strftime("%Y%m%d_%H%M%S")
            dump_root.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with open(args.residual_policy, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            (dump_root / "meta.json").write_text(_json.dumps({
                "residual_policy": str(Path(args.residual_policy).resolve()),
                "residual_policy_sha256": h.hexdigest(),
                "base_policy": f"openpi:pi05_libero@{args.openpi_host}:{args.openpi_port}",
                "prompt": args.prompt,
                "proprio_frame": args.proprio_frame,
                "raw_proprio": bool(args.raw_proprio),
                "fps": args.fps,
                "argv": sys.argv,
            }, indent=2))
            print(f"dumping residual obs bundles to {dump_root}")

    home_kwargs = dict(
        home_q_left=None,
        home_q_right=home_q,
        gripper_norm=home_gripper,
        max_time_s=args.home_max_time_s,
        tol_rad=args.home_tol_rad,
    )

    recording = args.repo_id is not None
    dataset = None

    if not recording:
        print("homing...")
        if not controller.home(**home_kwargs):
            logger.warning("homing did not converge; proceeding anyway")
        from viz import EpisodeRecorder
        recorder = EpisodeRecorder() if args.viz_dir else None
        try:
            _run_episode(
                controller, base_policy, residual,
                dataset=None, episode_time_s=None,
                fps=args.fps, recorder=recorder,
                replaying=False,
                proprio_frame=args.proprio_frame,
                sim_proprio_convention=not args.raw_proprio,
                dump_dir=dump_root / "ep000" if dump_root else None,
                video_dir=Path(args.viz_dir) if args.save_videos else None,
                video_cams=args.video_cams,
            )
        finally:
            if recorder is not None and len(recorder) > 0:
                viz_path = os.path.join(args.viz_dir, "episode.html")
                print(f"saving visualization to {viz_path}...")
                _save_viz(recorder, viz_path, residual, "episode (free run)", args.viz_stride, args.fps)
            controller.disconnect()
        return

    dataset = _build_dataset(args, controller)
    try:
        with VideoEncodingManager(dataset):
            print(f"homing before episode {dataset.num_episodes}...")
            if not controller.home(**home_kwargs):
                logger.warning("homing did not converge; proceeding anyway")

            from viz import EpisodeRecorder
            for ep_idx in range(args.num_episodes):
                print(f"press right arrow to start episode {dataset.num_episodes} / {args.num_episodes} "
                      f"({args.episode_time_s:.0f}s)...")
                _wait_for_right_arrow()

                print(f"recording episode {dataset.num_episodes} / {args.num_episodes} "
                      f"({args.episode_time_s:.0f}s)...")
                recorder = EpisodeRecorder() if args.viz_dir else None
                try:
                    _run_episode(
                        controller, base_policy, residual,
                        dataset=dataset,
                        episode_time_s=args.episode_time_s,
                        fps=args.fps,
                        task=args.task,
                        recorder=recorder,
                        replaying=False,
                        proprio_frame=args.proprio_frame,
                        sim_proprio_convention=not args.raw_proprio,
                        dump_dir=dump_root / f"ep{dataset.num_episodes:03d}" if dump_root else None,
                        video_dir=Path(args.viz_dir) if args.save_videos else None,
                        video_cams=args.video_cams,
                        video_stem=f"episode_{ep_idx:03d}",
                    )
                finally:
                    if recorder is not None and len(recorder) > 0:
                        viz_path = os.path.join(args.viz_dir, f"episode_{ep_idx:03d}.html")
                        print(f"saving visualization to {viz_path}...")
                        _save_viz(recorder, viz_path, residual, f"episode {ep_idx} — {args.task}", args.viz_stride, args.fps)
                dataset.save_episode()
                print(f"episode {dataset.num_episodes - 1} saved")

                if ep_idx < args.num_episodes - 1:
                    print("resetting environment — homing arm before next episode...")
                    if not controller.home(**home_kwargs):
                        logger.warning("homing did not converge; proceeding anyway")
    finally:
        if dataset is not None:
            dataset.finalize()
            if args.push_to_hub:
                try:
                    dataset.push_to_hub()
                except Exception:
                    logger.exception(
                        "push_to_hub failed; dataset is on disk at %s",
                        Path(args.output_dir).resolve(),
                    )
        controller.disconnect()


if __name__ == "__main__":
    main()
