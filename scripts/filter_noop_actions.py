#!/usr/bin/env python
"""
Filter out no-op / low-magnitude action frames from a LeRobot (v0.5.1, dataset format v3)
dataset and reupload the result under a new repo_id on the Hugging Face Hub.

Action schema (10-dim, all already deltas):
    [r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw, r_gripper, kp, kd]
    - r_x, r_y, r_z:      translation delta
    - r_qx..r_qw:         rotation delta as a quaternion (identity = [0,0,0,1])
    - r_gripper:          gripper delta
    - kp, kd:             always 0, excluded from magnitude computation

Usage:
    python filter_noop_actions.py \
        --source-repo-id my-org/my-dataset \
        --target-repo-id my-username/my-dataset_filtered \
        --translation-threshold 0.001 \
        --rotation-threshold 0.01 \
        --grip-threshold 0.01 \
        --push-to-hub

Notes:
- A frame is dropped only if translation, rotation, AND gripper deltas are
  ALL below their respective thresholds.
- Rotation magnitude is the angle of the delta quaternion (2*acos(|w|)),
  not its raw vector norm, since identity rotation is [0,0,0,1].
- Episodes that end up empty after filtering are skipped entirely.
- LeRobotDataset v3 requires monotonically increasing timestamps per
  episode; we don't splice timestamps around dropped frames, we just don't
  add them. add_frame()/save_episode() regenerate valid contiguous
  timestamps from fps for the frames that ARE kept.
- All non-action features (images, states, etc.) are copied through
  unchanged for kept frames.

Requires: lerobot==0.5.1, torch, tqdm
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRANSLATION_SLICE = slice(0, 3)   # r_x, r_y, r_z
QUAT_SLICE = slice(3, 7)          # r_qx, r_qy, r_qz, r_qw
GRIP_SLICE = slice(7, 8)          # r_gripper
# indices 8, 9 (kp, kd) are always 0 and intentionally excluded


def compute_action_magnitude(action: torch.Tensor) -> tuple[float, float, float]:
    """
    Returns (translation_mag, rotation_angle_rad, grip_mag) for a single
    action vector shaped [r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw, r_gripper, kp, kd].
    """
    action = action.float()
    translation = action[TRANSLATION_SLICE]
    quat = action[QUAT_SLICE]
    grip = action[GRIP_SLICE]

    translation_mag = torch.linalg.vector_norm(translation).item()
    grip_mag = torch.linalg.vector_norm(grip).item()

    quat_norm = quat.norm().clamp_min(1e-8)
    quat = quat / quat_norm
    w = quat[3].clamp(-1.0, 1.0)
    rotation_angle = 2.0 * torch.acos(w.abs()).item()

    return translation_mag, rotation_angle, grip_mag


def build_target_features(source_meta) -> dict:
    """
    Reconstruct the features dict expected by LeRobotDataset.create() from an
    existing dataset's metadata, stripping fields that create() derives itself
    (frame_index, episode_index, index, task_index, timestamp) since those are
    injected automatically by add_frame()/save_episode(). "task" is also
    excluded here since it's passed inside the frame dict separately, not
    copied verbatim as a feature.
    """
    auto_fields = {"frame_index", "episode_index", "index", "task_index", "timestamp", "task"}
    features = {}
    for key, spec in source_meta.features.items():
        if key in auto_fields:
            continue
        features[key] = dict(spec)
    return features


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-repo-id", required=True,
                         help="repo_id of the dataset to load from the Hub")
    parser.add_argument("--target-repo-id", required=True,
                         help="repo_id to push the filtered dataset to")
    parser.add_argument("--action-key", default="action",
                         help="Name of the action feature in the dataset (default: 'action')")
    parser.add_argument("--translation-threshold", type=float, default=0.001,
                         help="Minimum translation-delta norm to keep a frame (dataset's native length unit)")
    parser.add_argument("--rotation-threshold", type=float, default=0.01,
                         help="Minimum rotation-delta angle (radians) to keep a frame")
    parser.add_argument("--grip-threshold", type=float, default=0.01,
                         help="Minimum gripper-delta magnitude to keep a frame")
    parser.add_argument("--root", default=None, help="Optional local cache root for the source dataset")
    parser.add_argument("--target-root", default=None,
                         help="Optional local root to build the new dataset in before pushing")
    parser.add_argument("--push-to-hub", action="store_true",
                         help="Push the filtered dataset to the Hub when done")
    parser.add_argument("--private", action="store_true", help="Push as a private repo")
    parser.add_argument("--max-episodes", type=int, default=None,
                         help="Optional cap on number of source episodes to process (for testing)")
    args = parser.parse_args()

    logger.info("Loading source dataset '%s' from the Hub...", args.source_repo_id)
    source_ds = LeRobotDataset(args.source_repo_id, root=args.root)
    source_meta = source_ds.meta

    if args.action_key not in source_meta.features:
        raise ValueError(
            f"Action key '{args.action_key}' not found in dataset features: {list(source_meta.features)}"
        )

    action_names = source_meta.features[args.action_key].get("names")
    expected_names = ["r_x", "r_y", "r_z", "r_qx", "r_qy", "r_qz", "r_qw", "r_gripper", "kp", "kd"]
    if action_names is not None and list(action_names) != expected_names:
        logger.warning(
            "Action feature names %s don't match the expected schema %s this script was "
            "written for. Slicing indices may be wrong — double check before trusting results.",
            action_names, expected_names,
        )

    features = build_target_features(source_meta)

    target_root = Path(args.target_root) if args.target_root else None
    if target_root and target_root.exists():
        logger.warning("Target root %s already exists; removing it first.", target_root)
        shutil.rmtree(target_root)

    logger.info("Creating target dataset '%s'...", args.target_repo_id)
    target_ds = LeRobotDataset.create(
        repo_id=args.target_repo_id,
        fps=source_meta.fps,
        root=target_root,
        robot_type=getattr(source_meta, "robot_type", None),
        features=features,
        use_videos=len(source_meta.video_keys) > 0,
    )

    num_source_episodes = source_meta.total_episodes
    if args.max_episodes is not None:
        num_source_episodes = min(num_source_episodes, args.max_episodes)

    total_kept = 0
    total_dropped = 0
    episodes_written = 0
    episodes_skipped_empty = 0

    image_keys = set(source_meta.camera_keys)  # covers both video and image features

    for ep_idx in tqdm(range(num_source_episodes), desc="Filtering episodes"):
        ep_start = int(source_ds.meta.episodes["dataset_from_index"][ep_idx])
        ep_end = int(source_ds.meta.episodes["dataset_to_index"][ep_idx])

        kept_any = False

        for frame_idx in range(ep_start, ep_end):
            sample = source_ds[frame_idx]
            action = sample[args.action_key]

            translation_mag, rotation_angle, grip_mag = compute_action_magnitude(action)
            is_noop = (
                translation_mag < args.translation_threshold
                and rotation_angle < args.rotation_threshold
                and grip_mag < args.grip_threshold
            )
            if is_noop:
                total_dropped += 1
                continue

            frame = {}
            for key in features:
                value = sample[key]
                if key in image_keys and isinstance(value, torch.Tensor) and value.ndim == 3:
                    # __getitem__ returns images as (C, H, W); add_frame expects (H, W, C)
                    value = value.permute(1, 2, 0)
                frame[key] = value
            frame["task"] = sample.get("task", "")

            target_ds.add_frame(frame)
            kept_any = True
            total_kept += 1

        if kept_any:
            target_ds.save_episode()
            episodes_written += 1
        else:
            episodes_skipped_empty += 1
            logger.info("Episode %d dropped entirely (all frames below threshold).", ep_idx)

    logger.info("Finalizing dataset...")
    target_ds.finalize()

    logger.info(
        "Done. Kept %d frames, dropped %d frames (%.1f%% dropped). "
        "Wrote %d episodes, skipped %d empty episodes.",
        total_kept,
        total_dropped,
        100.0 * total_dropped / max(total_kept + total_dropped, 1),
        episodes_written,
        episodes_skipped_empty,
    )

    if args.push_to_hub:
        logger.info("Pushing '%s' to the Hub...", args.target_repo_id)
        target_ds.push_to_hub(private=args.private)
        logger.info("Push complete: https://huggingface.co/datasets/%s", args.target_repo_id)
    else:
        logger.info("Skipping push (pass --push-to-hub to upload). Local dataset root: %s", target_ds.root)


if __name__ == "__main__":
    main()