#!/usr/bin/env python
"""Drop the no-op frames from a recorded LeRobot dataset and write the rest as a
new dataset (optionally pushed to the Hub).

What counts as a no-op depends on the action space, which is classified by
reach (`lerobot_source.action_space`) since EE_DELTA and EE_POS recordings share
one feature schema:

  delta     the action IS the motion: a frame is a no-op when its translation,
            rotation and gripper deltas are all below threshold.
  ee_pose   the action is an absolute target (spacemouse_ee, gello_ee): a frame
            is a no-op when the target did not change from the PREVIOUS frame --
            translation, rotation and gripper all below threshold. The first
            frame of an episode has nothing to differ from and is always kept.
            A released SpaceMouse holds its target bitwise, so the defaults only
            need to clear float noise; raise them to also drop a creeping target
            (anti-windup pulling it back toward a lagging arm).

Thresholds are motion per frame in both modes: metres, radians (the angle of
the delta rotation, so the quaternion's double cover does not count), and the
normalised gripper unit. A frame survives if ANY of the three clears its
threshold. Episodes with no surviving frame are skipped.

Usage:
    python scripts/filter_noop_actions.py --source-repo-id <path or repo id> \\
        --target-repo-id <repo id> [--target-root DIR] [--mode auto|delta|ee_pose] \\
        [--translation-threshold M] [--rotation-threshold RAD] [--grip-threshold U] \\
        [--dry-run] [--push-to-hub]

`--dry-run` reads only the parquet (no video decoding) and prints what each
episode would keep -- the way to pick thresholds before writing anything.

Kept frames are copied through with every feature; add_frame()/save_episode()
regenerate contiguous timestamps, so the fps of the result is nominal and the
gaps where frames were dropped are not represented.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot_robot_bimanual_franka.lerobot_source import (  # noqa: E402
    GRIPPER, POS, QUAT, action_space, load_frames, resolve_root,
)

logger = logging.getLogger(__name__)

# (translation m, rotation rad, gripper) per mode: the delta ones are the values
# this script has always used; ee_pose's are "unchanged" up to float noise.
DEFAULT_THRESHOLDS = {
    "delta": (0.0005, 0.01, 0.01),
    "ee_pose": (1e-4, 1e-3, 1e-3),
}


def rotation_angle(quat_xyzw: np.ndarray, ref_xyzw: np.ndarray | None = None) -> float:
    """Angle of `quat` relative to `ref` (identity when None), sign-insensitive."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    if ref_xyzw is None:
        w = abs(float(q[3]))
    else:
        r = np.asarray(ref_xyzw, dtype=np.float64)
        r = r / max(float(np.linalg.norm(r)), 1e-12)
        w = abs(float(np.dot(q, r)))
    return 2.0 * float(np.arccos(min(w, 1.0)))


def motion(action: np.ndarray, prev: np.ndarray | None, mode: str) -> tuple[float, float, float]:
    """(translation, rotation angle, gripper) this frame commands. In delta mode
    that is the action; in ee_pose it is the change from `prev`."""
    a = np.asarray(action, dtype=np.float64)
    if mode == "delta":
        return (float(np.linalg.norm(a[POS])), rotation_angle(a[QUAT]), abs(float(a[GRIPPER])))
    p = np.asarray(prev, dtype=np.float64)
    return (float(np.linalg.norm(a[POS] - p[POS])), rotation_angle(a[QUAT], p[QUAT]),
            abs(float(a[GRIPPER] - p[GRIPPER])))


def keep_mask(actions: np.ndarray, mode: str, thresholds: tuple[float, float, float]) -> np.ndarray:
    """Which frames of ONE episode survive, in order."""
    t_pos, t_rot, t_grip = thresholds
    keep = np.ones(len(actions), dtype=bool)
    prev = None
    for i, a in enumerate(actions):
        if mode == "ee_pose" and prev is None:
            prev = a  # first frame: nothing to differ from
            continue
        dp, dr, dg = motion(a, prev, mode)
        keep[i] = dp >= t_pos or dr >= t_rot or dg >= t_grip
        prev = a
    return keep


def resolve_mode(requested: str, actions: np.ndarray) -> str:
    space = action_space(actions)
    classified = {"EE_DELTA": "delta", "EE_POS": "ee_pose"}[space]
    if requested != "auto" and requested != classified:
        raise SystemExit(
            f"--mode {requested} but the recording classifies as {space} by reach; "
            f"filtering it as {requested} would be meaningless.")
    return classified


def build_target_features(source_meta) -> dict:
    """The features dict LeRobotDataset.create() takes, minus what add_frame()/
    save_episode() derive themselves."""
    auto_fields = {"frame_index", "episode_index", "index", "task_index", "timestamp", "task"}
    return {k: dict(spec) for k, spec in source_meta.features.items() if k not in auto_fields}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-repo-id", required=True,
                        help="Local dataset path or repo id (looked up under ~/franka_data "
                             "and the HF cache first, then fetched from the Hub)")
    parser.add_argument("--target-repo-id", required=True,
                        help="repo_id of the filtered dataset")
    parser.add_argument("--mode", choices=("auto", "delta", "ee_pose"), default="auto",
                        help="Action space; auto classifies by reach, an explicit value "
                             "must agree with it (default: auto)")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--translation-threshold", type=float, default=None,
                        help="metres per frame (default per mode: "
                             f"delta {DEFAULT_THRESHOLDS['delta'][0]}, ee_pose {DEFAULT_THRESHOLDS['ee_pose'][0]})")
    parser.add_argument("--rotation-threshold", type=float, default=None,
                        help="radians per frame (default per mode: "
                             f"delta {DEFAULT_THRESHOLDS['delta'][1]}, ee_pose {DEFAULT_THRESHOLDS['ee_pose'][1]})")
    parser.add_argument("--grip-threshold", type=float, default=None,
                        help="normalised gripper units per frame (default per mode: "
                             f"delta {DEFAULT_THRESHOLDS['delta'][2]}, ee_pose {DEFAULT_THRESHOLDS['ee_pose'][2]})")
    parser.add_argument("--root", default=None,
                        help="Source dataset root (default: resolved from --source-repo-id)")
    parser.add_argument("--target-root", default=None,
                        help="Where to build the filtered dataset (default: LeRobot's cache)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report per-episode keep/drop counts from the parquet and exit")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true", help="Push as a private repo")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap on source episodes to process (for testing)")
    args = parser.parse_args()
    # force: importing the robot package already put a handler on the root logger.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    # Local first: the parquet is what classifies the action space and, for
    # --dry-run, all that is read.
    root = Path(args.root) if args.root else None
    if root is None:
        try:
            root = resolve_root(args.source_repo_id)
        except FileNotFoundError:
            logger.info("No local copy of %r; fetching from the Hub", args.source_repo_id)
    if root is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = Path(LeRobotDataset(args.source_repo_id).root)

    df, info = load_frames(root)
    if args.action_key not in info["features"]:
        raise SystemExit(f"no {args.action_key!r} feature; have {list(info['features'])}")
    all_actions = np.stack(df[args.action_key].to_numpy()).astype(np.float64)
    episode_of = df["episode_index"].to_numpy()
    mode = resolve_mode(args.mode, all_actions)

    defaults = DEFAULT_THRESHOLDS[mode]
    thresholds = (
        defaults[0] if args.translation_threshold is None else args.translation_threshold,
        defaults[1] if args.rotation_threshold is None else args.rotation_threshold,
        defaults[2] if args.grip_threshold is None else args.grip_threshold,
    )
    logger.info("%s: %d episodes, %d frames, mode %s, thresholds %.2e m / %.2e rad / %.2e grip",
                root, info["total_episodes"], len(df), mode, *thresholds)

    episode_ids = sorted(set(int(e) for e in episode_of))
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    masks: dict[int, np.ndarray] = {}
    for ep in episode_ids:
        order = np.flatnonzero(episode_of == ep)
        order = order[np.argsort(df["frame_index"].to_numpy()[order], kind="stable")]
        masks[ep] = keep_mask(all_actions[order], mode, thresholds)
        logger.info("ep%03d: keep %d / %d", ep, int(masks[ep].sum()), len(order))

    total = sum(len(m) for m in masks.values())
    kept = sum(int(m.sum()) for m in masks.values())
    empty = [ep for ep, m in masks.items() if not m.any()]
    logger.info("Would keep %d / %d frames (%.1f%% dropped); %d episode(s) would be empty%s",
                kept, total, 100.0 * (total - kept) / max(total, 1), len(empty),
                f": {empty}" if empty else "")
    if args.dry_run:
        return

    import torch  # noqa: F401  (LeRobotDataset returns tensors)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    source_ds = LeRobotDataset(args.source_repo_id, root=root)
    source_meta = source_ds.meta
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

    image_keys = set(source_meta.camera_keys)
    episodes_written = 0
    for ep in tqdm(episode_ids, desc="Filtering episodes"):
        mask = masks[ep]
        if not mask.any():
            logger.info("Episode %d dropped entirely (no frame clears the thresholds).", ep)
            continue
        ep_start = int(source_meta.episodes["dataset_from_index"][ep])
        ep_end = int(source_meta.episodes["dataset_to_index"][ep])
        assert ep_end - ep_start == len(mask), (ep, ep_end - ep_start, len(mask))

        for offset in np.flatnonzero(mask):
            sample = source_ds[ep_start + int(offset)]
            frame = {}
            for key in features:
                value = sample[key]
                if key in image_keys and hasattr(value, "ndim") and value.ndim == 3:
                    # __getitem__ returns images as (C, H, W); add_frame expects (H, W, C)
                    value = value.permute(1, 2, 0)
                frame[key] = value
            frame["task"] = sample.get("task", "")
            target_ds.add_frame(frame)
        target_ds.save_episode()
        episodes_written += 1

    logger.info("Finalizing dataset...")
    target_ds.finalize()
    logger.info("Done. Kept %d / %d frames in %d episodes. Local root: %s",
                kept, total, episodes_written, target_ds.root)

    if args.push_to_hub:
        logger.info("Pushing '%s' to the Hub...", args.target_repo_id)
        target_ds.push_to_hub(private=args.private)
        logger.info("Push complete: https://huggingface.co/datasets/%s", args.target_repo_id)
    else:
        logger.info("Skipping push (pass --push-to-hub to upload).")


if __name__ == "__main__":
    main()
