#!/usr/bin/env python3
"""Convert a regenerated LIBERO demo file into the SAIL and B-Spline training
HDF5s and our diffusion base's LeRobot dataset.

    python -m baselines.libero_bridge.dataset ~/libero_data_sysid/libero_90 \
        --out-dir ~/franka_data/baseline_prep/libero_90_sysid_2026_09_16

All three come from the same demos, episodes and frames; only what each policy
reads and predicts differs. Each task is filed by its index in the suite, not
its name: task 2 of libero_10 lands in `<out-dir>/task_2/` and is stamped
`libero_10/task_2`, which is where its policies and rollouts go too. A file
regenerate_on_plant.py re-recorded is stamped `<suite>_<plant>/task_2` instead,
so it is never filed beside the stock-plant demos. The suite is the nearest
directory in the source's path named after one, or --suite.

The sim half of scripts/prepare_baseline_datasets.py. It reads what
`multi-fast/scripts/libero/regenerate_libero_dataset.py` writes, not raw LIBERO:
the no-op transitions and failed demos are already gone, and `goal_pos` /
`goal_ori` -- the OSC controller's own absolute target for each step -- are
already recorded. That is why this is a re-key rather than a replay. The
reached/commanded split the real converters reconstruct by hand
(baselines/README.md) is already in the file:

    obs/ee_pos,  obs/ee_ori   where the arm WAS        -> the observation
    goal_pos,    goal_ori     where it was TOLD to go  -> B-Spline's target

Field list and why the fields are what they are: baselines/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from baselines.common import parse_image_size, write_demo_file  # noqa: E402
from baselines.run_record import read_source_repo_id  # noqa: E402

logger = logging.getLogger("baselines.libero")

# The action layout, the camera map, the benchmark resolution and the 180-degree
# flip live in sim_env: the rollout has to reproduce this frame exactly, and a
# definition here that the rollout re-derives is how the two drift. That module
# is numpy + cv2 at import time, so it costs nothing here.
from baselines.libero_bridge.sim_env import (  # noqa: E402
    CAMERAS, DEFAULT_IMAGE_SIZE, DP_STATE_KEYS, GRIPPER, policy_frame, resolve_task, suites,
    task_names, task_tag,
)

# What convert_task writes per task, in order.
KINDS = ("sail", "bspline", "lerobot")


def suite_of(source: Path) -> str:
    """The nearest directory in `source`'s path named after a LIBERO suite.

    Not simply the parent: `libero_10/libero_10_relabeled/<task>_demo.hdf5` is
    libero_10, and an index means nothing without the right suite.
    """
    path = source.expanduser().absolute()
    known = suites()
    for d in (path, *path.parents):
        if d.name in known:
            return d.name
    raise SystemExit(f"no LIBERO suite ({', '.join(known)}) in the path {path}; pass --suite")


def images(demo, image_size: tuple[int, int] | None, flip: bool) -> dict[str, np.ndarray]:
    out = {}
    for src, dst in CAMERAS:
        frames = demo[f"obs/{src}"][()]
        out[f"obs/{dst}"] = np.stack([policy_frame(f, image_size, flip) for f in frames])
    return out


def poses(demo):
    """-> (reached_pos, reached_rotvec, reached_quat_xyzw, commanded_pos, commanded_rotvec).

    The stored axis-angles are used as rotation vectors **unchanged**. Unlike the
    real converters, which run `common.canonical_rotvec` to pick one vector per
    rotation, nothing is re-derived here: robosuite's `quat2axisangle` is
    `2*acos(w)` times the quaternion's own axis, so theta runs to 2*pi and the
    axis never flips at pi. Measured on libero_90: 0 axis flips in 5565 frames,
    theta in [2.91, 3.35]. Round-tripping through `Rotation.as_rotvec`, which
    folds theta back into [0, pi], is what would INTRODUCE the discontinuity the
    real side has to correct for.
    """
    reached_pos = demo["obs/ee_pos"][()].astype(np.float64)
    reached_rotvec = demo["obs/ee_ori"][()].astype(np.float64)
    return (
        reached_pos,
        reached_rotvec,
        # Reproduces mujoco's own robot0_eef_quat exactly (signed dot 1.0 against
        # the stored robot_states), so the training observation is bit-for-bit
        # what the sim env reports at rollout.
        Rotation.from_rotvec(reached_rotvec).as_quat(),
        demo["goal_pos"][()].astype(np.float64),
        demo["goal_ori"][()].astype(np.float64),
    )


def pose_rows(pos, rotvec, gripper) -> np.ndarray:
    """(N,3) + (N,3) + (N,) -> (N,7), SAIL/B-Spline's shared pos+rotvec+gripper layout.

    `common.pos_rotvec_gripper` takes a quaternion and canonicalises; see `poses`
    for why neither applies to a rotation vector that is already continuous.
    """
    return np.concatenate([pos, rotvec, gripper[:, None]], axis=1).astype(np.float32)


def sail_episode(demo) -> dict[str, np.ndarray]:
    reached_pos, reached_rotvec, reached_quat, commanded_pos, commanded_rotvec = poses(demo)
    actions = demo["actions"][()].astype(np.float64)
    gripper_act = actions[:, GRIPPER]
    return {
        "obs/robot0_eef_pos": reached_pos.astype("float32"),
        "obs/robot0_eef_quat": reached_quat.astype("float32"),
        # Not in the training template's low_dim list; SAIL's AWE waypoint pass
        # reads it, which is why the real converter emits it too.
        "obs/robot0_joint_pos": demo["obs/joint_states"][()].astype("float32"),
        # Two finger joints, robosuite's native width. The real rig reports one
        # normalised opening, so this column is 2 wide here and 1 there.
        "obs/robot0_gripper_qpos": demo["obs/gripper_states"][()].astype("float32"),
        "actions": actions.astype("float32"),
        "absolute_actions": pose_rows(reached_pos, reached_rotvec, gripper_act),
        "commanded_absolute_actions": pose_rows(commanded_pos, commanded_rotvec, gripper_act),
    }


def bspline_episode(demo) -> dict[str, np.ndarray]:
    reached_pos, _, reached_quat, commanded_pos, commanded_rotvec = poses(demo)
    gripper_act = demo["actions"][()][:, GRIPPER].astype(np.float64)
    return {
        # obs is where the arm was, actions is where it was told to go -- using
        # the same pose for both would make the target the policy's own input.
        "obs/robot0_eef_pos": reached_pos.astype("float32"),
        "obs/robot0_eef_quat": reached_quat.astype("float32"),
        "obs/robot0_gripper_qpos": demo["obs/gripper_states"][()].astype("float32"),
        "actions": pose_rows(commanded_pos, commanded_rotvec, gripper_act),
    }


# Our diffusion base's feature names; the action is the real base's EE_POS less kp/kd.
LEROBOT_STATE = ("eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw",
                 "gripper_qpos_0", "gripper_qpos_1")
LEROBOT_ACTION = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")


def lerobot_episode(demo) -> dict[str, np.ndarray]:
    reached_pos, _, reached_quat, commanded_pos, commanded_rotvec = poses(demo)
    gripper_act = demo["actions"][()][:, GRIPPER].astype(np.float64)
    # On the state's hemisphere: from_rotvec keeps the relabeler's sign (see poses()).
    commanded_quat = Rotation.from_rotvec(commanded_rotvec).as_quat()
    state = {"robot0_eef_pos": reached_pos, "robot0_eef_quat": reached_quat,
             "robot0_gripper_qpos": demo["obs/gripper_states"][()]}
    return {
        "observation.state": np.concatenate(
            [state[k] for k, _ in DP_STATE_KEYS], axis=1).astype("float32"),
        "action": np.concatenate(
            [commanded_pos, commanded_quat, gripper_act[:, None]], axis=1).astype("float32"),
    }


def demos(source: Path, build, *, episodes, min_steps, include_images, image_size, flip):
    with h5py.File(source, "r") as f:
        data = f["data"]
        for name in sorted(data, key=lambda k: int(k.split("_")[1])):
            # LIBERO's indices are non-contiguous: regenerate_libero_dataset.py
            # names each group after the ORIGINAL demo and only creates it when
            # the replay succeeded. write_demo_file renumbers.
            ep_index = int(name.split("_")[1])
            if episodes is not None and ep_index not in episodes:
                continue
            demo = data[name]
            steps = len(demo["actions"])
            if steps < min_steps:
                logger.warning("ep%03d: %d steps, below --min-steps %d; skipped",
                               ep_index, steps, min_steps)
                continue
            arrays = build(demo)
            if include_images:
                arrays.update(images(demo, image_size, flip))
            yield ep_index, arrays


def default_repo_id(source: Path, suite: str, index: int) -> str:
    """`<suite>/task_<i>`, or `<suite>_<plant>/task_<i>` for a re-recorded file."""
    stamp = comparison_env(source)
    prefix = suite if stamp is None else f"{suite}_{json.loads(stamp)['plant']}"
    return f"{prefix}/{task_tag(index)}"


def comparison_env(source: Path) -> str | None:
    """The environment regenerate_on_plant.py recorded the demos in, as its JSON
    stamp; None for a stock-plant regenerate_libero_dataset.py file."""
    with h5py.File(source, "r") as f:
        stamp = f["data"].attrs.get("comparison_env")
    return None if stamp is None else str(stamp)


def convert_lerobot(source, root: Path | None, *, suite: str, episodes=None, min_steps=20,
                    include_images=True, image_size=DEFAULT_IMAGE_SIZE, flip_images=True,
                    source_repo_id=None) -> int:
    """The diffusion base's LeRobot dataset, from the same demos and frames as
    sail.hdf5 and bspline.hdf5.

    Images are stored as `image` (lossless PNG), not video, so the base trains on
    the pixels the baselines do. `source.json` at the root records what
    LeRobot's own metadata cannot: the source file, the environment stamp and
    which source demo each episode is.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from libero.libero import benchmark

    source = Path(source).expanduser()
    index = resolve_task(suite, source.stem)
    repo_id = source_repo_id or default_repo_id(source, suite, index)
    root = Path(root).expanduser() if root is not None else HF_LEROBOT_HOME / repo_id
    if root.exists():
        raise SystemExit(f"{root} exists; delete it to convert over it")
    language = benchmark.get_benchmark_dict()[suite]().get_task(index).language

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(LEROBOT_STATE),),
                              "names": list(LEROBOT_STATE)},
        "action": {"dtype": "float32", "shape": (len(LEROBOT_ACTION),),
                   "names": list(LEROBOT_ACTION)},
    }
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    dataset = None
    episode_map = []
    try:
        for ep_index, arrays in demos(source, lerobot_episode, episodes=episodes,
                                      min_steps=min_steps, include_images=include_images,
                                      image_size=image_size, flip=flip_images):
            images_ = {f"observation.images.{dst}": arrays.pop(f"obs/{dst}")
                       for _, dst in CAMERAS if f"obs/{dst}" in arrays}
            if dataset is None:
                for key, frames in images_.items():
                    features[key] = {"dtype": "image", "shape": tuple(frames.shape[1:]),
                                     "names": ["height", "width", "channel"]}
                dataset = LeRobotDataset.create(repo_id, fps=20, features=features, root=tmp,
                                                robot_type="libero_panda", use_videos=False)
            for t in range(len(arrays["action"])):
                frame = {k: v[t] for k, v in arrays.items()}
                frame.update({k: v[t] for k, v in images_.items()})
                frame["task"] = language
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_map.append(ep_index)
        if dataset is None:
            logger.error("no episodes converted")
            return 1
        dataset.finalize()
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    with h5py.File(source, "r") as f:
        first = next(iter(f["data"].values()))
        # What the frames were rendered at before policy_frame downsampled them.
        rendered = int(first[f"obs/{CAMERAS[0][0]}"].shape[1])
    (tmp / "source.json").write_text(json.dumps({
        "converter": "baselines/libero_bridge/dataset.py:lerobot",
        "source_dataset": str(source),
        "source_repo_id": repo_id,
        "libero_suite": suite, "libero_task_index": index,
        "libero_task": task_names(suite)[index],
        "comparison_env": json.loads(comparison_env(source) or "null"),
        "render_resolution": rendered,
        "source_episodes": episode_map,
    }, indent=1))
    tmp.rename(root)
    logger.info("wrote %d episodes to %s", len(episode_map), root)
    return 0


def convert(source, out: Path, kind: str, *, suite: str, episodes=None, min_steps=20,
            include_images=True, image_size=DEFAULT_IMAGE_SIZE, flip_images=True,
            source_repo_id=None) -> int:
    source = Path(source).expanduser()
    build = {"sail": sail_episode, "bspline": bspline_episode}[kind]
    index = resolve_task(suite, source.stem)
    name = task_names(suite)[index]
    # Checkpoints and rollouts are filed under this id: `<root>/libero_10/task_2/`,
    # one suite per directory (baselines/run_record.py:_safe_segment).
    repo_id = source_repo_id or default_repo_id(source, suite, index)
    # An index repeats across suites, so a task_<i>/ already stamped with another
    # id is a wrong --out-dir, not a file to replace.
    was = read_source_repo_id(out)
    if was is not None and was != repo_id:
        raise SystemExit(f"{out} holds {was}, not {repo_id}. Wrong --out-dir? "
                         f"Delete it to convert over it.")
    root_attrs = {"libero_task": name, "libero_suite": suite, "libero_task_index": index}
    env_stamp = comparison_env(source)
    if env_stamp is not None:
        root_attrs["comparison_env"] = env_stamp
    if kind == "sail":
        # robomimic's train.py loads env_args before it checks whether rollouts
        # are on. Type 6 is EnvType.REAL_TYPE, so is_robosuite_env() stays False
        # and training never tries to build a simulator -- the LIBERO rollout is
        # driven from outside robomimic, in multi-fast's venv.
        root_attrs["env_args"] = json.dumps({"env_name": name, "type": 6, "env_kwargs": {}})
        root_attrs["rotvec_hemisphere"] = "as stored by robosuite quat2axisangle (see poses())"
    return write_demo_file(
        out,
        demos(source, build, episodes=episodes, min_steps=min_steps,
              include_images=include_images, image_size=image_size, flip=flip_images),
        f"baselines/libero_bridge/dataset.py:{kind}",
        source_dataset=source,
        source_repo_id=repo_id,
        root_attrs=root_attrs,
    )


def convert_task(source: Path, out_dir: Path, *, suite: str, skip=(), lerobot_root=None,
                 **kwargs) -> int:
    """All three training sets for one task: sail.hdf5 and bspline.hdf5 into
    `out_dir`, the LeRobot dataset at `lerobot_root` (default the HF cache).

    The directory's name matters a little: bspline_bridge/train.py takes the
    hydra task name, and so the wandb tag, from the parent directory of the HDF5.
    """
    index = resolve_task(suite, source.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for kind in KINDS:
        if kind in skip:
            continue
        logger.info("=== %s: %s/%s, %s ===", kind, suite, task_tag(index),
                    task_names(suite)[index])
        if kind == "lerobot":
            rc = convert_lerobot(source, lerobot_root, suite=suite, **kwargs)
        else:
            rc = convert(source, out_dir / f"{kind}.hdf5", kind, suite=suite, **kwargs)
        if rc != 0:
            failed.append(kind)
    if failed:
        logger.error("%s: failed %s", source.name, ", ".join(failed))
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path,
                   help="a regenerated LIBERO <task>_demo.hdf5, or a suite directory of them")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="where sail.hdf5 and bspline.hdf5 go. A suite directory source "
                        "gets one task_<i>/ subdirectory per task, <i> its index")
    p.add_argument("--suite", default=None,
                   help="the LIBERO suite the source is from, which is what its task "
                        "indices mean. Default: the nearest directory in its path "
                        "named after one")
    p.add_argument("--episodes", default=None, help="comma-separated demo indices, default all")
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--no-images", action="store_true", help="skip camera frames (schema checks)")
    p.add_argument("--image-size", default="%dx%d" % DEFAULT_IMAGE_SIZE,
                   help="WxH to resize camera frames to, or `native` to keep the "
                        "source's. The file's size is what both policies train "
                        "and roll out at")
    p.add_argument("--no-flip-images", action="store_true",
                   help="keep the renderer's upside-down frames; only for comparing "
                        "against something else that did not flip")
    p.add_argument("--source-repo-id", default=None,
                   help="what the file is stamped with, and so where checkpoints and "
                        "rollouts are filed. Default <suite>/task_<i>, or <suite>_<plant>/task_<i> "
                        "for a regenerate_on_plant.py file")
    p.add_argument("--skip", nargs="*", choices=list(KINDS), default=[])
    p.add_argument("--lerobot-root", type=Path, default=None,
                   help="where the LeRobot dataset goes, or for a suite directory where its "
                        "task_<i>/ datasets go; default $HF_LEROBOT_HOME/<source-repo-id>, "
                        "where lerobot-train and baselines/tillicum/sync.sh look for it")
    args = p.parse_args()

    # force: importing franka_config installs a root handler, which makes a
    # plain basicConfig a silent no-op and swallows every progress line.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    episodes = {int(x) for x in args.episodes.split(",")} if args.episodes else None
    kwargs = dict(episodes=episodes, min_steps=args.min_steps,
                  include_images=not args.no_images,
                  image_size=None if args.image_size == "native" else parse_image_size(args.image_size),
                  flip_images=not args.no_flip_images, source_repo_id=args.source_repo_id)

    if not args.source.exists():
        p.error(f"{args.source} does not exist")
    suite = args.suite or suite_of(args.source)

    if args.source.is_dir():
        tasks = sorted(args.source.glob("*.hdf5"))
        if not tasks:
            p.error(f"no .hdf5 files in {args.source}")
        if args.source_repo_id:
            p.error("--source-repo-id names one task; it cannot be shared across a suite")
        names = task_names(suite)
        stray = [t.name for t in tasks if t.stem.removesuffix("_demo") not in names]
        if stray:
            p.error(f"not tasks of {suite}: {', '.join(stray)}. Pass --suite, or move them out")
        index = {t: resolve_task(suite, t.stem) for t in tasks}
        failed = []
        for task in sorted(tasks, key=index.get):
            tag = task_tag(index[task])
            try:
                rc = convert_task(task, args.out_dir / tag, suite=suite, skip=args.skip,
                                  lerobot_root=(None if args.lerobot_root is None
                                                else args.lerobot_root / tag),
                                  **kwargs)
            except OSError as exc:
                # An interrupted regeneration run leaves a truncated file. Say so
                # and keep going; the other tasks are still usable.
                logger.error("%s: unreadable, skipped (%s)", task.name, exc)
                rc = 1
            if rc != 0:
                failed.append(f"{tag} ({task.name})")
        logger.info("converted %d of %d %s tasks into %s", len(tasks) - len(failed), len(tasks),
                    suite, args.out_dir)
        if failed:
            logger.error("failed: %s", ", ".join(failed))
            return 1
        return 0

    return convert_task(args.source, args.out_dir, suite=suite, skip=args.skip,
                        lerobot_root=args.lerobot_root, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
