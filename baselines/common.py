"""Shared by both baseline dataset converters. See baselines/README.md."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import franka_config as fc  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.lerobot_source import (  # noqa: E402
    POS,
    QUAT,
    arm_prefix,
    check_action_space,
    load_frames,
    resolve_root,
)
from lerobot_robot_bimanual_franka.safety import ActionSafetyScreen  # noqa: E402
from sysid.lerobot_to_hdf5 import leading_trim  # noqa: E402

logger = logging.getLogger("baselines.common")

NUM_JOINTS = fc.num_joints()


def make_safety_screen(info: dict, arm: str) -> ActionSafetyScreen:
    arm_name = fc.profile(info["robot_type"]).arms[arm]
    return ActionSafetyScreen(
        {arm: fc.robot_base_in_world(arm_name)}, {arm: fc.ee_sphere(arm_name)}
    )


def reached_and_commanded_poses(
    qpos: np.ndarray, actions: np.ndarray, arm: str, screen: ActionSafetyScreen,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (reached_pos, reached_quat_xyzw, commanded_pos, commanded_quat_xyzw).

    reached is frame i's own measured pose, NOT a next-step lookahead.
    """
    reached_pos, reached_quat = eef_poses_from_qpos(qpos)

    commanded_pos = np.empty((len(actions), 3))
    commanded_quat = np.empty((len(actions), 4))
    for t, a in enumerate(actions):
        quat = a[QUAT] / max(float(np.linalg.norm(a[QUAT])), 1e-12)
        commanded_pos[t], commanded_quat[t] = screen.shape_goal({arm: (a[POS], quat)})[arm]
    return reached_pos, reached_quat, commanded_pos, commanded_quat


def home_rotvec_axis(arm: str) -> np.ndarray:
    """Unit axis of the home EE orientation's rotation vector, for `canonical_rotvec`."""
    _, quat = eef_poses_from_qpos(fc.home_q(key=arm)[None])
    rotvec = Rotation.from_quat(quat[0]).as_rotvec()
    return rotvec / np.linalg.norm(rotvec)


def canonical_rotvec(rotvec: np.ndarray, axis_ref: np.ndarray) -> np.ndarray:
    """One rotation vector per rotation: the one whose axis is on `axis_ref`'s side.

    A rotation has two rotation vectors, theta*a and (2pi-theta)*(-a), and
    `as_rotvec` returns whichever has theta <= pi. The gripper-down orientation
    this rig works at is a rotation by almost exactly pi, so a roll of a few
    degrees moves theta across pi and the returned vector jumps from ~+pi*a to
    ~-pi*a: 6% of consecutive frames in the first recording, and half the demos
    START on each side. A policy regressing that target averages the two and
    emits a rotation of some fraction of pi -- the wrist jerk at rollout.

    Keeping the axis in one hemisphere is continuous and single-valued for every
    orientation whose axis is not perpendicular to `axis_ref`, i.e. anything short
    of a 90-degree tilt of the gripper. The norm may exceed pi; from_rotvec reads
    that correctly, so nothing downstream converts back.
    """
    r = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    axis = r / np.maximum(theta, 1e-12)
    flip = (axis @ np.asarray(axis_ref, dtype=np.float64)) < 0.0
    out = r.copy()
    out[flip] = (2.0 * np.pi - theta[flip]) * (-axis[flip])
    return out


def pos_rotvec_gripper(pos: np.ndarray, quat_xyzw: np.ndarray, gripper: np.ndarray,
                       axis_ref: np.ndarray | None = None) -> np.ndarray:
    """(N,3) + (N,4 xyzw) + (N,) -> (N,7) pos+rotvec+gripper, SAIL/B-Spline's shared layout.

    `axis_ref` selects the rotation vector as `canonical_rotvec` does; without it
    the rows carry `as_rotvec`'s discontinuity at pi.
    """
    rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
    if axis_ref is not None:
        rotvec = canonical_rotvec(rotvec, axis_ref)
    return np.concatenate([pos, rotvec, gripper[:, None]], axis=1).astype(np.float32)


def parse_image_size(spec: str | None) -> tuple[int, int] | None:
    if spec is None:
        return None
    w, h = spec.lower().split("x")
    return int(w), int(h)


def load_episode_images(
    root: Path, ep_index: int, num_rows: int, image_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """`obs/{cam}_image` per frame, HWC uint8, resized to `image_size` (w, h) if given."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=str(root), root=root, episodes=[ep_index])
    cam_keys = list(dataset.meta.camera_keys)
    images: dict[str, list[np.ndarray]] = {k: [] for k in cam_keys}
    for i in range(len(dataset)):
        frame = dataset[i]
        for key in cam_keys:
            img = (frame[key] * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            if image_size is not None:
                import cv2
                img = cv2.resize(img, image_size)
            images[key].append(img)

    out = {}
    for key, frames in images.items():
        arr = np.stack(frames)
        if len(arr) != num_rows:
            raise ValueError(f"ep{ep_index}: {key} has {len(arr)} frames, expected {num_rows}")
        out[f"obs/{key.split('.')[-1]}_image"] = arr
    return out


def write_demo_file(
    out: Path,
    demos,
    converter_name: str,
    *,
    source_dataset,
    source_repo_id: str,
    root_attrs: dict | None = None,
) -> int:
    """`demos` -> one `data/demo_<i>/...` HDF5. The half of a conversion that is
    not about where the numbers came from.

    `demos` yields `(source_episode_index, {field: array})` lazily, so the source
    is walked one episode at a time rather than held in memory. Demos are
    **renumbered contiguously from 0**: robomimic sorts on `int(name[5:])` and
    B-Spline's replay conversion indexes `demo_{i}` by position, so a gap left by
    a skipped episode is a KeyError. The original index is kept on the group.

    `source_repo_id` is the dataset's id, stamped onto the file so a rollout can
    name its output directory after the task without the operator retyping it,
    and so `train_common.policies_dir` can find where the checkpoints go.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = 0
    try:
        written = _write(tmp, demos, converter_name, source_dataset, source_repo_id, root_attrs)
    except BaseException:
        # A failed conversion leaves nothing behind -- that is what the tmp file
        # is for. Without this a source that raises part way through (a truncated
        # HDF5, an interrupted run) leaves a partial .tmp next to the good files.
        tmp.unlink(missing_ok=True)
        raise

    if written == 0:
        tmp.unlink(missing_ok=True)
        logger.error("no episodes converted")
        return 1
    tmp.replace(out)
    logger.info("wrote %d demos to %s", written, out)
    return 0


def _write(tmp, demos, converter_name, source_dataset, source_repo_id, root_attrs) -> int:
    written = 0
    with h5py.File(tmp, "w") as f:
        f.attrs["source_dataset"] = str(source_dataset)
        f.attrs["source_repo_id"] = source_repo_id
        f.attrs["converter"] = converter_name
        f.attrs["converted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        grp = f.create_group("data")
        for key, val in (root_attrs or {}).items():
            grp.attrs[key] = val
        for ep_index, arrays in demos:
            demo = grp.create_group(f"demo_{written}")
            for field, arr in arrays.items():
                if field.endswith("_image"):
                    # One frame per chunk, uncompressed: training samples frames at
                    # random and gzip made every read decompress its neighbours too.
                    demo.create_dataset(field, data=arr, chunks=(1, *arr.shape[1:]))
                else:
                    demo.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
            # robomimic's SequenceDataset reads num_samples for every demo.
            demo.attrs["num_samples"] = len(arrays["actions"])
            demo.attrs["episode_index"] = ep_index
            written += 1
            logger.info("ep%03d -> demo_%d: %d steps", ep_index, written - 1,
                        len(arrays["actions"]))
    return written


def run_conversion(
    source: str,
    out: Path,
    convert_episode,
    converter_name: str,
    *,
    root_attrs: dict | None = None,
    root_override: str | None = None,
    episodes: set[int] | None = None,
    trim_start: str = "auto",
    max_trim: int = 5,
    min_steps: int = 20,
    include_images: bool = True,
    image_size: tuple[int, int] | None = None,
    source_repo_id: str | None = None,
) -> int:
    """One LeRobot recording -> one HDF5, via `convert_episode` and `write_demo_file`.

    `source_repo_id` defaults to `source`, which is already the HuggingFace id at
    every call site. It is deliberately NOT `str(root)`: the resolved path drops
    the org prefix (`HuskyMango/pickup-bowl` lives at `~/franka_data/pickup-bowl`),
    and LeRobot's own `meta/info.json` does not record the id at all.
    """
    root = resolve_root(root_override or source)
    df, info = load_frames(root)
    dt = 1.0 / float(info["fps"])
    arm = arm_prefix(info)
    screen = make_safety_screen(info, arm)

    def demos():
        for ep_index, group in df.groupby("episode_index"):
            ep_index = int(ep_index)
            if episodes is not None and ep_index not in episodes:
                continue
            group = group.sort_values("frame_index")
            ep_actions = np.stack(group["action"].to_numpy()).astype(np.float64)
            ep_states = np.stack(group["observation.state"].to_numpy()).astype(np.float64)
            check_action_space(ep_actions)

            trim = (leading_trim(ep_states[:, :NUM_JOINTS], dt, max_trim)
                    if trim_start == "auto" else int(trim_start))
            ep_actions, ep_states = ep_actions[trim:], ep_states[trim:]
            if len(ep_actions) < min_steps:
                logger.warning("ep%03d: %d steps after trimming %d, below --min-steps %d; skipped",
                               ep_index, len(ep_actions), trim, min_steps)
                continue

            arrays = convert_episode(ep_actions, ep_states, arm, screen)
            if include_images:
                imgs = load_episode_images(root, ep_index, len(ep_actions) + trim, image_size)
                for key, arr in imgs.items():
                    arrays[key] = arr[trim:]
            if trim:
                logger.info("ep%03d: trimmed %d leading frames", ep_index, trim)
            yield ep_index, arrays

    return write_demo_file(out, demos(), converter_name, source_dataset=root,
                           source_repo_id=source_repo_id or source, root_attrs=root_attrs)
