"""Shared by both baseline dataset converters. See baselines/README.md."""

from __future__ import annotations

import logging
import sys
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


def pos_rotvec_gripper(pos: np.ndarray, quat_xyzw: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """(N,3) + (N,4 xyzw) + (N,) -> (N,7) pos+rotvec+gripper, SAIL/B-Spline's shared layout."""
    rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
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
) -> int:
    """One recording -> one `data/demo_<i>/...` HDF5, via `convert_episode`."""
    root = resolve_root(root_override or source)
    df, info = load_frames(root)
    dt = 1.0 / float(info["fps"])
    arm = arm_prefix(info)
    screen = make_safety_screen(info, arm)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = 0
    with h5py.File(tmp, "w") as f:
        f.attrs["source_dataset"] = str(root)
        f.attrs["converter"] = converter_name
        grp = f.create_group("data")
        for key, val in (root_attrs or {}).items():
            grp.attrs[key] = val
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

            demo = grp.create_group(f"demo_{written}")
            for field, arr in arrays.items():
                demo.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
            # robomimic's SequenceDataset reads num_samples for every demo.
            demo.attrs["num_samples"] = len(ep_actions)
            demo.attrs["episode_index"] = ep_index
            written += 1
            logger.info("ep%03d -> demo_%d: %d steps (trimmed %d)",
                        ep_index, written - 1, len(ep_actions), trim)

    if written == 0:
        tmp.unlink(missing_ok=True)
        logger.error("no episodes converted")
        return 1
    tmp.replace(out)
    logger.info("wrote %d demos to %s", written, out)
    return 0
