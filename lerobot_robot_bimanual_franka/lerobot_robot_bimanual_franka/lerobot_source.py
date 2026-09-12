"""Read a recorded LeRobot dataset from its parquet shards, without decoding
camera streams. Shared by the sysid export and the baseline converters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# The EE_POS action schema, in feature order.
EE_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
POS, QUAT = slice(0, 3), slice(3, 7)
GRIPPER, KP, KD = 7, 8, 9

# Reach bounds separating an absolute goal from a delta. The delta envelope is
# +/-0.05 m per axis (norm <= 0.0866) and the FR3 works at 0.3-0.8 m of reach, so
# nothing legitimate lands between them.
_DELTA_MAX_NORM = 0.10
_ABS_MIN_NORM = 0.15

_SEARCH_DIRS = (Path.home() / "franka_data", Path.home() / ".cache/huggingface/lerobot")


def resolve_root(spec: str) -> Path:
    """Dataset root from a path or a repo id, without touching the network."""
    candidates = [Path(spec).expanduser()]
    candidates += [d / spec for d in _SEARCH_DIRS]
    candidates += [d / spec.split("/")[-1] for d in _SEARCH_DIRS]
    for path in candidates:
        if (path / "meta" / "info.json").is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"no LeRobot dataset for {spec!r}; looked in "
        + ", ".join(str(c) for c in candidates)
    )


def load_frames(root: Path) -> tuple[pd.DataFrame, dict]:
    """Every data shard, concatenated, plus the dataset's info.json.

    The parquet is read directly rather than through `LeRobotDataset`: nothing
    here needs an image, and the dataset class would decode camera streams to
    reach seven joint angles.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    shards = sorted((root / "data").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no data/**/*.parquet under {root}")
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    return df, info


def task_names(root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    return {int(idx): str(name) for name, idx in tasks["task_index"].items()}


def arm_prefix(info: dict) -> str:
    """The single `l_`/`r_` prefix this action carries; rejects bimanual."""
    names = list(info["features"]["action"]["names"])
    prefixes = sorted({n.rsplit("_", 1)[0] for n in names if n.endswith("_x")})
    if len(prefixes) != 1:
        raise ValueError(
            f"expected exactly one arm in the action, found {prefixes or names}; "
            "this converter is single-arm."
        )
    prefix = prefixes[0]
    expected = [f"{prefix}_{k}" for k in EE_KEYS] + ["kp", "kd"]
    if names != expected:
        raise ValueError(f"unexpected action schema {names}, expected {expected}")
    return prefix


def check_action_space(actions: np.ndarray) -> None:
    """EE_POS or EE_DELTA, decided by reach.

    Both modes emit the SAME feature names, so the recording cannot say which
    one ran; this refuses rather than reading a delta as an absolute goal.
    """
    norms = np.linalg.norm(actions[:, POS], axis=1)
    if float(np.median(norms)) > _ABS_MIN_NORM:
        return
    if float(np.max(norms)) < _DELTA_MAX_NORM:
        raise ValueError(
            "this looks like an EE_DELTA recording (max |action_pos| = "
            f"{float(np.max(norms)):.3f} m); record with EE_POS instead, or "
            "relabel with scripts/replay_dataset.py --mode ee_pose."
        )
    raise ValueError(
        f"cannot classify the action space: |action_pos| median "
        f"{float(np.median(norms)):.3f} m, max {float(np.max(norms)):.3f} m."
    )
