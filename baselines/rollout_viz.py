#!/usr/bin/env python3
"""Per-episode chunk visualizations for real B-Spline and SAIL rollouts.

    python -m baselines.rollout_viz <run_dir>
    python -m baselines.rollout_viz ~/franka_data/outputs/HuskyMango   # every real run under it

Writes `episode_<NNN>.html` into each run directory: residual_wrapper/viz.py's
`save_rollout_html`, the page run_residual.py writes for its own runs. It shows
the arm, its realized EE path, the plan from the most recent inference, and
every dispatched goal as a static dashed path. The real rollouts log each plan
to chunks.npz and call `render_after_run` when the run ends. A run recorded
before chunks.npz existed has no plans, so the step's dispatched goal is drawn
in their place.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CHUNKS_FILE = "chunks.npz"
METHODS = ("bspline", "sail")
_POSE_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw")

logger = logging.getLogger("baselines.rollout_viz")


class ChunkLog:
    """Every episode's plans for one run, as chunks.npz.

    chunk_step_003 (K,) is the dispatch step each of episode 3's K plans arrived
    at; chunk_pose_003 (K, N, 7) the plans as base-frame [xyz, quat_xyzw] poses,
    NaN-padded to the longest. Rewritten whole per episode, as ForceLog is.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.arrays: dict[str, np.ndarray] = {}

    def add(self, episode: int, chunks: list[tuple[int, np.ndarray]]) -> None:
        if not chunks:
            return
        pose = np.full((len(chunks), max(len(c) for _, c in chunks), 7), np.nan, np.float32)
        for i, (_, c) in enumerate(chunks):
            pose[i, :len(c)] = c
        self.arrays[f"chunk_step_{episode:03d}"] = np.array([s for s, _ in chunks], np.int64)
        self.arrays[f"chunk_pose_{episode:03d}"] = pose
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **self.arrays)
        tmp.replace(self.path)


def render_run(run_dir) -> list[Path]:
    """Write episode_<NNN>.html for every recorded episode of one run."""
    sys.path.insert(0, str(_ROOT / "residual_wrapper"))
    import franka_config as fc
    import pandas as pd
    from lerobot_robot_bimanual_franka.franka_fk import franka_fk_chain
    from viz import EpisodeRecorder, save_rollout_html

    run_dir = Path(run_dir)
    files = sorted((run_dir / "dataset" / "data").glob("*/*.parquet"))
    if not files:
        logger.warning("%s: no recorded dataset, nothing to draw", run_dir)
        return []
    manifest = json.loads((run_dir / "manifest.json").read_text())
    env = manifest["environment"]
    info = json.loads((run_dir / "dataset" / "meta" / "info.json").read_text())
    names = [str(n).removeprefix(f"{env['key_prefix']}_")
             for n in np.ravel(info["features"]["action"]["names"])]
    col = {n: i for i, n in enumerate(names)}
    ee_pos = manifest["policy"]["control_mode"] == "EE_POS"
    base = fc.robot_base_in_world(env["physical_arm"])
    stride = max(1, round(info["fps"] / fc.control_fps()))
    episodes = {}
    if (run_dir / "episodes.jsonl").is_file():
        for line in (run_dir / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            episodes[e.get("dataset_episode_index", e["episode"])] = e
    chunk_file = run_dir / CHUNKS_FILE
    chunks = dict(np.load(chunk_file)) if chunk_file.is_file() else {}

    df = pd.concat(pd.read_parquet(f, columns=["observation.state", "action", "episode_index"])
                   for f in files)
    written = []
    for ds_idx, rows in df.groupby("episode_index"):
        e = episodes.get(int(ds_idx), {"episode": int(ds_idx)})
        idx = e["episode"]
        q = np.stack(rows["observation.state"].to_numpy())[:, :7].astype(np.float64)
        act = np.stack(rows["action"].to_numpy()).astype(np.float64)
        goal = act[:, [col[k] for k in _POSE_KEYS]]
        fk = np.array([franka_fk_chain(qi)[7, :3, 3] for qi in q])

        rec = EpisodeRecorder()
        for k in range(len(q)):
            rec.record(q=q[k], actual_ee_pos=fk[k], base_desired_pos=goal[k],
                       total_desired_pos=goal[k], kp=act[k, col["kp"]], kd=act[k, col["kd"]],
                       gripper=act[k, col["gripper"]], res_gripper=0.0)
        label = "predicted chunk"
        if f"chunk_step_{idx:03d}" in chunks:
            for s, p in zip(chunks[f"chunk_step_{idx:03d}"], chunks[f"chunk_pose_{idx:03d}"]):
                p = p[~np.isnan(p[:, 0])]
                rec.record_chunk(int(s), fk[min(int(s), len(fk) - 1)], p[:, :3], p[:, :3], p, p)
        elif ee_pos:
            label = "dispatched goal (no chunk log)"
            for k in range(len(goal)):
                g = goal[k:k + 1]
                rec.record_chunk(k, fk[k], g[:, :3], g[:, :3], g, g)

        path = run_dir / f"episode_{idx:03d}.html"
        save_rollout_html(
            rec, str(path),
            title=f"{manifest['run']['method']} {run_dir.name}: episode {idx} ({e.get('verdict', '?')})",
            frame_stride=stride, fps=info["fps"] / stride,
            robot_base_in_world_translation=tuple(float(v) for v in base.translation),
            robot_base_in_world_quat_wxyz=base.quat_wxyz,
            forecast_name=label,
            reference_trail=base.apply(goal[:, :3]) if ee_pos else None,
            reference_name="dispatched goals",
        )
        written.append(path)
    logger.info("%s: wrote %d episode page(s)", run_dir, len(written))
    return written


def render_after_run(run_dir) -> None:
    """The rollouts' end-of-run hook. Never raises: the run is already on disk."""
    print(f"\r\nrendering chunk visualizations into {run_dir}\r", flush=True)
    try:
        render_run(run_dir)
    except Exception:
        logger.exception("visualization failed; re-run python -m baselines.rollout_viz %s",
                         run_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run directories, or directories to search for them")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    for path in args.paths:
        for m in sorted(Path(path).expanduser().rglob("manifest.json")):
            doc = json.loads(m.read_text())
            # Real runs only: a LIBERO sim run's environment has no physical arm.
            if doc["run"]["method"] in METHODS and "physical_arm" in doc.get("environment", {}):
                render_run(m.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
