"""Offline visualization: compare a reference trajectory against a previously
recorded sysid replay without connecting to the robot.

Usage
-----
python sysid/viz_sysid.py <ref.hdf5> <replayed.hdf5> [--output viz.html] [--fps 20] [--stride 1]

The reference HDF5 must have the structure produced by the sim:
    f[group][episode][field] → (T, D) dataset

The replayed HDF5 must have the structure written by sysid.py's save_sysid_hdf5:
    f["data"]["episode_0"][field] → (T, D) dataset
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _viz import compute_trajectory_errors, save_comparison_html, save_errors_json  # noqa: E402


def _load_ref(path: str, episode_idx: int = 2) -> dict[str, np.ndarray]:
    """Load one episode from a sim-format HDF5 (group → episode → fields)."""
    traj: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        group = f[list(f.keys())[0]]
        keys = list(group.keys())
        ep_key = keys[min(episode_idx, len(keys) - 1)]
        print(f"loading reference episode: {ep_key}")
        for field in group[ep_key]:
            traj[field] = group[ep_key][field][:]
    return traj


def _load_replayed(path: str) -> dict[str, np.ndarray]:
    """Load from the flat layout written by sysid.py's save_sysid_hdf5."""
    recorded: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        ep = f["data"]["episode_0"]
        for field in ep:
            recorded[field] = ep[field][:]
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a prior sysid replay against a reference trajectory."
    )
    parser.add_argument("ref_file",      help="Reference HDF5 (sim format)")
    parser.add_argument("replayed_file", help="Replayed HDF5 (sysid.py output)")
    parser.add_argument(
        "--output", default=None,
        help="Output HTML path (default: <replayed_file>.html)",
    )
    parser.add_argument("--fps",    type=float, default=20.0, help="Playback rate in Hz")
    parser.add_argument("--stride", type=int,   default=1,    help="Animate every Nth step")
    parser.add_argument(
        "--episode-idx", type=int, default=2,
        help="Episode index within the reference HDF5 group (default 2)",
    )
    args = parser.parse_args()

    ref      = _load_ref(args.ref_file, episode_idx=args.episode_idx)
    recorded = _load_replayed(args.replayed_file)

    out = args.output or str(Path(args.replayed_file).with_suffix(".html"))
    save_comparison_html(ref, recorded, out, fps=args.fps, frame_stride=args.stride)

    traj_name = Path(args.replayed_file).stem
    errors = compute_trajectory_errors(ref, recorded, name=traj_name)
    errors_path = str(Path(out).with_suffix(".errors.json"))
    save_errors_json([errors], errors_path)


if __name__ == "__main__":
    main()
