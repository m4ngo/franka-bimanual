"""Offline aggregate visualization: overlay all sim/real episode pairs of a run.

Usage
-----
python sysid/aggregate_viz.py <ref.hdf5> <run_dir> [--output <run_dir>/aggregate.html] [--fps 20]

<ref.hdf5> is the sim reference dataset (f[group][episode][field]); <run_dir> is
a sysid.py run directory containing *_record_*.hdf5 files. Episodes are matched
via the `reference_episode` HDF5 attr stamped by sysid.py (falling back to
parsing the filename after "record_").
"""

import argparse
import sys
from pathlib import Path

import h5py

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _viz import save_aggregate_html  # noqa: E402


def _load_episode(group: h5py.Group) -> dict:
    return {field: group[field][:] for field in group}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sim-vs-real overlay for a sysid run.")
    parser.add_argument("ref_file", help="Reference HDF5 (sim format)")
    parser.add_argument("run_dir", help="sysid.py run directory with *_record_*.hdf5 files")
    parser.add_argument("--output", default=None, help="Output HTML (default: <run_dir>/aggregate.html)")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    rec_files = sorted(run_dir.glob("*_record_*.hdf5"))
    if not rec_files:
        sys.exit(f"no *_record_*.hdf5 files in {run_dir}")

    items = []
    with h5py.File(args.ref_file, "r") as ref_f:
        ref_group = ref_f[list(ref_f.keys())[0]]
        for path in rec_files:
            with h5py.File(path, "r") as f:
                ep = f["data/episode_0"]
                name = ep.attrs.get("reference_episode") or path.stem.split("record_", 1)[-1]
                if name not in ref_group:
                    print(f"skipping {path.name}: reference episode {name!r} not in {args.ref_file}")
                    continue
                items.append((name, _load_episode(ref_group[name]), _load_episode(ep)))

    if not items:
        sys.exit("no episodes matched between run dir and reference file")
    out = args.output or str(run_dir / "aggregate.html")
    save_aggregate_html(items, out, fps=args.fps)


if __name__ == "__main__":
    main()
