#!/usr/bin/env python3
"""One EE_POS recording -> three training-ready artifacts: sysid HDF5, SAIL
HDF5, B-Spline HDF5. See baselines/README.md for what each one contains and why.

    python scripts/prepare_baseline_datasets.py --source-repo-id sysid-8-28 \
        --out-dir ~/franka_data/baseline_prep/sysid-8-28

All three converters read the same recording purely offline (parquet + the
recorded video, never the arm), so this is safe to run any time after a
recording finishes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import baselines.bspline_bridge.dataset as bspline_dataset  # noqa: E402
import baselines.sail_bridge.dataset as sail_dataset  # noqa: E402
import sysid.lerobot_to_hdf5 as sysid_hdf5  # noqa: E402
from baselines.common import parse_image_size  # noqa: E402

logger = logging.getLogger("prepare_baseline_datasets")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo-id", required=True,
                    help="dataset root, or a repo id resolved under ~/franka_data")
    p.add_argument("--out-dir", required=True, type=Path,
                    help="directory to write sysid.hdf5, sail.hdf5, bspline.hdf5 into")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices, default all")
    p.add_argument("--trim-start", default="auto")
    p.add_argument("--max-trim", type=int, default=5)
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--no-images", action="store_true",
                    help="skip camera frames in the SAIL/B-Spline HDF5s (schema/shape checks only)")
    p.add_argument("--image-size", default=None,
                    help="WxH to resize the SAIL and B-Spline camera frames, e.g. 84x84. "
                         "Default keeps the recording's resolution; the file's size is "
                         "what both policies train and roll out at")
    p.add_argument("--skip", nargs="*", choices=["sysid", "sail", "bspline"], default=[],
                    help="artifacts to skip")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    episodes = {int(x) for x in args.episodes.split(",")} if args.episodes else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "sysid": lambda: sysid_hdf5.convert(
            args.source_repo_id, args.out_dir / "sysid.hdf5", episodes=episodes,
            trim_start=args.trim_start, max_trim=args.max_trim, min_steps=args.min_steps,
        ),
        "sail": lambda: sail_dataset.convert(
            args.source_repo_id, args.out_dir / "sail.hdf5", episodes=episodes,
            source_repo_id=args.source_repo_id,
            trim_start=args.trim_start, max_trim=args.max_trim, min_steps=args.min_steps,
            include_images=not args.no_images,
            image_size=parse_image_size(args.image_size),
        ),
        "bspline": lambda: bspline_dataset.convert(
            args.source_repo_id, args.out_dir / "bspline.hdf5", episodes=episodes,
            source_repo_id=args.source_repo_id,
            trim_start=args.trim_start, max_trim=args.max_trim, min_steps=args.min_steps,
            include_images=not args.no_images,
            image_size=parse_image_size(args.image_size),
        ),
    }

    failed = []
    for name, job in jobs.items():
        if name in args.skip:
            continue
        logger.info("=== %s ===", name)
        if job() != 0:
            failed.append(name)

    if failed:
        logger.error("failed: %s", ", ".join(failed))
        return 1
    logger.info("wrote all requested artifacts to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
