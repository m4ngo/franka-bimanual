#!/usr/bin/env python3
"""Combine multiple sysid aggregate_sim_format.hdf5 files into one.

Each input aggregate has the sim layout produced by sysid.py's
save_sim_format_hdf5(): f[group_key][episode_key][field] -> (T, D), with
per-episode attrs (mode, reference_episode, stop_reason, etc.) and an
optional top-level `aggregate_sim_format_group_key` run.json field recording
which group_key was used (fixed to "data" for LeRobotDataset-sourced runs,
otherwise the mirrored sim group key).

Because different input aggregates can carry different group_keys (a
HDF5-sourced run mirrors its sim condition's directory name, e.g.
"kp_actn0.50_damp_actn0.50", while a LeRobotDataset-sourced run always uses
"data"), and because episode names can collide across runs (every replay
sweep restarts numbering at episode_000000), this script FLATTENS everything
into a single output group by default. Episode name collisions are resolved
by prefixing with a per-source tag; the original group_key and episode_key
and source file are stamped onto each episode's attrs so provenance is never
lost, even if the merged file is later separated from this script's log.

Usage:
    python combine_aggregates.py out.hdf5 agg1.hdf5 agg2.hdf5 [agg3.hdf5 ...]

    # Keep each input's original group_key as a separate top-level group
    # instead of flattening (only safe if group_keys don't collide either):
    python combine_aggregates.py --keep-groups out.hdf5 agg1.hdf5 agg2.hdf5

    # Custom output group name when flattening (default: "data"):
    python combine_aggregates.py --group-key merged out.hdf5 agg1.hdf5 agg2.hdf5
"""

import argparse
import logging
import os
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)


def _source_tag(path: str, index: int) -> str:
    """Short, filesystem-derived tag identifying a source file, used to
    disambiguate colliding episode names. Prefers the parent directory name
    (sysid run dirs are timestamped + tagged, so this is usually unique and
    human-readable) and falls back to a positional index if that's still
    ambiguous relative to other inputs."""
    stem = Path(path).resolve().parent.name or Path(path).stem
    return f"{stem}_{index}"


def combine(
    out_path: str,
    in_paths: list[str],
    keep_groups: bool = False,
    group_key: str = "data",
) -> None:
    if len(in_paths) < 2:
        raise ValueError("need at least 2 input files to combine")

    out_path = os.path.abspath(os.path.expanduser(out_path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"

    n_episodes_total = 0
    n_collisions = 0
    seen_names: dict[str, set[str]] = {}  # out_group -> set of episode names written
    group_key_sources: dict[str, str] = {}  # out_group -> first source file that created it

    with h5py.File(tmp_path, "w") as fout:
        for i, in_path in enumerate(in_paths):
            in_path = os.path.abspath(os.path.expanduser(in_path))
            tag = _source_tag(in_path, i)
            with h5py.File(in_path, "r") as fin:
                src_group_keys = list(fin.keys())
                if len(src_group_keys) != 1:
                    logger.warning(
                        "%s: expected exactly one top-level group, found %s; "
                        "using the first",
                        in_path, src_group_keys,
                    )
                src_group_key = src_group_keys[0]
                src_group = fin[src_group_key]

                out_group_key = src_group_key if keep_groups else group_key
                if keep_groups and out_group_key in group_key_sources:
                    raise ValueError(
                        f"--keep-groups was set but group_key {out_group_key!r} "
                        f"from {in_path!r} collides with the same group_key "
                        f"already taken from {group_key_sources[out_group_key]!r}; "
                        "rerun without --keep-groups to flatten instead"
                    )
                if out_group_key in fout:
                    out_group = fout[out_group_key]
                else:
                    out_group = fout.create_group(out_group_key)
                    seen_names[out_group_key] = set()
                group_key_sources.setdefault(out_group_key, in_path)

                for ep_key in src_group.keys():
                    ep_in = src_group[ep_key]
                    out_ep_key = ep_key
                    if out_ep_key in seen_names[out_group_key]:
                        out_ep_key = f"{tag}__{ep_key}"
                        n_collisions += 1
                        suffix = 1
                        while out_ep_key in seen_names[out_group_key]:
                            out_ep_key = f"{tag}__{ep_key}__{suffix}"
                            suffix += 1
                    seen_names[out_group_key].add(out_ep_key)

                    ep_out = out_group.create_group(out_ep_key)
                    for field in ep_in.keys():
                        ep_in.copy(field, ep_out)
                    for k, v in ep_in.attrs.items():
                        ep_out.attrs[k] = v
                    # Provenance, so this episode is traceable even if the
                    # merged file is later separated from this run's logs.
                    ep_out.attrs["source_file"] = in_path
                    ep_out.attrs["source_group_key"] = src_group_key
                    ep_out.attrs["source_episode_key"] = ep_key

                    n_episodes_total += 1

                logger.info(
                    "%s: merged %d episodes from group %r into output group %r",
                    in_path, len(src_group.keys()), src_group_key, out_group_key,
                )

    os.replace(tmp_path, out_path)
    logger.info(
        "wrote %s: %d episodes from %d files (%d renamed on collision)",
        out_path, n_episodes_total, len(in_paths), n_collisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple sysid aggregate_sim_format.hdf5 files into one.",
    )
    parser.add_argument("out_file", help="Path to write the combined HDF5 file")
    parser.add_argument("in_files", nargs="+", help="Two or more aggregate HDF5 files to combine")
    parser.add_argument(
        "--keep-groups", action="store_true",
        help="Preserve each input's original top-level group_key instead of "
             "flattening all episodes into one group. Errors out on a "
             "group_key collision rather than silently merging.",
    )
    parser.add_argument(
        "--group-key", default="data",
        help="Output top-level group name when flattening (default: 'data'). Ignored with --keep-groups.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    combine(args.out_file, args.in_files, keep_groups=args.keep_groups, group_key=args.group_key)


if __name__ == "__main__":
    main()
