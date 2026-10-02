#!/usr/bin/env python3
"""Join episode files into one, for the plant fit or a rollout over several runs.

    python sysid/merge_episodes.py ~/sysid/outputs/combined/ee_pose/combined.hdf5 \\
        ~/sysid/outputs/<excite run>/ee_pose/excitation.hdf5 \\
        ~/sysid/outputs/sysid-8-28/ee_pose/sysid-8-28.hdf5 --prefix sysid0828_ \\
        --validate sysid0828_ep000 sysid0828_ep004

The FIRST argument is the file to write; everything after it is read. An
existing output is refused unless --overwrite is given, and an output that is
also an input is refused outright: the writer replaces the file whole, so a
recording listed first would be gone.

Episodes keep their names unless renamed; `--prefix` applies to the file listed
just before it, `--rename old=new` to one episode, and `--validate` suffixes the
named episodes `_validate` and every other one `_train` (the split the fit's
val_regex reads). Names must end up unique. The output is validated before it is
written and lands in the layout everything reads (EPISODE_HDF5.md).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-fast"))

from utils.sysid import episode_hdf5  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="the file to WRITE (its directory should hold nothing else the fit would read)")
    ap.add_argument("inputs", nargs="+", help="episode files to read; a --prefix after a file applies to it")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    ap.add_argument("--prefix", action="append", default=[], metavar="PREFIX",
                    help="prefix for the episodes of the input listed just before this flag")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--validate", nargs="*", default=None, metavar="NAME",
                    help="episodes to suffix _validate; the rest get _train")
    ap.add_argument("--strict", action="store_true", help="refuse legacy files missing the frame attrs")
    args, rest = ap.parse_known_args()
    if rest:
        ap.error(f"unrecognised: {rest}")
    out = Path(args.out).expanduser()
    if any(out.resolve() == Path(i).expanduser().resolve() for i in args.inputs):
        ap.error(f"{args.out} is both the output and an input; the first argument is the "
                 f"file to WRITE, and writing replaces it whole")
    if out.exists() and not args.overwrite:
        ap.error(f"{args.out} exists; the first argument is the file to WRITE. Pick another "
                 f"path, or pass --overwrite to replace it")

    # argparse cannot tie a repeated --prefix to its neighbouring input, so walk argv.
    prefixes = {}
    argv = sys.argv[1:]
    last_input = None
    for i, tok in enumerate(argv):
        if tok in args.inputs:
            last_input = tok
        elif tok == "--prefix" and last_input is not None:
            prefixes[last_input] = argv[i + 1]

    renames = dict(r.split("=", 1) for r in args.rename)
    episodes = []
    for path in args.inputs:
        for name, arrays, attrs, curve in episode_hdf5.read_episodes(path):
            attrs = episode_hdf5.with_defaults(attrs) if not args.strict else attrs
            name = prefixes.get(path, "") + name
            name = renames.get(name, name)
            attrs = {**attrs, "merged_from": f"{path}:{name}"}
            episodes.append([name, arrays, attrs, curve])
    if args.validate is not None:
        names = {e[0] for e in episodes}
        unknown = sorted(set(args.validate) - names)
        if unknown:
            raise SystemExit(f"--validate names {unknown} are not among {sorted(names)}")
        for e in episodes:
            e[0] = f"{e[0]}_{'validate' if e[0] in args.validate else 'train'}"
    names = [e[0] for e in episodes]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"duplicate episode names {dupes}; use --prefix or --rename")

    problems = [pr for e in episodes
                for pr in episode_hdf5.validate_episode(*e, legacy_ok=not args.strict)]
    if problems:
        raise SystemExit("not written:\n  " + "\n  ".join(problems))
    out = episode_hdf5.write_episodes(args.out, [tuple(e) for e in episodes],
                                      root_attrs={"merged_from": list(args.inputs)},
                                      producer="sysid/merge_episodes.py")
    for e in episodes:
        print(f"  {e[0]:<28} {int(e[2]['num_samples']):>5} steps")
    print(f"wrote {len(episodes)} episode(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
