"""Stitch time-aligned episode videos side by side (2) or into a 2x2 grid (3-4).

Inputs are the per-camera mp4s from run_residual.py --save-videos: frame index
== control step, so same-fps videos are synchronized from frame 0 (the first
post-homing step). Shorter videos are padded with their dimmed last frame and
an 'ended' marker rather than truncated -- differing episode lengths are
signal, not error.

Usage:
  python stitch_videos.py bowl-7-20
    run-dir mode with defaults: auto-discovers the episode_<cam>.mp4 pairs
    present in BOTH bowl-7-20/base/ and bowl-7-20/resi/, labels them
    base/residual, writes bowl-7-20/episode_compare.mp4.
  python stitch_videos.py --run-dir bowl-7-20 --cams cam_3 cam_5 [--episode N] [-o out.mp4]
    explicit camera subset / recorded-episode naming (episode_NNN_<cam>.mp4);
    grid rows = cameras, columns = base | residual.
  python stitch_videos.py task-name
    trajectory mode (auto-detected): task-name/base/<N>/ and task-name/resi/<N>/
    each hold *_<cam>.mp4 files for trajectory N. Each trajectory is stitched
    side by side (base | residual) exactly as in run-dir mode, then the
    trajectories are concatenated one after another in numeric order into a
    single output video, with a "traj N" label burned into every tile.
  python stitch_videos.py a.mp4 b.mp4 [c.mp4 d.mp4] -o out.mp4 [--labels A B ...]
    free-form file mode.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PANE_H = 360  # common pane height (px); widths equalized by centered padding


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    return cap


def _tile(frame: np.ndarray, label: str, ended: bool, width: int) -> np.ndarray:
    f = cv2.resize(frame, (int(round(frame.shape[1] * PANE_H / frame.shape[0])), PANE_H))
    if ended:
        f = (f * 0.55).astype(np.uint8)
    for color, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(f, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thick, cv2.LINE_AA)
        if ended:
            cv2.putText(f, "ended", (6, PANE_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thick, cv2.LINE_AA)
    if f.shape[1] < width:  # center-pad to the common tile width
        pad = width - f.shape[1]
        f = cv2.copyMakeBorder(f, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=0)
    return f


def _discover_cams(base_dir: Path, resi_dir: Path, stem: str) -> list:
    """Camera names with a stem_<cam>.mp4 pair in both dirs, sorted."""
    def _names(d: Path) -> set:
        return {q.name[len(stem) + 1:-4] for q in d.glob(f"{stem}_*.mp4")}
    return sorted(_names(base_dir) & _names(resi_dir))


def _stitch_group(paths, labels, cols, info: dict):
    """Open `paths`, yield synchronized grid frames (dimmed+'ended' padding for
    shorter videos), then release.

    `info` is a caller-provided dict this generator fills in with 'fps' (from
    the first source, available immediately) and 'counts' (frames read per
    source, available once the generator is exhausted) -- a plain out-param,
    since a generator's return value isn't reachable via normal iteration.
    """
    caps = [_open(q) for q in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 20.0
    info["fps"] = fps
    for q, c in zip(paths[1:], caps[1:]):
        f2 = c.get(cv2.CAP_PROP_FPS) or fps
        if abs(f2 - fps) > 0.1:
            print(f"WARNING: fps mismatch: {paths[0].name}={fps:.2f} vs {q.name}={f2:.2f}; "
                  "videos will drift out of sync")

    widths = []
    firsts = []
    for c, q in zip(caps, paths):
        ok, f = c.read()
        if not ok:
            sys.exit(f"no frames in {q}")
        firsts.append(f)
        widths.append(int(round(f.shape[1] * PANE_H / f.shape[0])))
    tile_w = max(widths)

    last = list(firsts)
    ended = [False] * len(caps)
    counts = [1] * len(caps)
    while True:
        tiles = [_tile(last[i], labels[i], ended[i], tile_w) for i in range(len(caps))]
        rows = [cv2.hconcat(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
        if len(tiles) % cols:  # odd count: pad the last row with a blank pane
            rows[-1] = cv2.hconcat([rows[-1], np.zeros_like(tiles[0])])
        grid = cv2.vconcat(rows)
        yield grid

        for i, c in enumerate(caps):
            if ended[i]:
                continue
            ok, f = c.read()
            if ok:
                last[i] = f
                counts[i] += 1
            else:
                ended[i] = True
        if all(ended):
            break
    for c in caps:
        c.release()
    info["counts"] = counts


def _run_trajectory_mode(run: Path, args, p: argparse.ArgumentParser) -> None:
    stem = "episode" if args.episode is None else f"episode_{args.episode:03d}"
    base_root, resi_root = run / "base", run / "resi"

    def _traj_dirs(root: Path) -> set:
        return {d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()}

    trajs = sorted(_traj_dirs(base_root) & _traj_dirs(resi_root), key=int)
    if not trajs:
        p.error(f"no numbered trajectory dirs found under both {base_root} and {resi_root}")
    print(f"trajectories: {', '.join(trajs)}")

    cams = args.cams
    if not cams:
        # cams must be consistent across trajectories; discover from the first.
        cams = _discover_cams(base_root / trajs[0], resi_root / trajs[0], stem)
        if not cams:
            p.error(f"no {stem}_<cam>.mp4 pairs found under {base_root}/{trajs[0]} and {resi_root}/{trajs[0]}")
        print(f"cameras: {', '.join(cams)}")
    cols = 2

    out = Path(args.output) if args.output else run / f"{stem}_compare.mp4"
    writer = None
    ref_fps = None  # first trajectory's fps; later ones warn on mismatch but keep playing at this rate
    n_out = 0
    for traj in trajs:
        paths, labels = [], []
        for cam in cams:
            for variant, root, label in (("base", base_root, "base"), ("resi", resi_root, "residual")):
                paths.append(root / traj / f"{stem}_{cam}.mp4")
                base_label = label if len(cams) == 1 else f"{label} {cam}"
                labels.append(f"traj {traj}: {base_label}")
        missing = [str(q) for q in paths if not q.exists()]
        if missing:
            sys.exit("missing input video(s):\n  " + "\n  ".join(missing))

        info: dict = {}
        for grid in _stitch_group(paths, labels, cols, info):
            if writer is None:
                ref_fps = info["fps"]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), ref_fps,
                                          (grid.shape[1], grid.shape[0]))
            elif abs(info["fps"] - ref_fps) > 0.1:
                print(f"WARNING: traj {traj} fps={info['fps']:.2f} differs from "
                      f"traj {trajs[0]} fps={ref_fps:.2f}; output stays at {ref_fps:.2f}")
            writer.write(grid)
            n_out += 1
        counts = info["counts"]
        print(f"  traj {traj}: " + ", ".join(f"{q.name}={n}" for q, n in zip(paths, counts)))

    writer.release()
    print(f"wrote {out} ({n_out} frames total across {len(trajs)} trajectories, {ref_fps:.1f} fps)")


def _is_trajectory_mode(run: Path) -> bool:
    base_dir, resi_dir = run / "base", run / "resi"
    if not (base_dir.is_dir() and resi_dir.is_dir()):
        return False
    has_numbered_subdirs = any(
        d.is_dir() and d.name.isdigit() for d in base_dir.iterdir()
    )
    return has_numbered_subdirs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", help="2-4 mp4 paths (alternative to --run-dir)")
    p.add_argument("--run-dir", default=None, help="dir containing base/ and resi/ episode videos")
    p.add_argument("--cams", nargs="+", default=None, help="camera names (run-dir mode)")
    p.add_argument("--episode", type=int, default=None,
                   help="episode number for recorded runs (episode_NNN_*); default: free-run naming")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    # Bare directory argument == run-dir mode with defaults.
    if len(args.videos) == 1 and Path(args.videos[0]).is_dir() and not args.run_dir:
        args.run_dir, args.videos = args.videos[0], []

    if args.run_dir:
        run = Path(args.run_dir)
        if _is_trajectory_mode(run):
            _run_trajectory_mode(run, args, p)
            return

        stem = "episode" if args.episode is None else f"episode_{args.episode:03d}"
        cams = args.cams
        if not cams:
            cams = _discover_cams(run / "base", run / "resi", stem)
            if not cams:
                p.error(f"no {stem}_<cam>.mp4 pairs found under {run}/base and {run}/resi")
            print(f"cameras: {', '.join(cams)}")
        paths, labels = [], []
        for cam in cams:  # row-major: rows = cams, cols = base | residual
            for variant, label in (("base", "base"), ("resi", "residual")):
                paths.append(run / variant / f"{stem}_{cam}.mp4")
                labels.append(label if len(cams) == 1 else f"{label} {cam}")
        if args.labels:
            labels = args.labels
        out = Path(args.output) if args.output else run / f"{stem}_compare.mp4"
        cols = 2
    else:
        if not 2 <= len(args.videos) <= 4:
            p.error("pass 2-4 video paths, or use --run-dir")
        paths = [Path(v) for v in args.videos]
        labels = args.labels or [f"{q.parent.name}/{q.stem}" for q in paths]
        out = Path(args.output) if args.output else Path("stitched.mp4")
        cols = 1 if len(paths) == 1 else 2
    missing = [str(q) for q in paths if not q.exists()]
    if missing:
        sys.exit("missing input video(s):\n  " + "\n  ".join(missing))
    if len(labels) != len(paths):
        p.error(f"{len(labels)} labels for {len(paths)} videos")

    caps = [_open(q) for q in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 20.0
    for q, c in zip(paths[1:], caps[1:]):
        f2 = c.get(cv2.CAP_PROP_FPS) or fps
        if abs(f2 - fps) > 0.1:
            print(f"WARNING: fps mismatch: {paths[0].name}={fps:.2f} vs {q.name}={f2:.2f}; "
                  "videos will drift out of sync")

    # Common tile width from each source's first frame (after height-normalizing).
    widths = []
    firsts = []
    for c, q in zip(caps, paths):
        ok, f = c.read()
        if not ok:
            sys.exit(f"no frames in {q}")
        firsts.append(f)
        widths.append(int(round(f.shape[1] * PANE_H / f.shape[0])))
    tile_w = max(widths)

    last = list(firsts)
    ended = [False] * len(caps)
    counts = [1] * len(caps)
    writer = None
    n_out = 0
    while True:
        tiles = [_tile(last[i], labels[i], ended[i], tile_w) for i in range(len(caps))]
        rows = [cv2.hconcat(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
        if len(tiles) % cols:  # odd count: pad the last row with a blank pane
            rows[-1] = cv2.hconcat([rows[-1], np.zeros_like(tiles[0])])
        grid = cv2.vconcat(rows)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (grid.shape[1], grid.shape[0]))
        writer.write(grid)
        n_out += 1

        for i, c in enumerate(caps):
            if ended[i]:
                continue
            ok, f = c.read()
            if ok:
                last[i] = f
                counts[i] += 1
            else:
                ended[i] = True
        if all(ended):
            break
    writer.release()
    for c in caps:
        c.release()

    for q, n in zip(paths, counts):
        print(f"{q}: {n} frames")
    print(f"wrote {out} ({n_out} frames, {fps:.1f} fps, {'x'.join(map(str, grid.shape[1::-1]))})")


if __name__ == "__main__":
    main()