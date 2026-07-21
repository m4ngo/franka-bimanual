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
        stem = "episode" if args.episode is None else f"episode_{args.episode:03d}"
        cams = args.cams
        if not cams:
            # Auto-discover cameras with a video pair in BOTH base/ and resi/.
            def _discover(sub: str) -> set:
                return {q.name[len(stem) + 1:-4] for q in (run / sub).glob(f"{stem}_*.mp4")}
            cams = sorted(_discover("base") & _discover("resi"))
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
