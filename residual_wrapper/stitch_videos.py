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
  python stitch_videos.py --runs <base-run> <residual-run> [--cams cam_2 cam_3] [-o out_dir]
    two run_residual.py run directories (the videos/ and episodes.jsonl of each):
    episode N of each run as one row of cameras, the runs stacked, with a race
    bar and a timer per row in the episode's own wall-clock seconds. Writes
    out_dir/episode_NNN.mp4 per pair and out_dir/all_episodes.mp4 (H.264).
"""

import argparse
import json
import subprocess
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


# --------------------------------------------------------------------------
# --runs: two rollout run directories raced against each other
# --------------------------------------------------------------------------

RACE_TILE = 360          # px per camera tile (width)
STAMP_ROWS = 18          # source rows holding run_residual's step/20 stamp; cropped
RACE_STRIP = 54          # px label/timer strip above each run's row
RACE_HEADER = 44         # px episode header
RACE_HOLD_S = 1.5        # hold the final frame after both episodes end
# BGR: the report's success blue / timeout orange, ink and muted grey.
RACE_OK, RACE_BAD = (214, 120, 42), (52, 104, 235)
RACE_INK, RACE_MUTED, RACE_BG = (245, 245, 245), (150, 150, 150), (24, 24, 24)
FONT = cv2.FONT_HERSHEY_DUPLEX


def _run_label(run: Path) -> str:
    m = json.loads((run / "manifest.json").read_text())
    pol = m.get("policy") or {}
    if pol.get("residual_enabled") is True:
        return "base + residual"
    if pol.get("residual_enabled") is False:
        return "base"
    return m["run"]["method"]


def _run_episodes(run: Path) -> dict:
    return {e["episode"]: e for e in map(json.loads, (run / "episodes.jsonl").read_text().splitlines())}


def _cam_role(cam: str) -> str:
    try:
        import franka_config as fc
        return fc.camera(cam).role or cam
    except Exception:  # noqa: BLE001
        return cam


def _read_all(path: Path) -> list:
    cap = _open(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = f[:f.shape[0] - STAMP_ROWS]
        frames.append(cv2.resize(f, (RACE_TILE, round(RACE_TILE * f.shape[0] / f.shape[1])),
                                 interpolation=cv2.INTER_CUBIC))
    cap.release()
    if not frames:
        sys.exit(f"no frames in {path}")
    return frames


def _text(img, s, org, scale, color, thick=1, anchor="left"):
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    x = org[0] - w if anchor == "right" else org[0]
    cv2.putText(img, s, (x, org[1]), FONT, scale, color, thick, cv2.LINE_AA)


def _race_row(frames_by_cam, k, ep, label, roles, t_max):
    """One run's strip (label, timer, race bar) above its camera tiles at frame k."""
    n = len(frames_by_cam[0])
    done = k >= n - 1
    dt = ep["wall_time_s"] / max(ep["steps"], 1)
    t = min(k, n - 1) * dt if not done else ep["wall_time_s"]
    width = RACE_TILE * len(frames_by_cam)
    strip = np.full((RACE_STRIP, width, 3), RACE_BG, np.uint8)
    good = ep.get("success")
    color = (RACE_OK if good else RACE_BAD) if done else RACE_INK
    _text(strip, label, (12, 34), 0.85, RACE_INK, 1)
    timer = f"{t:5.2f} s" + (f"  {ep.get('verdict', '')}" if done else "")
    _text(strip, timer, (width - 12, 34), 0.85, color, 2 if done else 1, anchor="right")
    (tw, _), _ = cv2.getTextSize(timer, FONT, 0.85, 2 if done else 1)
    _text(strip, f"step {min(k, n - 1)}", (width - 30 - tw, 34), 0.55, RACE_MUTED, 1, anchor="right")
    # Race bar: both rows share the slower episode's length as full scale.
    bar = int((width - 24) * min(t / t_max, 1.0))
    cv2.rectangle(strip, (12, RACE_STRIP - 10), (width - 12, RACE_STRIP - 6), (60, 60, 60), -1)
    cv2.rectangle(strip, (12, RACE_STRIP - 10), (12 + bar, RACE_STRIP - 6),
                  color if done else RACE_MUTED, -1)
    tiles = []
    for frames, role in zip(frames_by_cam, roles):
        f = frames[min(k, len(frames) - 1)].copy()
        if done:
            f = (f * 0.6).astype(np.uint8)
        for c, th in ((0, 3), (RACE_INK, 1)):
            cv2.putText(f, role, (8, 22), FONT, 0.55, (c, c, c) if c == 0 else c, th, cv2.LINE_AA)
        tiles.append(f)
    return cv2.vconcat([strip, cv2.hconcat(tiles)])


class _H264:
    """Raw BGR frames piped to ffmpeg: mp4v does not play in a browser or Slack."""

    def __init__(self, path: Path, size: tuple, fps: float):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            sys.exit(f"ffmpeg failed writing {self.p.args[-1]}")


def _run_pair_mode(runs: list, cams, out_dir: Path) -> None:
    labels = [_run_label(r) for r in runs]
    eps = [_run_episodes(r) for r in runs]
    if cams is None:
        names = [{q.name.split("_", 2)[2][:-4] for q in (r / "videos").glob("episode_*_*.mp4")}
                 for r in runs]
        cams = sorted(set.intersection(*names))
        if not cams:
            sys.exit("no camera recorded in both runs")
    roles = [_cam_role(c) for c in cams]
    shared = sorted(set(eps[0]) & set(eps[1]))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Play at the loop's measured rate, not the nominal 20 Hz, so the wall-clock
    # timers keep pace with the viewer's own clock.
    fps = round(sum(e[i]["steps"] for e in eps for i in shared)
                / sum(e[i]["wall_time_s"] for e in eps for i in shared), 2)
    tile_h = len(_read_all(runs[0] / "videos" / f"episode_{shared[0]:03d}_{cams[0]}.mp4")[0])
    width = RACE_TILE * len(cams)
    height = RACE_HEADER + len(runs) * (RACE_STRIP + tile_h)
    height += height % 2
    compiled = _H264(out_dir / "all_episodes.mp4", (width, height), fps)
    for i in shared:
        frames = [[_read_all(r / "videos" / f"episode_{i:03d}_{c}.mp4") for c in cams] for r in runs]
        pair = [e[i] for e in eps]
        t_max = max(e["wall_time_s"] for e in pair)
        n_out = max(len(f[0]) for f in frames) + int(RACE_HOLD_S * fps)
        head = "  vs  ".join(f"{lab} {e['wall_time_s']:.2f} s" for lab, e in zip(labels, pair))
        clip = _H264(out_dir / f"episode_{i:03d}.mp4", (width, height), fps)
        for k in range(n_out):
            header = np.full((RACE_HEADER, width, 3), (12, 12, 12), np.uint8)
            _text(header, f"episode {i}", (12, 30), 0.8, RACE_INK, 2)
            _text(header, head, (width - 12, 30), 0.6, RACE_MUTED, 1, anchor="right")
            rows = [_race_row(f, k, e, lab, roles, t_max) for f, e, lab in zip(frames, pair, labels)]
            canvas = cv2.vconcat([header] + rows)
            if canvas.shape[0] < height:  # libx264 needs even dimensions
                canvas = cv2.copyMakeBorder(canvas, 0, height - canvas.shape[0], 0, 0,
                                            cv2.BORDER_CONSTANT, value=RACE_BG)
            clip.write(canvas)
            compiled.write(canvas)
        clip.close()
        print(f"  episode {i}: " + ", ".join(f"{lab} {e['steps']} steps {e['wall_time_s']:.2f} s "
                                            f"{e.get('verdict')}" for lab, e in zip(labels, pair)))
    compiled.close()
    print(f"wrote {out_dir}/all_episodes.mp4 and {len(shared)} per-episode clips "
          f"({width}x{height}, {fps:g} fps)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", help="2-4 mp4 paths (alternative to --run-dir)")
    p.add_argument("--run-dir", default=None, help="dir containing base/ and resi/ episode videos")
    p.add_argument("--cams", nargs="+", default=None, help="camera names (run-dir mode)")
    p.add_argument("--episode", type=int, default=None,
                   help="episode number for recorded runs (episode_NNN_*); default: free-run naming")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--runs", nargs=2, default=None, metavar=("BASE_RUN", "RESIDUAL_RUN"),
                   help="two run_residual.py run directories to race episode by episode")
    args = p.parse_args()

    if args.runs:
        runs = [Path(r).expanduser() for r in args.runs]
        out = Path(args.output).expanduser() if args.output else runs[1].parent / "compare_videos"
        _run_pair_mode(runs, args.cams, out)
        return

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