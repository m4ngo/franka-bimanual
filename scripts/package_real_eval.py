#!/usr/bin/env python3
"""Bundle one task's real-arm evaluation into one folder and one zip.

    python scripts/package_real_eval.py HuskyMango/filter-10-6-bowl-plate
    python scripts/package_real_eval.py HuskyMango/<dataset> --since 20261020
    python scripts/package_real_eval.py HuskyMango/<dataset> --runs <run dir>="bsp 1x" ...

Picks one run per condition from ~/franka_data/outputs/<dataset>/: the newest
with at least --episodes episodes. The conditions are B-Spline at 1x, 2x and 4x,
SAIL at --sail-speed, the base diffusion policy alone ("base dp",
run_residual.py --no-residual) and with the ZAP residual ("base dp + ZAP").
Runs are matched by condition, not by checkpoint, so any checkpoint works; the
table printed first says which run, and which checkpoint, each condition got.
Writes ~/franka_data/analysis/<dataset>_real_eval_<YYYYMMDD>/ and a zip of it:

    summary.txt        one row per run: label, success, time, checkpoint, then rollout_summary.py's table
    charts/            one sheet per run and comparison.png (rollout_report.py --runs)
    stats/runs.json    per run: directory, checkpoints, gains, success, time, force
    stats/<label>/     the run's manifest.json and episodes.jsonl
    viz/<label>/       the per-episode 3D pages the rollouts write (missing baseline ones are rendered)
    videos/<label>/    episode_NNN_<verdict>.mp4 per episode and all_episodes.mp4, cut from
                       the dataset each run recorded, cameras side by side

Reads only what the rollouts wrote; nothing here touches the robot.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import rollout_report  # noqa: E402
import rollout_summary  # noqa: E402
from baselines.run_record import DEFAULT_ROOT  # noqa: E402

ANALYSIS_ROOT = Path.home() / "franka_data" / "analysis"
BSP_SPEEDS = (1.0, 2.0, 4.0)
# Banner above each clip, and the pixel scale of each camera tile under it.
BANNER_PX = 44
TILE_SCALE = 2
OK_RGB, BAD_RGB = (42, 120, 214), (235, 104, 52)   # rollout_report's success / failure pair


def slug(label: str) -> str:
    """rollout_report's file name for a label, so charts, videos and viz agree."""
    return label.replace(" ", "_").replace("+", "plus")


def condition(doc: dict, sail_speed: float) -> str | None:
    """The comparison column a run belongs to, or None."""
    method = doc["run"]["method"]
    speed = (doc.get("parameters") or {}).get("speed_up_times")
    if method == "bspline" and speed in BSP_SPEEDS:
        return f"bsp {speed:g}x"
    if method == "sail" and speed == sail_speed:
        return f"sail {speed:g}x"
    if method == "multifast":
        return "base dp + ZAP" if (doc.get("policy") or {}).get("residual_enabled") else "base dp"
    return None


def order(sail_speed: float) -> list[str]:
    return [*(f"bsp {s:g}x" for s in BSP_SPEEDS), f"sail {sail_speed:g}x", "base dp", "base dp + ZAP"]


def pick_runs(task_dir: Path, episodes: int, since: str | None, sail_speed: float) -> dict[str, Path]:
    """label -> the newest run of each condition with at least `episodes` episodes."""
    found: dict[str, Path] = {}
    for manifest in sorted(task_dir.glob("*/manifest.json")):   # run ids sort by time
        run = manifest.parent
        if since and run.name < since:
            continue
        doc = json.loads(manifest.read_text())
        if ((doc.get("summary") or {}).get("episodes") or 0) < episodes:
            continue
        label = condition(doc, sail_speed)
        if label:
            found[label] = run
    return {label: found[label] for label in order(sail_speed) if label in found}


def checkpoints(doc: dict) -> dict[str, str | None]:
    """The checkpoint(s) a run loaded: one for a baseline, base and residual for ours."""
    pol = doc.get("policy") or {}
    if doc["run"]["method"] == "multifast":
        return {"base": (pol.get("base_policy") or {}).get("path"),
                "residual": (pol.get("residual_policy") or {}).get("path")}
    return {"checkpoint": (pol.get("checkpoint") or {}).get("path")}


def link_or_copy(src: Path, dst: Path) -> None:
    """A hard link where the filesystem allows one: the pages are ~700 MB per bundle."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

class CameraReader:
    """One camera's frames, decoded forward through its video files episode by episode."""

    def __init__(self, dataset_root: Path, key: str) -> None:
        self.root, self.key = dataset_root, key
        self.path: Path | None = None
        self.container = None
        self.frames = None
        self.next_index = 0

    def read(self, chunk: int, file: int, start: int, n: int) -> list[np.ndarray]:
        import av

        path = self.root / "videos" / self.key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
        if path != self.path or start < self.next_index:
            self.close()
            self.path, self.container = path, av.open(str(path))
            self.frames, self.next_index = self.container.decode(video=0), 0
        out = []
        while len(out) < n:
            frame = next(self.frames, None)
            if frame is None:
                raise RuntimeError(f"{path} ended at frame {self.next_index}, wanted {start + n}")
            if self.next_index >= start:
                out.append(frame.to_ndarray(format="rgb24"))
            self.next_index += 1
        return out

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None


def h264_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)


def close_writer(proc: subprocess.Popen) -> None:
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")


def compose(tiles: list[np.ndarray], title: str, verdict: str, t: float) -> np.ndarray:
    import cv2

    body = np.concatenate([cv2.resize(i, None, fx=TILE_SCALE, fy=TILE_SCALE,
                                      interpolation=cv2.INTER_NEAREST) for i in tiles], axis=1)
    h, w = BANNER_PX + body.shape[0], body.shape[1]
    out = np.full((h + h % 2, w + w % 2, 3), 255, np.uint8)   # yuv420p wants even sides
    out[BANNER_PX:BANNER_PX + body.shape[0], :w] = body
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, title, (10, 29), font, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    right = f"{verdict}  {t:5.2f}s"
    (tw, _), _ = cv2.getTextSize(right, font, 0.7, 2)
    cv2.putText(out, right, (out.shape[1] - tw - 10, 29), font, 0.7,
                OK_RGB if verdict == "success" else BAD_RGB, 2, cv2.LINE_AA)
    return out


def export_videos(run: Path, label: str, out: Path) -> int:
    """episode_NNN_<verdict>.mp4 per recorded episode, and all of them in all_episodes.mp4."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = run / "dataset"
    if not (root / "meta" / "info.json").is_file():
        print(f"  ! {run.name}: no recorded dataset, no videos", file=sys.stderr)
        return 0
    meta = LeRobotDatasetMetadata(run.name, root=root)
    keys = sorted(meta.video_keys)
    log = run / "episodes.jsonl"
    rows = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()] if log.is_file() else []
    by_index = {r["dataset_episode_index"]: r for r in rows if r.get("dataset_episode_index") is not None}
    readers = {k: CameraReader(root, k) for k in keys}
    out.mkdir(parents=True, exist_ok=True)
    every = None
    try:
        for i in range(meta.total_episodes):
            ep = meta.episodes[i]
            n = int(ep["length"])
            tiles = {k: readers[k].read(int(ep[f"videos/{k}/chunk_index"]), int(ep[f"videos/{k}/file_index"]),
                                        int(round(float(ep[f"videos/{k}/from_timestamp"]) * meta.fps)), n)
                     for k in keys}
            row = by_index.get(i, {})
            verdict = row.get("verdict", "unknown")
            spliced = "  (spliced)" if (row.get("notes") or {}).get("spliced_from") else ""
            one = None
            for f in range(n):
                frame = compose([tiles[k][f] for k in keys], f"{label}   episode {i}{spliced}",
                                verdict, f / meta.fps)
                if one is None:
                    height, width = frame.shape[:2]
                    every = every or h264_writer(out / "all_episodes.mp4", width, height, meta.fps)
                    one = h264_writer(out / f"episode_{i:03d}_{verdict}.mp4", width, height, meta.fps)
                one.stdin.write(frame.tobytes())
                every.stdin.write(frame.tobytes())
            if one is not None:
                close_writer(one)
    finally:
        for r in readers.values():
            r.close()
        if every is not None:
            close_writer(every)
    return meta.total_episodes


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

def viz_pages(run: Path, method: str) -> list[Path]:
    pages = sorted(run.glob("episode_*.html"))
    if not pages and method in ("bspline", "sail"):
        from baselines.rollout_viz import render_run

        render_run(run)
        pages = sorted(run.glob("episode_*.html"))
    return pages


def run_record(label: str, run: Path, doc: dict) -> dict:
    s, p = doc.get("summary") or {}, doc.get("parameters") or {}
    kp, ratio = rollout_report.run_osc(doc)
    return {"label": label, "run_id": doc["run"]["run_id"], "run_dir": str(run),
            "method": doc["run"]["method"], "status": doc["run"]["status"],
            "speed": p.get("speed_up_times"), "osc_kp": kp, "osc_damping_ratio": ratio,
            "home_pose": (doc.get("environment") or {}).get("home_pose"),
            **checkpoints(doc),
            **{k: s.get(k) for k in ("episodes", "successes", "success_rate",
                                     "median_time_to_success_s", "mean_time_to_success_s",
                                     "ee_force_n")}}


def checkpoint_text(record: dict) -> str:
    if "checkpoint" in record:
        return str(record["checkpoint"])
    residual = record.get("residual")
    return f"{record.get('base')}" + (f" + {residual}" if residual else "")


def summary_text(repo_id: str, records: list[dict], runs: list[Path]) -> str:
    lines = [f"{repo_id}, real FR3, packaged {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"  {'label':<15} {'run':<27} {'eps':>4} {'ok':>4} {'rate':>5} {'median s':>9}  checkpoint"]
    for r in records:
        rate = "-" if r["success_rate"] is None else f"{r['success_rate']:.0%}"
        median = "-" if r["median_time_to_success_s"] is None else f"{r['median_time_to_success_s']:.2f}"
        lines.append(f"  {r['label']:<15} {r['run_id']:<27} {r['episodes']!s:>4} {r['successes']!s:>4} "
                     f"{rate:>5} {median:>9}  {checkpoint_text(r)}")
    table = io.StringIO()
    with contextlib.redirect_stdout(table):
        rollout_summary.render(repo_id, runs)
    return "\n".join(lines) + "\n" + table.getvalue()


def write_zip(folder: Path) -> Path:
    path = folder.with_suffix(".zip")
    tmp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                # mp4 and png are compressed already; deflating them again only costs time.
                kind = zipfile.ZIP_STORED if f.suffix in (".mp4", ".png") else zipfile.ZIP_DEFLATED
                z.write(f, f.relative_to(folder.parent), compress_type=kind)
    tmp.replace(path)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", help="the training dataset, e.g. HuskyMango/filter-10-6-bowl-plate")
    p.add_argument("--episodes", type=int, default=20, help="fewest episodes a run needs to count")
    p.add_argument("--since", default=None, help="only runs whose id sorts at or after this, e.g. 20261020")
    p.add_argument("--sail-speed", type=float, default=5.0, help="the SAIL --speed to report")
    p.add_argument("--runs", nargs="+", default=None, metavar="DIR=LABEL",
                   help="these run directories instead of picking; a label is optional")
    p.add_argument("--out", type=Path, default=None,
                   help=f"bundle folder; default {ANALYSIS_ROOT}/<dataset>_real_eval_<YYYYMMDD>")
    p.add_argument("--no-videos", action="store_true", help="skip the clips (the slow part)")
    args = p.parse_args()

    task_dir = Path(DEFAULT_ROOT).expanduser() / args.repo_id
    if args.runs:
        picked = {}
        for spec in args.runs:
            path, _, label = spec.partition("=")
            run = Path(path).expanduser().resolve()
            doc = json.loads((run / "manifest.json").read_text())
            picked[label or condition(doc, args.sail_speed) or run.name] = run
    else:
        if not task_dir.is_dir():
            p.error(f"no rollouts under {task_dir}")
        picked = pick_runs(task_dir, args.episodes, args.since, args.sail_speed)
        for label in order(args.sail_speed):
            if label not in picked:
                print(f"  ! no {label} run with at least {args.episodes} episodes under {task_dir}",
                      file=sys.stderr)
    if not picked:
        return 1

    docs = {label: json.loads((run / "manifest.json").read_text()) for label, run in picked.items()}
    records = [run_record(label, picked[label], docs[label]) for label in picked]
    print(f"{args.repo_id}: {len(picked)} runs")
    for r in records:
        print(f"  {r['label']:<15} {r['run_id']:<27} {r['successes']}/{r['episodes']}  {checkpoint_text(r)}")
    bases = {r["base"] for r in records if "base" in r}
    if len(bases) > 1:
        print(f"  ! base dp and base dp + ZAP ran different base policies: {sorted(map(str, bases))}",
              file=sys.stderr)

    name = args.repo_id.rstrip("/").split("/")[-1]
    out = (args.out or ANALYSIS_ROOT / f"{name}_real_eval_{time.strftime('%Y%m%d')}").expanduser()
    if out.exists():
        print(f"replacing {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    runs = list(picked.values())
    labels = list(picked)
    (out / "summary.txt").write_text(summary_text(args.repo_id, records, runs))
    (out / "stats").mkdir()
    (out / "stats" / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
    for label, run in picked.items():
        (out / "stats" / slug(label)).mkdir()
        for f in ("manifest.json", "episodes.jsonl"):
            if (run / f).is_file():
                shutil.copy2(run / f, out / "stats" / slug(label) / f)

    print("charts:")
    (out / "charts").mkdir()
    rollout_report.real_report(runs, labels, out / "charts")

    for label, run in picked.items():
        pages = viz_pages(run, docs[label]["run"]["method"])
        for page in pages:
            link_or_copy(page, out / "viz" / slug(label) / page.name)
        print(f"viz {label}: {len(pages)} pages")

    if not args.no_videos:
        for label, run in picked.items():
            n = export_videos(run, label, out / "videos" / slug(label))
            print(f"videos {label}: {n} episodes")

    zipped = write_zip(out)
    host = socket.gethostname().split(".")[0]
    print(f"\nwrote {out}\nand   {zipped} ({zipped.stat().st_size / 1e6:.0f} MB)")
    print(f"on your PC:\n  scp {getpass.getuser()}@{host}:{zipped} .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
