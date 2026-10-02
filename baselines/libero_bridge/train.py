#!/usr/bin/env python3
"""Train SAIL and B-Spline on the converted LIBERO tasks, one policy per task.

    python -m baselines.libero_bridge.train ~/franka_data/baseline_prep/libero_90 \
        --tasks-file baselines/libero_bridge/teacher_tasks.txt --steps 100000

A loop, not a new trainer: each task's HDF5 is handed to the unchanged
`baselines.{sail,bspline}_bridge.train` as a subprocess, so the sim and the
hardware policies are trained by exactly the same code. Both trainers derive
their output directory from the id the converter stamped on the file, so
checkpoints land under `~/franka_data/policies/<suite>/task_<i>/<backend>/`
without an --output-dir here.

Tasks are named by their index in the suite (`--tasks 9 29`), or by directory.

A task that already holds a checkpoint is skipped, so an interrupted sweep is
resumed by running the same command again (--retrain overrides that). With
--resume, an unfinished training continues from its newest checkpoint instead
of starting over.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from baselines.libero_bridge import sim_env  # noqa: E402
from baselines.run_record import read_source_repo_id  # noqa: E402
from baselines.train_common import policies_dir  # noqa: E402

logger = logging.getLogger("baselines.libero.train")

BACKENDS = ("sail", "bspline")
# Where each trainer leaves the file its rollout takes as --ckpt.
_CKPT_GLOB = {"sail": "*/models/model_epoch_*.pth",
              "bspline": "*/checkpoints/latest.ckpt"}
# What each trainer's --resume takes, from that checkpoint.
_RESUME_FROM = {"sail": lambda ckpt: ckpt,
                "bspline": lambda ckpt: ckpt.parent.parent}


def backend_dir(hdf5: Path, backend: str) -> Path | None:
    """`~/franka_data/policies/<suite>/task_<i>/<backend>`, from the id the
    converter stamped on the file."""
    repo_id = read_source_repo_id(hdf5)
    return None if not repo_id else policies_dir(repo_id, None) / backend


def checkpoint_for(hdf5: Path, backend: str) -> Path | None:
    """The newest checkpoint this task's `backend` policy already has, if any."""
    root = backend_dir(hdf5, backend)
    if root is None:
        return None
    found = glob.glob(str(root / _CKPT_GLOB[backend]))
    return Path(max(found, key=os.path.getmtime)) if found else None


def marker_for(hdf5: Path, backend: str) -> Path | None:
    """Where a FINISHED training records the budget it finished.

    The skip test cannot be "a checkpoint exists": both trainers checkpoint
    periodically, so an interrupted run leaves an early one behind and the task
    would be skipped as done and then rolled out half-trained.
    """
    root = backend_dir(hdf5, backend)
    return None if root is None else root / ".trained.json"


def finished_steps(hdf5: Path, backend: str) -> int | None:
    marker = marker_for(hdf5, backend)
    if marker is None or not marker.is_file():
        return None
    try:
        return int(json.loads(marker.read_text())["steps"])
    except Exception:
        return None


def mark_finished(hdf5: Path, backend: str, steps: int | None, ckpt: Path | None) -> None:
    marker = marker_for(hdf5, backend)
    if marker is None:
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "steps": steps, "backend": backend, "dataset": str(hdf5),
        "checkpoint": str(ckpt) if ckpt else None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


def converted(task_dir: Path) -> bool:
    return any((task_dir / f"{b}.hdf5").is_file() for b in BACKENDS)


def prep_suite(prep_dir: Path) -> str:
    """The LIBERO suite `libero_bridge.dataset` stamped on this directory's tasks."""
    import h5py

    found = set()
    for d in filter(converted, prep_dir.iterdir()):
        hdf5 = next(d / f"{b}.hdf5" for b in BACKENDS if (d / f"{b}.hdf5").is_file())
        with h5py.File(hdf5, "r") as f:
            found.add(str(f["data"].attrs.get("libero_suite", "")))
    if len(found) != 1 or "" in found:
        raise SystemExit(f"{prep_dir} should hold one LIBERO suite's tasks, found "
                         f"{sorted(found) or 'none'}")
    return found.pop()


def _index_order(task_dir: Path) -> tuple:
    # task_2 before task_10; name-tagged directories after, by name.
    index = sim_env.tag_index(task_dir.name)
    return (index is None, index or 0, task_dir.name)


def task_dirs(prep_dir: Path, names: list[str] | None) -> list[Path]:
    """Converted task directories by index (`9`), tag (`task_9`) or directory
    name; every one when `names` is empty.

    libero_90 was converted before tasks were filed by index, so its directories
    carry the task's name. An index finds those through the suite's own order.
    """
    if not names:
        return sorted(filter(converted, prep_dir.iterdir()), key=_index_order)
    suite = None
    found, missing = [], []
    for n in names:
        d = prep_dir / n
        index = sim_env.tag_index(n)
        if not converted(d) and index is not None:
            d = prep_dir / sim_env.task_tag(index)
            if not converted(d):
                suite = suite or prep_suite(prep_dir)
                d = prep_dir / sim_env.task_names(suite)[sim_env.resolve_task(suite, n)]
        if converted(d):
            found.append(d)
        else:
            missing.append(n)
    if missing:
        raise SystemExit(f"not converted under {prep_dir}: {', '.join(missing)}")
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prep_dir", type=Path,
                   help="directory of <task>/{sail,bspline}.hdf5 from libero_bridge.dataset")
    p.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="task indices in the suite (9), their tags (task_9) or directory "
                        "names; default every converted task")
    p.add_argument("--tasks-file", type=Path, default=None,
                   help="one task per line, as for --tasks; '#' comments ignored")
    p.add_argument("--steps", type=int, default=None,
                   help="gradient-step budget per policy; both trainers take it, so the "
                        "two are trained equally long")
    p.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    p.add_argument("--resume", action="store_true",
                   help="continue an unfinished training (checkpoints, no .trained.json) from its "
                        "newest checkpoint instead of starting over")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-task logs; default <prep_dir>/train_logs")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed to each trainer")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    names = list(args.tasks or [])
    if args.tasks_file:
        names += [ln.split("#")[0].strip() for ln in args.tasks_file.read_text().splitlines()
                  if ln.split("#")[0].strip()]
    tasks = task_dirs(args.prep_dir, names or None)
    log_dir = args.log_dir or (args.prep_dir / "train_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(d, b) for d in tasks for b in args.backend]
    logger.info("%d task(s) x %d backend(s) = %d training runs; logs in %s",
                len(tasks), len(args.backend), len(jobs), log_dir)

    failed, skipped, trained = [], 0, 0
    for i, (task_dir, backend) in enumerate(jobs, 1):
        hdf5 = task_dir / f"{backend}.hdf5"
        label = f"{task_dir.name}/{backend}"
        if not hdf5.is_file():
            logger.warning("[%d/%d] %s: no %s, skipping", i, len(jobs), label, hdf5.name)
            failed.append(f"{label} (no dataset)")
            continue
        done = None if args.retrain else finished_steps(hdf5, backend)
        if done is not None and (args.steps is None or done >= args.steps):
            logger.info("[%d/%d] %s: already trained to %s steps (%s)", i, len(jobs),
                        label, done, checkpoint_for(hdf5, backend))
            skipped += 1
            continue

        cmd = [sys.executable, "-m", f"baselines.{backend}_bridge.train", str(hdf5)]
        if args.steps is not None:
            cmd += ["--steps", str(args.steps)]
        resumable = args.resume and not args.retrain and done is None
        partial = checkpoint_for(hdf5, backend) if resumable else None
        if partial is not None:
            cmd += ["--resume", str(_RESUME_FROM[backend](partial))]
        cmd += list(args.extra)
        if args.dry_run:
            logger.info("[%d/%d] %s: %s", i, len(jobs), label, " ".join(cmd))
            continue

        log = log_dir / f"{task_dir.name}.{backend}.log"
        logger.info("[%d/%d] %s -> %s", i, len(jobs), label, log)
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=str(_REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if rc != 0:
            logger.error("   failed (exit %d) after %.1f min; see %s", rc, mins, log)
            failed.append(label)
            continue
        trained += 1
        ckpt = checkpoint_for(hdf5, backend)
        mark_finished(hdf5, backend, args.steps, ckpt)
        logger.info("   done in %.1f min -> %s", mins, ckpt)

    logger.info("trained %d, skipped %d, failed %d", trained, skipped, len(failed))
    for f in failed:
        logger.error("  failed: %s", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
