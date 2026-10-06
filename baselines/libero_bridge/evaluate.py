#!/usr/bin/env python3
"""Roll every trained LIBERO baseline out in sim and record its success rate.

    python -m baselines.libero_bridge.evaluate ~/franka_data/baseline_prep/libero_90 \
        --tasks-file baselines/libero_bridge/teacher_tasks.txt --num-episodes 20

A loop over `scripts/libero_rollout.sh`, which is what owns a policy server's
lifetime. One server per (task, backend): loading a diffusion checkpoint costs
tens of seconds, but sharing one across tasks would silently evaluate the
previous task's checkpoint the moment a task was skipped.

Runs in the workspace venv -- the shell script picks the policy's interpreter
and multi-fast's for the env. Read the results afterwards with

    python scripts/rollout_summary.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from baselines.libero_bridge.train import (  # noqa: E402
    BACKENDS, checkpoint_for, finished_steps, prep_suite, task_dirs,
)
from baselines.libero_bridge.sim_env import tag_index  # noqa: E402
from baselines.run_record import DEFAULT_ROOT, read_source_repo_id  # noqa: E402

logger = logging.getLogger("baselines.libero.evaluate")

ROLLOUT_SH = _REPO_ROOT / "scripts" / "libero_rollout.sh"

# The two trained here, plus multi-fast's base policy. pi05 is a pretrained
# checkpoint rather than something this repo trains, so it has no dataset and
# nothing to gate on -- it just runs.
EVAL_BACKENDS = (*BACKENDS, "pi05")
PRETRAINED = ("pi05",)


def sweep_runs(root: Path, sweep_id: str, suite: str, task_dir: Path,
               backend: str) -> list[tuple[Path, str | None]]:
    """(run dir, status) of every run of this task and backend tagged `sweep_id`."""
    index = tag_index(task_dir.name)
    found = []
    for manifest in root.rglob("manifest.json"):
        try:
            m = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        run, env = m.get("run") or {}, m.get("environment") or {}
        same_task = (env.get("task_index") == index if index is not None
                     else env.get("task") == task_dir.name)
        if (run.get("sweep") == sweep_id and run.get("method") == backend
                and env.get("suite") == suite and same_task):
            found.append((manifest.parent, run.get("status")))
    return found


def summarise_sweep(sweep_id: str) -> None:
    """Print this invocation's rollouts as one table, per task and per backend."""
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "rollout_summary.py"),
           "--sweep", sweep_id]
    subprocess.call(cmd, cwd=str(_REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prep_dir", type=Path,
                   help="directory of <task>/{sail,bspline}.hdf5; the checkpoints are "
                        "found from the id each one carries")
    p.add_argument("--backend", nargs="+", choices=EVAL_BACKENDS,
                   default=list(EVAL_BACKENDS))
    p.add_argument("--tasks", nargs="+", default=None,
                   help="task indices (9), tags (task_9) or directory names; default all")
    p.add_argument("--tasks-file", type=Path, default=None,
                   help="one task per line, as for --tasks; '#' comments ignored")
    p.add_argument("--suite", default=None,
                   help="read from prep_dir's files; passing a different one is an error")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="per-rollout logs; default <prep_dir>/rollout_logs")
    p.add_argument("--allow-unfinished", action="store_true",
                   help="roll out a task whose training has not written .trained.json. "
                        "Both trainers checkpoint periodically, so without this the "
                        "sweep would silently evaluate a half-trained policy")
    p.add_argument("--sweep-id", default=None,
                   help="tag every rollout of this invocation with this id; default "
                        "sweep_<timestamp>. Recall it later with "
                        "rollout_summary.py --sweep <id>")
    p.add_argument("--resume", action="store_true",
                   help="skip a task and backend this sweep already completed, and move "
                        "an unfinished run of it to <outputs>_unfinished/ before rerunning "
                        "it, so the sweep's pooled rows count each episode once")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed to each rollout")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    names = list(args.tasks or [])
    if args.tasks_file:
        names += [ln.split("#")[0].strip() for ln in args.tasks_file.read_text().splitlines()
                  if ln.split("#")[0].strip()]
    tasks = task_dirs(args.prep_dir, names or None)
    # An index means a different task in every suite, so the env's suite comes
    # from the files, never from a default.
    suite = prep_suite(args.prep_dir)
    if args.suite and args.suite != suite:
        p.error(f"{args.prep_dir} holds {suite} tasks, not {args.suite}")
    log_dir = args.log_dir or (args.prep_dir / "rollout_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_id = args.sweep_id or f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    jobs = [(d, b) for d in tasks for b in args.backend]
    logger.info("%d task(s) x %d backend(s) = %d rollouts of %d episodes; logs in %s",
                len(tasks), len(args.backend), len(jobs), args.num_episodes, log_dir)
    logger.info("sweep id %s -- recall with: python scripts/rollout_summary.py --sweep %s",
                sweep_id, sweep_id)

    extra = list(args.extra)
    out_root = (Path(extra[extra.index("--output-root") + 1]).expanduser()
                if "--output-root" in extra[:-1] else DEFAULT_ROOT)

    failed, missing, ran = [], 0, 0
    for i, (task_dir, backend) in enumerate(jobs, 1):
        label = f"{task_dir.name}/{backend}"
        if args.resume:
            runs = sweep_runs(out_root, sweep_id, suite, task_dir, backend)
            done_run = next((d for d, status in runs if status == "completed"), None)
            if done_run is not None:
                logger.info("[%d/%d] %s: %s already completed this sweep, skipping",
                            i, len(jobs), label, done_run.name)
                continue
            for d, status in runs:
                dest = out_root.parent / f"{out_root.name}_unfinished" / d.relative_to(out_root)
                logger.warning("[%d/%d] %s: moving %s run %s to %s", i, len(jobs), label,
                               status, d.name, dest)
                if not args.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(d), str(dest))
        ckpt = done = None
        if backend not in PRETRAINED:
            hdf5 = task_dir / f"{backend}.hdf5"
            ckpt = checkpoint_for(hdf5, backend) if hdf5.is_file() else None
            if ckpt is None:
                logger.warning("[%d/%d] %s: no checkpoint yet, skipping", i, len(jobs), label)
                missing += 1
                continue
            done = finished_steps(hdf5, backend)
            if done is None and not args.allow_unfinished:
                logger.warning("[%d/%d] %s: training has not finished (%s is a periodic "
                               "checkpoint); skipping. --allow-unfinished overrides",
                               i, len(jobs), label, ckpt.name)
                missing += 1
                continue

        cmd = ["bash", str(ROLLOUT_SH), "--backend", backend,
               "--suite", suite, "--task", task_dir.name,
               "--num-episodes", str(args.num_episodes),
               "--sweep-id", sweep_id]
        if ckpt is not None:
            cmd += ["--start-server", "--ckpt", str(ckpt)]
        if backend in PRETRAINED:
            # Filed beside the baselines trained on this task, however its
            # directory is named.
            stamp = next(filter(None, (read_source_repo_id(task_dir / f"{b}.hdf5")
                                       for b in BACKENDS)), None)
            if stamp:
                cmd += ["--train-dataset", stamp]
        if args.save_video:
            cmd.append("--save-video")
        cmd += list(args.extra)
        if args.dry_run:
            logger.info("[%d/%d] %s: %s", i, len(jobs), label, " ".join(cmd))
            continue

        log = log_dir / f"{task_dir.name}.{backend}.log"
        logger.info("[%d/%d] %s (%s)", i, len(jobs), label,
                    "pretrained" if ckpt is None else f"{ckpt.name}, {done} steps")
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=str(_REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if rc != 0:
            logger.error("   failed (exit %d) after %.1f min; see %s", rc, mins, log)
            failed.append(label)
            continue
        ran += 1
        # The rollout prints its own rate; echo it so the sweep is readable
        # without opening every log.
        for line in reversed(log.read_text().splitlines()):
            if "success rate" in line:
                logger.info("   %s  (%.1f min)", line.strip(), mins)
                break

    logger.info("rolled out %d, skipped %d without a finished checkpoint, failed %d",
                ran, missing, len(failed))
    for f in failed:
        logger.error("  failed: %s", f)
    if ran and not args.dry_run:
        summarise_sweep(sweep_id)
    logger.info("recall this sweep: python scripts/rollout_summary.py --sweep %s", sweep_id)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
