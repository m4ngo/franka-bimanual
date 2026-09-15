#!/usr/bin/env python3
"""Compare the rollouts recorded under one task's output directory.

    python scripts/rollout_summary.py HuskyMango/pickup-bowl
    python scripts/rollout_summary.py --all
    python scripts/rollout_summary.py HuskyMango/pickup-bowl --json

Reads the manifests written by the three rollout entrypoints (baselines/ROLLOUT.md)
and prints one row per run. It only reads; nothing here touches the robot.

Success rate and time-to-success are the two numbers the comparison exists for,
so they lead. Time is over SUCCESSES only -- a failure's duration is the timeout
and says nothing about how fast a method is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.run_record import DEFAULT_ROOT  # noqa: E402


def load(run_dir: Path) -> dict | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"  ! unreadable manifest in {run_dir.name}: {exc}", file=sys.stderr)
        return None


def find_runs(root: Path, task: str | None) -> dict[str, list[Path]]:
    """task -> run directories, newest last."""
    base = root / task if task else root
    if not base.is_dir():
        return {}
    runs: dict[str, list[Path]] = {}
    for manifest in sorted(base.rglob("manifest.json")):
        run = manifest.parent
        key = str(run.parent.relative_to(root))
        runs.setdefault(key, []).append(run)
    return runs


def _fmt(value, spec="", dash="-"):
    return dash if value is None else format(value, spec)


def render(task: str, runs: list[Path]) -> None:
    print(f"\n{task}")
    head = (f"  {'run':<26} {'method':<10} {'eps':>4} {'ok':>4} {'rate':>6} "
            f"{'median s':>9} {'mean s':>8}  {'status':<12} notes")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for run in runs:
        doc = load(run)
        if doc is None:
            continue
        s, r = doc.get("summary", {}), doc.get("run", {})
        params = doc.get("parameters", {})
        notes = []
        if params.get("exec_fps"):
            notes.append(f"{params['exec_fps']:.0f}Hz")
        if params.get("speed_up_times", 1.0) not in (None, 1.0):
            notes.append(f"{params['speed_up_times']}x")
        if params.get("precision_modulation"):
            notes.append("precision")
        if params.get("eag"):
            notes.append("eag")
        if doc.get("policy", {}).get("residual_enabled") is False:
            notes.append("base-only")
        if s.get("aborted_episodes"):
            notes.append(f"{s['aborted_episodes']} aborted")
        if doc.get("environment", {}).get("git", {}).get("dirty"):
            notes.append("dirty-tree")
        print(f"  {r.get('run_id', run.name):<26} {r.get('method', '?'):<10} "
              f"{_fmt(s.get('episodes')):>4} {_fmt(s.get('successes')):>4} "
              f"{_fmt(s.get('success_rate'), '.0%'):>6} "
              f"{_fmt(s.get('median_time_to_success_s'), '.2f'):>9} "
              f"{_fmt(s.get('mean_time_to_success_s'), '.2f'):>8}  "
              f"{r.get('status', '?'):<12} {' '.join(notes)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", nargs="?", default=None,
                   help="training dataset id, e.g. HuskyMango/pickup-bowl")
    p.add_argument("--all", action="store_true", help="every task under the root")
    p.add_argument("--root", default=None, help=f"default {DEFAULT_ROOT}")
    p.add_argument("--json", action="store_true",
                   help="emit the manifests as one JSON array instead of a table")
    args = p.parse_args()

    if not args.task and not args.all:
        p.error("name a task, or pass --all")
    root = Path(args.root or DEFAULT_ROOT).expanduser()
    if not root.is_dir():
        print(f"no rollouts yet: {root} does not exist")
        return 1

    runs = find_runs(root, None if args.all else args.task)
    if not runs:
        print(f"no runs under {root / (args.task or '')}")
        return 1

    if args.json:
        docs = [load(r) for task in sorted(runs) for r in runs[task]]
        print(json.dumps([d for d in docs if d], indent=2))
        return 0
    for task in sorted(runs):
        render(task, runs[task])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
