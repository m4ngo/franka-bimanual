#!/usr/bin/env python3
"""Compare the rollouts recorded under one task's output directory.

The positional argument is any path PREFIX under ~/franka_data/outputs, so the
same command narrows to a suite, a task or one single run:

    python scripts/rollout_summary.py --all                     # every run
    python scripts/rollout_summary.py libero_90                 # one suite
    python scripts/rollout_summary.py libero_10/task_2          # one task
    python scripts/rollout_summary.py libero_90/KITCHEN_SCENE9_turn_on_the_stove/20260924_170208-bspline
    python scripts/rollout_summary.py HuskyMango/pickup-bowl --json

Reads the manifests written by the three rollout entrypoints (baselines/ROLLOUT.md)
and prints one row per run. It only reads; nothing here touches the robot.

Success rate and time-to-success are the two numbers the comparison exists for,
so they lead. Time is over SUCCESSES only -- a failure's duration is the timeout
and says nothing about how fast a method is. Sim runs also show the wrist force
as eval_fast.py reports it: each episode's mean, p95 and max |F|, averaged over
episodes, in newtons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.libero_bridge.sim_env import tag_index  # noqa: E402
from baselines.run_record import DEFAULT_ROOT, force_summary, osc_text  # noqa: E402

FORCE_HEAD = "|F| N avg/p95/max"


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


def _force(f: dict | None) -> str:
    return "-" if not f else f"{f['mean']:.1f}/{f['p95']:.0f}/{f['max']:.0f}"


def sweep_of(run_dir: Path) -> str | None:
    """The sweep id a run was tagged with, or None for a standalone rollout."""
    return ((load(run_dir) or {}).get("run") or {}).get("sweep")


def resolve_sweep(root: Path, sweep: str) -> str | None:
    """`latest` -> the newest sweep id under root; anything else passes through."""
    if sweep != "latest":
        return sweep
    ids = {s for m in root.rglob("manifest.json") if (s := sweep_of(m.parent))}
    return max(ids) if ids else None


def only_sweep(runs: dict[str, list[Path]], sweep: str) -> dict[str, list[Path]]:
    kept = {task: [r for r in paths if sweep_of(r) == sweep]
            for task, paths in runs.items()}
    return {task: paths for task, paths in kept.items() if paths}


def render_sweep(sweep: str, runs: dict[str, list[Path]]) -> None:
    """One row per backend and suite: the sweep pooled, which is the comparison.
    A backend that ran several suites also gets a row pooling all of them."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for paths in runs.values():
        for run in paths:
            doc = load(run)
            if doc:
                doc["_run_dir"] = run
                key = (doc.get("run", {}).get("method", "?"),
                       (doc.get("environment") or {}).get("suite") or "-")
                groups.setdefault(key, []).append(doc)
    rows = []
    for method in sorted({m for m, _ in groups}):
        suites = sorted(u for m, u in groups if m == method)
        rows += [(method, u, groups[(method, u)]) for u in suites]
        if len(suites) > 1:
            rows.append((method, "all", [d for u in suites for d in groups[(method, u)]]))
    print(f"\n{sweep} -- pooled over tasks")
    head = (f"  {'method':<10} {'suite':<10} {'tasks':>5} {'eps':>5} {'ok':>5} {'rate':>6} "
            f"{'mean median s':>14} {FORCE_HEAD:>17}  notes")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for method, suite, docs in rows:
        eps = sum(d["summary"].get("episodes") or 0 for d in docs)
        ok = sum(d["summary"].get("successes") or 0 for d in docs)
        meds = [d["summary"]["median_time_to_success_s"] for d in docs
                if d["summary"].get("median_time_to_success_s") is not None]
        speeds = {d.get("parameters", {}).get("speed_up_times") for d in docs}
        speeds.discard(None)
        notes = f"{speeds.pop()}x" if len(speeds) == 1 else ""
        oscs = {osc_text(d.get("environment")) for d in docs}
        if len(oscs) == 1 and (osc := oscs.pop()):
            notes = ", ".join(n for n in (notes, osc) if n)
        # Pooled over episodes, as the report sheets pool them.
        force = force_summary([e for d in docs for e in load_episodes(d)])
        print(f"  {method:<10} {suite:<10} {len(docs):>5} {eps:>5} {ok:>5} "
              f"{_fmt(ok / eps if eps else None, '.1%'):>6} "
              f"{_fmt(sum(meds) / len(meds) if meds else None, '.2f'):>14} "
              f"{_force(force):>17}  {notes}")


def load_episodes(doc: dict) -> list[dict]:
    path = doc["_run_dir"] / "episodes.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def render(task: str, runs: list[Path]) -> None:
    # A LIBERO task is filed by index (libero_10/task_2), which names nothing.
    language = ((load(runs[0]) or {}).get("environment") or {}).get("language")
    tagged = tag_index(task.rsplit("/", 1)[-1]) is not None
    print(f"\n{task}" + (f"  {language}" if tagged and language else ""))
    head = (f"  {'run':<26} {'method':<10} {'eps':>4} {'ok':>4} {'rate':>6} "
            f"{'median s':>9} {'mean s':>8} {FORCE_HEAD:>17}  {'status':<12} notes")
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
        if osc := osc_text(doc.get("environment")):
            notes.append(f"[{osc}]")
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
              f"{_fmt(s.get('mean_time_to_success_s'), '.2f'):>8} "
              f"{_force(s.get('ee_force_n')):>17}  "
              f"{r.get('status', '?'):<12} {' '.join(notes)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", nargs="?", default=None,
                   help="any path prefix under --root: a training dataset id "
                        "(HuskyMango/pickup-bowl), a suite (libero_90), a suite/task, "
                        "or one <timestamp>-<method> run directory")
    p.add_argument("--all", action="store_true", help="every task under the root")
    p.add_argument("--sweep", default=None,
                   help="only the rollouts tagged with this sweep id (evaluate.py "
                        "prints it), plus a pooled row per backend. 'latest' picks "
                        "the newest sweep on disk")
    p.add_argument("--root", default=None, help=f"default {DEFAULT_ROOT}")
    p.add_argument("--json", action="store_true",
                   help="emit the manifests as one JSON array instead of a table")
    args = p.parse_args()

    if not args.task and not args.all and not args.sweep:
        p.error("name a task, or pass --all, or pass --sweep <id>")
    root = Path(args.root or DEFAULT_ROOT).expanduser()
    if not root.is_dir():
        print(f"no rollouts yet: {root} does not exist")
        return 1

    runs = find_runs(root, None if (args.all or not args.task) else args.task)
    if not runs:
        print(f"no runs under {root / (args.task or '')}")
        return 1

    sweep = None
    if args.sweep:
        sweep = resolve_sweep(root, args.sweep)
        if sweep is None:
            print("no sweep-tagged runs on disk; only rollouts started through "
                  "baselines.libero_bridge.evaluate carry a sweep id")
            return 1
        runs = only_sweep(runs, sweep)
        if not runs:
            print(f"no runs tagged {sweep!r}")
            return 1

    if args.json:
        docs = [load(r) for task in sorted(runs) for r in runs[task]]
        print(json.dumps([d for d in docs if d], indent=2))
        return 0
    for task in sorted(runs):
        render(task, runs[task])
    if sweep:
        render_sweep(sweep, runs)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
