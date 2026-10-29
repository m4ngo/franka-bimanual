#!/usr/bin/env python3
"""File an eval_distill.py comparison run beside the SAIL / B-Spline / dp rollouts.

    python -m baselines.libero_bridge.import_eval_distill \
        ~/checkpoints/<student>/libero_90_9/comparison.json \
        ~/checkpoints/<student>/libero_90_9/comparison_base.json

eval_distill.py runs ZAP (base dp + residual) and, with eval_base=True, the
base alone, in multi-fast's own process and container, where this package is
not installed. With `libero.comparison_env=True` it writes one comparison.json
per run: the environment it ran in and every start state's outcome. This turns
each into the run directory libero_bridge.rollout writes -- filed under the
base's training dataset, as method `multifast` or `dp` -- so
scripts/rollout_summary.py reads all four methods as one table, and
scripts/rollout_report.py draws them once they carry a --sweep-id.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "franka_config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import franka_config as fc  # noqa: E402

from baselines import run_record as rr  # noqa: E402
from baselines.force_log import ForceLog  # noqa: E402
from baselines.libero_bridge.rollout import SimEpisode  # noqa: E402
from baselines.libero_bridge.sim_env import task_names  # noqa: E402


def import_run(path: Path, root: Path, train_dataset: str | None, sweep: str | None) -> Path:
    doc = json.loads(path.read_text())
    hz = float(fc.control_fps())
    resolved = rr.resolve_train_dataset(train_dataset, doc["base_policy"].get("train_dataset"))
    run_dir = rr.RunDir(resolved["repo_id"], doc["method"], root=root)
    record = rr.RunRecord(run_dir, doc["method"], resolved, sweep=sweep)
    record.set("policy",
               source=str(path),
               base_policy=doc["base_policy"],
               student=doc["student"],
               zero_residual=doc["zero_residual"],
               control_mode="chunk_relative_target",
               control_mode_reason="eval_distill.py: the base's chunk-relative targets plus the "
                                   "residual, each slot one env step's absolute target")
    record.set("environment",
               simulator="libero/robosuite",
               runner="multi-fast eval_distill.py",
               plant=doc["comparison_env"]["plant"],
               comparison_env=doc["comparison_env"],
               # LIBERO's own name, as the rollouts record it: the report matches tasks by it.
               suite=doc["suite"], task_index=doc["task_id"],
               task=task_names(doc["suite"])[doc["task_id"]],
               impedance_mode=doc["impedance_mode"],
               control_freq=hz,
               max_steps=doc["max_episode_steps"],
               settle_steps=doc["comparison_env"]["settle_steps"],
               init_states=doc["comparison_env"]["init_states"],
               seed=doc["seed"])
    record.set("parameters", chunk_size=doc["chunk_size"], execution_horizon=doc["execution_horizon"],
               n_envs=doc["n_envs"])
    # A comparison.json from before eval_distill.py recorded the wrench has none.
    forces = None
    if all("ee_force" in ep for ep in doc["episodes"]):
        forces = ForceLog(run_dir.force_profiles_path, control_freq=hz)
        record.set("outputs", force_profiles=str(run_dir.force_profiles_path))
    for i, ep in enumerate(doc["episodes"]):
        t = round(float(ep["env_steps"]) / hz, 3)
        steps = int(ep["env_steps"])
        record.add_episode(SimEpisode(
            episode=i, init_state=int(ep["init_state"]), success=bool(ep["success"]),
            verdict="success" if ep["success"] else "timeout",
            wall_time_s=t, sim_time_s=t, steps=steps,
            # One base chunk, and one residual, per execution horizon.
            inferences=math.ceil(steps / doc["execution_horizon"]),
            ee_force_n=forces.add(i, ep["ee_force"], ep["ee_torque"]) if forces else None,
            notes={"env": ep["env"]}))
    record.finish("completed")
    return run_dir.path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="eval_distill comparison*.json")
    p.add_argument("--output-root", type=Path, default=None, help=f"default {rr.DEFAULT_ROOT}")
    p.add_argument("--train-dataset", default=None,
                   help="default: the id the dp server reported for the base")
    p.add_argument("--sweep-id", default=None,
                   help="tag the runs as one sweep, which is what rollout_report.py draws")
    args = p.parse_args()
    for f in args.files:
        out = import_run(f.expanduser(), args.output_root or rr.DEFAULT_ROOT, args.train_dataset,
                         args.sweep_id)
        print(f"{f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
