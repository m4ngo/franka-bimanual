#!/usr/bin/env python3
"""Replay recorded LIBERO demos through the sim rollout's own action path.

    multi-fast/.venv/bin/python scripts/check_libero_sim_rollout.py --task 9
    multi-fast/.venv/bin/python scripts/check_libero_sim_rollout.py --task 9 \
        --targets reached --osc-kp 300 --osc-damping-ratio 0.5

The rollout drives LIBERO through ABSOLUTE pose targets, which the env does not
take: `sim_env.SimTask.action` inverts each one into the normalised OSC delta.
By default this checks that inverse the only way that settles it -- by feeding a
demo's own recorded targets (`goal_pos` / `goal_ori`, what B-Spline trains on)
through the executor and asking whether the task still succeeds. A failure
there is a bug in the executor, not in a policy: the targets replayed are the
ones the demonstration itself produced.

`--targets reached` instead replays what SAIL trains on -- the poses the arm
reached -- on SAIL's own schedule: precision-labelled rows at `slow_fps`, the
rest at `fast_fps`, as `rollout.sail_episode` plays them. That is a measurement,
not a pass/fail: it asks whether a controller can follow SAIL's targets at
SAIL's speed with no policy in the loop (the SAIL paper's App. G.1). The labels
come from the converted `sail.hdf5`, so SAIL's AWE pass must have run on it.

Runs in multi-fast/.venv (robosuite + libero). No policy and no ZMQ.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "franka_config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import franka_config as fc  # noqa: E402

from baselines.libero_bridge import sim_env  # noqa: E402
from baselines.libero_bridge.rollout import Stepper  # noqa: E402
from baselines.libero_bridge.train import task_dirs  # noqa: E402
from baselines.policy_math import slowdown_mode  # noqa: E402

DEFAULT_SOURCE = Path.home() / "libero_data"
DEFAULT_PREP = Path.home() / "franka_data" / "baseline_prep"
SAIL_KEY = "absolute_actions_with_precision"

# Low-dim only: the point is the action path, and rendering 256px frames for
# every step of every demo doubles the runtime for nothing.
SHAPES = {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2]}


def sail_schedule(rows: np.ndarray, fast_fps: float) -> list[float]:
    """The rate each row is sent at, by the rollout's precision rule."""
    fast = fast_fps
    slow = float(fc.policy("baselines.exec.slow_fps"))
    window = int(fc.policy("baselines.sail.slowdown_window_size"))
    executed, out = [], []
    for i, row in enumerate(rows):
        executed.append(row)
        out.append(slow if slowdown_mode(executed, row, rows[i:], window) else fast)
    return out


def count_saturation(task: sim_env.SimTask, counts: dict) -> None:
    """Count control substeps with any joint torque at its limit. Re-run after
    every reset: a hard reset builds a new robot."""
    robot = task.env.env.robots[0]
    control = robot.control

    def counted(action, policy_step=False):
        control(action, policy_step)
        low, high = robot.torque_limits
        counts["substeps"] += 1
        counts["saturated"] += int(np.any((robot.torques <= low) | (robot.torques >= high)))
    robot.control = counted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--task", required=True,
                   help="the task's index in --suite (9), its tag (task_9), or its LIBERO name")
    p.add_argument("--source-dir", type=Path, default=None,
                   help=f"regenerated LIBERO demos; default {DEFAULT_SOURCE}/<suite>")
    p.add_argument("--prep-dir", type=Path, default=None,
                   help=f"--targets reached: converted tasks; default {DEFAULT_PREP}/<suite>")
    p.add_argument("--demos", type=int, default=5, help="how many demos to replay")
    p.add_argument("--slack", type=float, default=1.5,
                   help="step budget as a multiple of the demo's own length")
    p.add_argument("--targets", choices=("commanded", "reached"), default="commanded")
    p.add_argument("--fast-fps", type=float, default=None,
                   help="--targets reached: rate unlabelled rows play at; default "
                        "baselines.exec.fast_fps. 20 is the demos' own speed")
    p.add_argument("--osc-kp", type=float, default=None, help="default robosuite's 150")
    p.add_argument("--osc-damping-ratio", type=float, default=None, help="default 1.0")
    args = p.parse_args()

    source_dir = args.source_dir or (DEFAULT_SOURCE / args.suite)
    task_name = sim_env.task_names(args.suite)[sim_env.resolve_task(args.suite, args.task)]
    src = source_dir / f"{task_name}_demo.hdf5"
    if not src.is_file():
        raise SystemExit(f"{src} does not exist")
    labelled = None
    fast_fps = float(args.fast_fps or fc.policy("baselines.exec.fast_fps"))
    if args.targets == "reached":
        d, = task_dirs(args.prep_dir or (DEFAULT_PREP / args.suite), [str(args.task)])
        labelled = d / "sail.hdf5"

    osc = {"kp": args.osc_kp, "damping_ratio": args.osc_damping_ratio}
    # 128 px: nothing here looks at an image, and the env still has to render.
    task = sim_env.SimTask(args.suite, args.task, resolution=128,
                           controller={k: v for k, v in osc.items() if v is not None})
    live = sim_env.osc_report(task.controller)
    print(f"{args.suite}/{task.name}  (task {task.task_index}): {task.language}")
    print(f"{args.targets} targets; osc kp {live['kp'][0]:g} damping "
          f"{live['damping_ratio'][0]:g} step limit {100 * live['output_max'][0]:g} cm")

    passed = failed = 0
    times_ok, lengths = [], []
    counts = {"substeps": 0, "saturated": 0}
    with h5py.File(src, "r") as f, (h5py.File(labelled, "r") if labelled
                                    else contextlib.nullcontext()) as g:
        names = sorted(f["data"], key=lambda k: int(k.split("_")[1]))[: args.demos]
        # The converter renumbers demos from 0 in source order.
        converted = sorted(g["data"], key=lambda k: int(k.split("_")[1])) if g is not None else []
        for i, name in enumerate(names):
            d = f["data"][name]
            actions = d["actions"][()]
            budget = int(len(actions) * args.slack)
            stepper = Stepper(task, SHAPES, frame_stack=1, max_steps=budget)
            stepper.begin(0, state=d["states"][()][0])
            count_saturation(task, counts)
            errors = []
            if args.targets == "commanded":
                goal_pos, goal_ori = d["goal_pos"][()], d["goal_ori"][()]
                for t in range(len(actions)):
                    if stepper.finished:
                        break
                    # The delta the executor derives for this target, against the
                    # action the demo actually sent from the same state.
                    got = task.action(goal_pos[t], goal_ori[t], float(actions[t][6]))
                    errors.append(float(np.abs(got[:6] - actions[t][:6]).max()))
                    stepper.send(goal_pos[t], goal_ori[t], float(actions[t][6]), stepper.control_freq)
            else:
                rows = g["data"][converted[i]][SAIL_KEY][()].astype(np.float64)
                if len(rows) != len(actions):
                    raise SystemExit(f"{labelled} demo {i} has {len(rows)} rows, {name} "
                                     f"{len(actions)}; reconvert")
                # Row 0 is the pose the arm starts at; the rollout never enters there.
                schedule = sail_schedule(rows, fast_fps)
                for row, hz in zip(rows[1:], schedule[1:]):
                    stepper.send(row[:3], row[3:6], float(row[6]), hz)
                while not stepper.finished:
                    stepper.send(rows[-1, :3], rows[-1, 3:6], float(rows[-1, 6]), stepper.control_freq)
            ok = stepper.done
            passed += ok
            failed += not ok
            lengths.append(len(actions) / stepper.control_freq)
            if ok:
                times_ok.append(stepper.sim_time)
            print(f"  {name}: {'SUCCESS' if ok else 'FAILED '} in {stepper.sim_time:5.2f} s sim "
                  f"(demo took {len(actions) / stepper.control_freq:.2f} s)"
                  + (f", delta-vs-recorded max {max(errors):.2e} mean {np.mean(errors):.2e}"
                     if errors else ""))

    task.close()
    total = passed + failed
    print(f"\n{passed}/{total} demos succeeded in {np.mean(times_ok) if times_ok else float('nan'):.2f} "
          f"s sim on average (the demos are {np.mean(lengths):.2f} s long); a joint torque at "
          f"its limit on {counts['saturated'] / max(1, counts['substeps']):.1%} of control substeps")
    if args.targets == "reached":
        return 0
    if failed:
        print("FAIL -- the executor does not reproduce the demonstrations")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
