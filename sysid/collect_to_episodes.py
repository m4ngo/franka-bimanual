#!/usr/bin/env python3
"""Convert a multi-fast task-rollout file into an episode file (EPISODE_HDF5.md).

    python sysid/collect_to_episodes.py multi-fast/logs/sysid_task/<run>/data.hdf5 <out.hdf5>
    python sysid/collect_to_episodes.py <run>/data.hdf5 <out.hdf5> --kind lifted

`multi-fast/scripts/sysid/collect_task_rollouts.py` writes the older collect
layout: groups `traj_libero_*` (the policy acting in the LIBERO scene) and
`traj_lifted_*` (the same action sequence replayed from `lifted_init_qpos` in
a bare env), a normalised 7-dim OSC delta as `action`, a `skeleton` dataset,
and none of the frame attrs. `plot_dataset_coverage.py` reads that as is, but
`merge_episodes.py` validates against the schema and refuses it. This writes
the same trajectories in the layout everything else reads, so a task rollout
can be merged with a real recording:

  action          `[eef_goal_pos, eef_goal_quat]` -- the goal the controller was
                  sent, `absolute_pose_quat` / `EE_POS`, the convention the sim
                  replays and the real sysid recordings use.
  policy_action   the recorded 7-dim action, `osc_delta_norm7`.
  frame           `base_sim` / `robosuite_grip_site`. The collector records
                  robosuite's world-frame `eef_pos` and OSC goal; the base body's
                  world position (row 0 of the recorded `skeleton`) is subtracted
                  here, as `replay_goals_in_sim.py` does, and kept as
                  `sim_base_pos_world`. LIBERO places the base without rotating
                  it, so a translation is the whole transform. Against a real
                  `base` file the EE panels then differ only by the grip site to
                  flange offset (97 mm along the gripper's z).
  obs_timing      `post_period`: the recorder logs the state after each action.
                  A lifted episode starts at `lifted_init_qpos`; a libero one
                  never had its start recorded, so `init_qpos` is row 0 and
                  `init_qpos_source` says so.

Episodes are named `<suite>_t<task>[_lifted]_r<NN>`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-fast"))

from utils.sysid import episode_hdf5  # noqa: E402

KINDS = {"libero": "task_rollout_libero", "lifted": "task_rollout_lifted"}
CARRIED = ("qpos", "qvel", "eef_pos", "eef_quat", "eef_goal_pos", "eef_goal_quat",
           "eef_lin_vel", "eef_ang_vel", "tau_cmd", "t_sim")


def convert(name: str, arrays: dict, attrs: dict, fps: float | None) -> tuple:
    kind = next(k for k, v in KINDS.items() if v == attrs["kind"])
    n = int(attrs["num_samples"])
    out = {k: np.asarray(arrays[k], dtype=np.float32) for k in CARRIED if k in arrays}
    base = np.asarray(arrays["skeleton"][:, 0, :], dtype=np.float64)
    if np.ptp(base, axis=0).max() > 1e-6:
        raise ValueError(f"{name}: the base body moves during the episode ({np.ptp(base, axis=0)})")
    base = base[0]
    for k in ("eef_pos", "eef_goal_pos"):
        out[k] = (out[k].astype(np.float64) - base).astype(np.float32)
    out["action"] = np.concatenate([out["eef_goal_pos"], out["eef_goal_quat"]], axis=1)
    out["policy_action"] = np.asarray(arrays["action"], dtype=np.float32)
    if fps is None:
        dt = np.median(np.diff(np.asarray(arrays["t_sim"], dtype=np.float64)))
        fps = float(round(1.0 / dt))
    if kind == "lifted":
        init_qpos, init_src = np.asarray(attrs["lifted_init_qpos"], dtype=np.float64), "lifted_init_qpos"
    else:
        init_qpos, init_src = out["qpos"][0].astype(np.float64), "first_row"
    rollout = name.rsplit("rollout", 1)[1]
    ep_name = f"{attrs['suite']}_t{int(attrs['task_id'])}{'_lifted' if kind == 'lifted' else ''}_r{rollout}"
    ep_attrs = {
        "num_samples": n, "fps": fps,
        "frame": "base_sim", "quat_order": "xyzw", "ee_convention": "robosuite_grip_site",
        "action_format": "absolute_pose_quat", "action_space": "EE_POS",
        "policy_action_format": "osc_delta_norm7", "obs_timing": "post_period",
        "init_qpos": init_qpos, "init_qpos_source": init_src, "sim_base_pos_world": base,
        "ee_pos0": out["eef_pos"][0].astype(np.float64), "ee_quat0": out["eef_quat"][0].astype(np.float64),
        "source": "sim", "kind": attrs["kind"], "suite": attrs["suite"], "task_id": int(attrs["task_id"]),
        "task_name": attrs["task_name"], "task_language": attrs.get("task_language"),
        "success": bool(attrs.get("success", False)), "policy_source": attrs.get("policy_source"),
    }
    return ep_name, out, ep_attrs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="collect_task_rollouts.py data.hdf5")
    ap.add_argument("out", help="episode file to WRITE")
    ap.add_argument("--kind", choices=("libero", "lifted", "both"), default="libero",
                    help="which half to convert (default: libero, the task rollout itself)")
    ap.add_argument("--fps", type=float, default=None, help="override the rate read off t_sim")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    if out.exists() and not args.overwrite:
        ap.error(f"{out} exists; pass --overwrite to replace it")
    wanted = {KINDS[k] for k in (("libero", "lifted") if args.kind == "both" else (args.kind,))}
    root = episode_hdf5.read_root_attrs(src)
    episodes = [convert(name, arrays, attrs, args.fps)
                for name, arrays, attrs, _ in episode_hdf5.read_episodes(src)
                if attrs.get("kind") in wanted]
    if not episodes:
        raise SystemExit(f"{src} holds no {args.kind} trajectories")
    problems = [p for e in episodes for p in episode_hdf5.validate_episode(*e, legacy_ok=False)]
    if problems:
        raise SystemExit("not written:\n  " + "\n  ".join(problems))
    episode_hdf5.write_episodes(out, episodes, producer="sysid/collect_to_episodes.py", root_attrs={
        "converted_from": str(src),
        "plant": root.get("plant"), "policy_source": root.get("policy_source"),
        "git_sha": root.get("git_sha"), "task_name": root.get("task_name"),
    })
    for name, arrays, attrs in episodes:
        print(f"  {name:<28} {attrs['num_samples']:>5} steps  success={attrs['success']}")
    print(f"wrote {len(episodes)} episode(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
