#!/usr/bin/env python3
"""Convert a LeRobot EE_POS recording into the HDF5 the sim plant fit reads.

    python sysid/lerobot_to_hdf5.py ~/franka_data/sysid-8-28
    python sysid/lerobot_to_hdf5.py sysid-8-28 --episodes 0,1,2 --dry-run

Writes one multi-episode file in `excite_panda.py`'s ee_pose layout
(`data/<episode>/<field>`, `action_format = absolute_pose_quat`), which is what
`multi-fast/scripts/sysid/fit_sim_controller.py` loads through `fit.real_dir`.

The file lands in an `ee_pose/` subdirectory for excite_panda's reason:
`load_trajectories` globs `*.hdf5` non-recursively and applies whichever action
space the CALLER configured, so a directory holding both spaces double-counts
every episode and reads 0.4 m goal positions as deltas.

What is recorded, and what is derived
-------------------------------------
A recording keeps `observation.state` (7 joint angles + gripper) and the action
and nothing else, so three of the five fields the fit requires are reconstructed
here:

  qpos          the logged joint angles, verbatim.
  qvel          CENTRAL DIFFERENCE of qpos at the dataset's fps -- the recorder
                never stored measured dq. `fit.loss.w_dq` therefore scores a
                differentiated 20 Hz signal; set it to 0.0 to fit on qpos and the
                EE alone.
  eef_pos/quat  franka_fk of qpos: the FK scripts/replay_dataset.py relabels
                through, pinned against the control loop's own chain by
                test_osc_stack's round-trip test.
  action        the recorded EE_POS action after `ActionSafetyScreen`, i.e. the
                goal the controller actually pursued. Nothing else touches an
                EE_POS goal -- no delta envelope, no latched orientation and no
                recording noise, all of which live on the EE_DELTA branch (see
                `ee_goals.OSCGoalBuilder.absolute`).

Row alignment is already the fit's: LeRobot writes obs_t and then calls
send_action, so state precedes its action and sim step t scores against row t+1.

The leading rows
----------------
Every episode starts with home(), and the first published state can predate the
settle -- the episodes recorded so far open with one inter-row step of 6-10 rad/s
on a wrist joint, which no FR3 can do in 50 ms. `--trim-start auto` drops every
row up to the last step in the opening window that
`franka.max_joint_velocity_rad_s` rejects. It matters because the fit resets sim
to `init_qpos = qpos[0]` and rolls forward from there, so a bogus first row is a
transient CMA-ES would otherwise explain with plant parameters.

That limit is the conservative (Panda) set the config carries, so a genuinely
fast wrist swing can also exceed it -- the opening transients are 2-4x over,
a fast swing lands just above. Steps past the opening window are therefore
counted and reported, never trimmed: deleting an interior row would splice two
non-adjacent states into one dt, which is a worse lie than the one it fixes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import (  # noqa: E402
    eef_poses_from_qpos,
    flange_quat_to_o_t_ee,
)
from lerobot_robot_bimanual_franka.lerobot_source import (  # noqa: E402
    KD,
    KP,
    POS,
    QUAT,
    arm_prefix,
    check_action_space,
    load_frames,
    resolve_root,
    task_names,
)
from lerobot_robot_bimanual_franka.safety import ActionSafetyScreen  # noqa: E402

logger = logging.getLogger("lerobot_to_hdf5")

NUM_JOINTS = fc.num_joints()

_ACTION_COLUMNS = ["goal_x", "goal_y", "goal_z", "goal_qx", "goal_qy", "goal_qz", "goal_qw"]
_ACTION_FORMAT = "absolute_pose_quat"
_ACTION_SPACE = "EE_POS"


# ---------------------------------------------------------------------------
# Per-episode conversion
# ---------------------------------------------------------------------------

def _qvel_limit() -> np.ndarray:
    return np.asarray(fc.control("franka.max_joint_velocity_rad_s"), dtype=np.float64)


def leading_trim(qpos: np.ndarray, dt: float, max_trim: int) -> int:
    """Rows up to and including the LAST non-physical step in the opening window.

    Not "while the first step is non-physical": the settle shows up one row in as
    often as at row zero (it depends on where the first published state landed
    relative to home()), and stopping at the first physical step leaves the jump
    in the data with a clean row in front of it.
    """
    window = min(max_trim, max(len(qpos) - 2, 0))
    bad = np.any(np.abs(np.diff(qpos[:window + 1], axis=0)) / dt > _qvel_limit(), axis=1)
    return int(np.max(np.flatnonzero(bad)) + 1) if bad.any() else 0


def steps_over_qvel_limit(qpos: np.ndarray, dt: float) -> int:
    """Rows whose step implies more than the configured joint-velocity limit."""
    return int(np.sum(np.any(np.abs(np.diff(qpos, axis=0)) / dt > _qvel_limit(), axis=1)))


def convert_episode(actions: np.ndarray, states: np.ndarray, dt: float,
                    screen: ActionSafetyScreen, arm: str) -> tuple[dict, dict]:
    """One episode's frames -> the fit's arrays, plus what to report about them."""
    qpos = states[:, :NUM_JOINTS].astype(np.float64)
    # np.gradient is the central difference with one-sided ends: the least-biased
    # estimate of dq at the row it is written to.
    qvel = np.gradient(qpos, dt, axis=0)

    # `action` is an O_T_EE goal, so this needs the Hand-corrected pose, not the
    # bare flange franka_fk returns.
    eef_pos, eef_quat = eef_poses_from_qpos(qpos)

    goal_pos = np.empty((len(actions), 3))
    goal_quat = np.empty((len(actions), 4))
    shaped = 0
    for t, a in enumerate(actions):
        quat = a[QUAT] / max(float(np.linalg.norm(a[QUAT])), 1e-12)
        gp, gq = screen.shape_goal({arm: (a[POS].astype(np.float64), quat)})[arm]
        shaped += int(not np.array_equal(gp, a[POS]))
        goal_pos[t], goal_quat[t] = gp, gq

    arrays = {
        "action": np.concatenate([goal_pos, goal_quat], axis=1).astype(np.float32),
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "eef_pos": eef_pos.astype(np.float32),
        "eef_quat": eef_quat.astype(np.float32),
        # The goal IS the action in EE_POS; carried under both names so the file
        # matches what excite_panda writes and the sysid viz tools read.
        "eef_goal_pos": goal_pos.astype(np.float32),
        "eef_goal_quat": goal_quat.astype(np.float32),
    }
    track = np.linalg.norm(goal_pos - eef_pos, axis=1)
    stats = {
        "steps": len(actions),
        "safety_shaped_steps": shaped,
        "steps_over_qvel_limit": steps_over_qvel_limit(qpos, dt),
        "track_err_med_mm": float(np.median(track) * 1e3),
        "track_err_p95_mm": float(np.percentile(track, 95) * 1e3),
        "peak_qvel_rad_s": float(np.max(np.abs(qvel))),
    }
    return arrays, stats


def write_hdf5(path: Path, episodes: list[tuple[str, dict, dict]], root_attrs: dict) -> None:
    """`data/<episode>/<field>`, the group name `fit_sim_controller` hardcodes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        for key, val in root_attrs.items():
            f.attrs[key] = val
        grp = f.create_group("data")
        for name, arrays, attrs in episodes:
            ep = grp.create_group(name)
            for field, arr in arrays.items():
                ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
            for key, val in attrs.items():
                ep.attrs[key] = val
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert(
    source: str,
    out: Path,
    episodes: set[int] | None = None,
    trim_start: str = "auto",
    max_trim: int = 5,
    min_steps: int = 20,
    dry_run: bool = False,
) -> int:
    """One recording -> one multi-episode HDF5."""
    root = resolve_root(source)
    df, info = load_frames(root)
    fps = float(info["fps"])
    dt = 1.0 / fps
    if fps != float(fc.control_fps()):
        logger.warning("dataset fps %g != config control_fps %d; using the dataset's",
                       fps, fc.control_fps())

    arm = arm_prefix(info)
    arm_name = fc.profile(info["robot_type"]).arms[arm]
    screen = ActionSafetyScreen(
        {arm: fc.robot_base_in_world(arm_name)}, {arm: fc.ee_sphere(arm_name)}
    )
    tasks = task_names(root)
    logger.info("%s: %d episodes, %d frames, %g fps, arm %r -> %s",
                root, info["total_episodes"], info["total_frames"], fps, arm, arm_name)

    converted: list[tuple[str, dict, dict]] = []
    for ep_index, group in df.groupby("episode_index"):
        ep_index = int(ep_index)
        if episodes is not None and ep_index not in episodes:
            continue
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].to_numpy()).astype(np.float64)
        states = np.stack(group["observation.state"].to_numpy()).astype(np.float64)
        check_action_space(actions)

        trim = (leading_trim(states[:, :NUM_JOINTS], dt, max_trim)
                if trim_start == "auto" else int(trim_start))
        actions, states = actions[trim:], states[trim:]
        if len(actions) < min_steps:
            logger.warning("ep%03d: %d steps after trimming %d, below --min-steps %d; skipped",
                           ep_index, len(actions), trim, min_steps)
            continue

        arrays, stats = convert_episode(actions, states, dt, screen, arm)
        # The fit pins kp and damping_ratio, so a recording that moved them is not
        # describable by the file it is about to be written into.
        for col, label in ((KP, "kp"), (KD, "kd")):
            lo, hi = float(actions[:, col].min()), float(actions[:, col].max())
            if hi > lo:
                logger.warning("ep%03d: the %s action varies (%.3f..%.3f); the fit pins "
                               "the gains", ep_index, label, lo, hi)

        attrs = {
            "num_samples": stats["steps"],
            "action_format": _ACTION_FORMAT,
            "action_space": _ACTION_SPACE,
            "action_columns": _ACTION_COLUMNS,
            "init_qpos": arrays["qpos"][0].astype(np.float64),
            "fps": fps,
            "episode_index": ep_index,
            "task": tasks.get(int(group["task_index"].iloc[0]), ""),
            "trimmed_leading_rows": trim,
            "kp_action": float(actions[0, KP]),
            "kd_action": float(actions[0, KD]),
            "qvel_source": "central_difference",
            "eef_source": "franka_fk",
            "source_dataset": str(root),
            "arm": arm_name,
            **{k: v for k, v in stats.items() if k != "steps"},
        }
        converted.append((f"ep{ep_index:03d}", arrays, attrs))
        logger.info(
            "ep%03d: %4d steps (trimmed %d)  goal-vs-EE %.0f/%.0f mm med/p95  "
            "peak |qvel| %.2f rad/s  screened %d  over qvel limit %d",
            ep_index, stats["steps"], trim, stats["track_err_med_mm"],
            stats["track_err_p95_mm"], stats["peak_qvel_rad_s"],
            stats["safety_shaped_steps"], stats["steps_over_qvel_limit"])

    if not converted:
        logger.error("no episodes converted")
        return 1

    if dry_run:
        logger.info("dry run: would write %d trajectories to %s", len(converted), out)
        return 0

    write_hdf5(out, converted, {
        "source_dataset": str(root),
        "robot_type": info["robot_type"],
        "arm": arm_name,
        "fps": fps,
        "converter": "sysid/lerobot_to_hdf5.py",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    logger.info("wrote %d trajectories to %s", len(converted), out)
    logger.info("fit with: fit.real_dir=%s fit.traj_weights=null   (%s)",
                out.parent, ", ".join(name for name, _, _ in converted))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset root, or a repo id resolved under ~/franka_data")
    p.add_argument("--out", default=None,
                   help="output file (default ~/sysid/outputs/<name>/ee_pose/<name>.hdf5)")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices")
    p.add_argument("--trim-start", default="auto",
                   help="'auto' (drop non-physical leading rows) or a row count")
    p.add_argument("--max-trim", type=int, default=5,
                   help="cap on the rows --trim-start auto may drop")
    p.add_argument("--min-steps", type=int, default=20,
                   help="skip episodes shorter than this after trimming")
    p.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.setLevel(logging.INFO)   # the robot stack configures logging on import

    root = resolve_root(args.dataset)
    episodes = {int(x) for x in args.episodes.split(",")} if args.episodes else None
    out = Path(args.out).expanduser() if args.out else (
        Path.home() / "sysid/outputs" / root.name / "ee_pose" / f"{root.name}.hdf5")

    return convert(args.dataset, out, episodes=episodes, trim_start=args.trim_start,
                    max_trim=args.max_trim, min_steps=args.min_steps, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
