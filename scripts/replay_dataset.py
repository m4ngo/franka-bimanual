#!/usr/bin/env python3
"""Re-record an existing LeRobot dataset on the arm, optionally converting its
EE_DELTA actions to equivalent EE_POS ones.

    --mode delta     replay the source actions verbatim through EE_DELTA
    --mode ee_pose   convert each step to the ABSOLUTE OSC goal that delta would
                     have produced, and replay that through EE_POS

Both modes drive the real arm and capture fresh observations, because the point
is to re-measure the source trajectories under the CURRENT controller -- these
datasets were recorded before `torque.osc.cross_coupling_compensation` was turned
off, so the old observations no longer describe how this stack behaves.

What `ee_pose` converts, exactly
-------------------------------
`BimanualFranka` in EE_DELTA builds its goal as

    goal_pos  = measured_ee_pos + fudge * clip(delta_pos)
    goal_ori  = latched, rewritten only on a nonzero rotation delta

so the equivalent absolute action is that goal. It is computed OFFLINE from the
source episode's own recorded joint angles (forward kinematics), not live from
the replaying arm -- computing it live would just be EE_DELTA under another name.
The goal sequence is therefore identical to the source run's, and the two modes
are directly comparable: they command the same goals and differ only in how each
step is re-anchored.

Both use the SAME `OSCGoalBuilder` and `ActionSafetyScreen` the robot uses, so
the conversion cannot drift from the control path.

Caveats worth knowing before trusting the output
------------------------------------------------
- **Torques are not bit-identical to the source run**, and cannot be: the
  controller changed since. What is reproduced is the goal sequence.
- **The anchor is one frame stale.** LeRobot records `obs_t` and then calls
  `send_action`, which re-reads the arm state; the true anchor sat between
  `obs_t` and `obs_{t+1}`. `obs_t` is the closest thing the dataset preserves.
- **Recording noise is unrecoverable.** If the source was recorded with
  `--noise True`, the perturbation was added inside `send_action` and never
  stored. Conversion runs noise-free; pass `--use-noise` to add fresh noise.
- **The gripper column changes meaning.** `{arm}_gripper` is now an ABSOLUTE
  normalized position; in these datasets the recorded action column is a delta.
  Replaying it verbatim would drive the gripper to whatever the number means
  today, so the target is taken from the measured `observation.state` gripper,
  which is what the arm actually held.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
import numpy as np  # noqa: E402
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.pipeline_features import (  # noqa: E402
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.video_utils import VideoEncodingManager  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

from lerobot_robot_bimanual_franka import ControlMode, SingleArmFrankaConfig  # noqa: E402
from lerobot_robot_bimanual_franka.ee_goals import OSCGoalBuilder, delta_rotvec  # noqa: E402
from lerobot_robot_bimanual_franka.franka_fk import franka_fk  # noqa: E402
from lerobot_robot_bimanual_franka.safety import ActionSafetyScreen  # noqa: E402
from sysid.lerobot_to_hdf5 import flange_quat_to_o_t_ee  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

logger = logging.getLogger(__name__)

_PROFILE = "single_arm_franka"
_ARM = next(iter(fc.profile(_PROFILE).arms))
NUM_JOINTS = fc.num_joints()

# Source action layout, and the target's: identical shape, different meaning for
# the first seven entries.
POS, QUAT, GRIP = slice(0, 3), slice(3, 7), 7


def to_ee_pose_actions(actions: np.ndarray, states: np.ndarray, cfg) -> np.ndarray:
    """One episode of EE_DELTA actions -> the absolute EE_POS actions that command
    the same OSC goals.

    `actions` (T, 10) and `states` (T, 8) come straight out of the source dataset.
    Uses the robot's own goal builder and safety screen, so this is the control
    path rather than a copy of it.
    """
    goals = OSCGoalBuilder(
        translation_fudge=cfg.ee_translation_fudge,
        rotation_fudge=cfg.ee_rotation_fudge,
        use_noise=cfg.use_noise,
        noise_pos_scale=cfg.noise_pos_scale,
        noise_rot_scale=cfg.noise_rot_scale,
    )
    safety = ActionSafetyScreen(
        {_ARM: cfg.base_in_world(_ARM)}, {_ARM: fc.ee_sphere(cfg.arm_name(_ARM))}
    )

    out = np.zeros_like(actions)
    for t, (a, s) in enumerate(zip(actions, states)):
        ee_pos, ee_quat = franka_fk(np.asarray(s[:NUM_JOINTS], dtype=np.float64))
        # OSCGoalBuilder anchors on the arm's O_T_EE; franka_fk gives the flange.
        ee_quat = flange_quat_to_o_t_ee(ee_quat[None, :])[0]
        if t == 0:
            # robosuite reset_goal(): the first step's held orientation is the pose
            # the episode starts from, matching connect()/home() on the real robot.
            goals.reset(_ARM, ee_quat)
        goal = goals.from_delta(_ARM, a[POS], delta_rotvec(a[QUAT]), ee_pos, ee_quat)
        goal_pos, goal_quat = safety.shape_goal({_ARM: goal})[_ARM]
        out[t, POS], out[t, QUAT] = goal_pos, goal_quat
        # Absolute normalized target; see the module docstring on why this comes
        # from the measured column rather than the source action column.
        out[t, GRIP] = s[NUM_JOINTS]
        out[t, 8:] = a[8:]                    # kp, kd pass through unchanged
    return out


def bound_deltas(actions: np.ndarray, max_pos: float | None,
                 max_rot: float | None) -> tuple[np.ndarray, int, int]:
    """Scale each step's delta down to a magnitude bound, preserving direction.

    `--mode delta` only. With `cross_coupling_compensation` off, a rotation command
    leaks into translation (osc.py's own behaviour -- the flange walks), so replaying
    a trajectory recorded WITH the compensation can be more volatile than the run it
    came from. This caps how much any single step may ask for.

    NOT a third limit layer in the sense CLAUDE.md forbids: it never runs on the
    control path. It rewrites the recorded action sequence before replay, and the
    bounded value is what gets written to the new dataset, so what is stored is
    exactly what was sent and the output replays as itself.

    Scaled by NORM rather than clipped per axis, unlike `clip_delta`: that one
    reproduces osc.py's per-axis `scale_action` and is a parity mechanism, while this
    is an operator's volatility limit, and a per-axis clip would rotate the commanded
    direction rather than just shorten it.
    """
    out = actions.copy()
    n_pos = n_rot = 0
    for t, a in enumerate(actions):
        if max_pos is not None:
            mag = float(np.linalg.norm(a[POS]))
            if mag > max_pos:
                out[t, POS] = a[POS] * (max_pos / mag)
                n_pos += 1
        if max_rot is not None:
            rotvec = delta_rotvec(a[QUAT])
            angle = float(np.linalg.norm(rotvec))
            if angle > max_rot:
                out[t, QUAT] = Rotation.from_rotvec(rotvec * (max_rot / angle)).as_quat()
                n_rot += 1
    return out, n_pos, n_rot


def _source_task(source, meta, frame_index: int) -> str:
    """The episode's task string, without decoding the frame's video."""
    try:
        idx = int(source.hf_dataset[frame_index]["task_index"])
        return meta.tasks.index[idx] if hasattr(meta.tasks, "index") else str(meta.tasks[idx])
    except Exception:
        return ""


def episode_slice(meta, ep_idx: int) -> tuple[int, int]:
    return (int(meta.episodes["dataset_from_index"][ep_idx]),
            int(meta.episodes["dataset_to_index"][ep_idx]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo-id", required=True)
    p.add_argument("--target-repo-id", required=True)
    p.add_argument("--mode", choices=("delta", "ee_pose"), required=True)
    p.add_argument("--output-dir", required=True, help="local root for the new dataset")
    p.add_argument("--source-root", default=None, help="local root of the source dataset")
    p.add_argument("--task", default=None, help="single_task string; default: the source's")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="source episode indices; default all")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--fps", type=int, default=None, help="default: the source's")
    p.add_argument("--depth", action="store_true",
                   help="enable depth observations (off by default; these sources have none)")
    p.add_argument("--use-noise", action="store_true",
                   help="add fresh EE_DELTA goal noise; see the docstring")
    p.add_argument("--translation-fudge", type=float, default=None,
                   help="ee_pose mode ONLY: fudge used to convert deltas to goals. "
                        "Rejected in delta mode, where deltas replay as recorded.")
    p.add_argument("--rotation-fudge", type=float, default=None,
                   help="ee_pose mode ONLY; see --translation-fudge")
    p.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    p.add_argument("--max-delta-pos-m", type=float, default=None,
                   help="delta mode ONLY: cap each step's translation-delta norm (m). "
                        "Direction preserved. Use when cross_coupling_compensation is "
                        "off and the replay is more volatile than the source run.")
    p.add_argument("--max-delta-rot-rad", type=float, default=None,
                   help="delta mode ONLY: cap each step's rotation-delta angle (rad)")
    p.add_argument("--overwrite", action="store_true",
                   help="delete a non-empty --output-dir first (e.g. the half-written "
                        "meta/ a failed run leaves behind)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="convert and report divergence stats without touching the arm")
    args = p.parse_args()

    init_logging()
    source = LeRobotDataset(args.source_repo_id, root=args.source_root)
    meta = source.meta
    fps = args.fps or meta.fps

    control_mode = ControlMode.EE_DELTA if args.mode == "delta" else ControlMode.EE_POS
    cfg = SingleArmFrankaConfig(control_mode=control_mode, depth=args.depth, use_noise=args.use_noise)

    # The fudges are an EE_POS-mode CONVERSION knob and nothing else. In delta mode
    # the recorded deltas are replayed untouched and the live `tuning:` block scales
    # them exactly as it would on any run today -- that is the whole point of the
    # delta re-record. Setting a fudge there would rescale the replay to chase the
    # source run's torques, which is the opposite of re-measuring under the current
    # controller, so it is refused rather than silently honoured.
    if args.mode == "delta" and (args.translation_fudge is not None
                                 or args.rotation_fudge is not None):
        p.error("--translation-fudge/--rotation-fudge apply to --mode ee_pose only. "
                "In delta mode the recorded deltas are replayed as recorded and "
                "tuning.ee_*_fudge from config/control.yaml scales them live.")
    # And the mirror image: ee_pose exists to re-encode the source run's commands
    # exactly, so a volatility bound there would silently change the very goals it
    # is supposed to preserve.
    if args.mode == "ee_pose" and (args.max_delta_pos_m is not None
                                   or args.max_delta_rot_rad is not None):
        p.error("--max-delta-pos-m/--max-delta-rot-rad apply to --mode delta only. "
                "ee_pose reproduces the source run's goal sequence exactly; bounding "
                "it would defeat that. Bound the delta replay instead.")
    if args.translation_fudge is not None:
        cfg.ee_translation_fudge = args.translation_fudge
    if args.rotation_fudge is not None:
        cfg.ee_rotation_fudge = args.rotation_fudge

    logger.info("%s -> %s | mode=%s control_mode=%s fps=%d", args.source_repo_id,
                args.target_repo_id, args.mode, control_mode.value, fps)
    if args.mode == "delta":
        logger.info("actions replayed AS RECORDED (no rescaling); the live tuning block "
                    "scales them: translation_fudge=%.3f rotation_fudge=%.3f use_noise=%s",
                    cfg.ee_translation_fudge, cfg.ee_rotation_fudge, cfg.use_noise)
    else:
        logger.info("conversion trims: translation_fudge=%.3f rotation_fudge=%.3f use_noise=%s",
                    cfg.ee_translation_fudge, cfg.ee_rotation_fudge, cfg.use_noise)
    logger.info("controller: default_kp=%s cross_coupling_compensation=%s uncouple_pos_ori=%s",
                fc.control("torque.osc.default_kp"),
                fc.control("torque.osc.cross_coupling_compensation"),
                fc.control("torque.osc.uncouple_pos_ori"))

    ep_indices = args.episodes if args.episodes is not None else list(range(meta.total_episodes))
    if args.max_episodes is not None:
        ep_indices = ep_indices[:args.max_episodes]

    # Phase 1: pull every episode's actions/states out and convert. Done up front
    # so a conversion error surfaces before the arm has moved at all.
    # Read actions/states off hf_dataset, NOT through source[i]: the latter decodes
    # this frame's video for every camera, which is ~35 s per episode of work for
    # columns the conversion never touches.
    cols = source.hf_dataset.select_columns(["action", "observation.state"]).with_format("numpy")
    episodes = []
    bounded_pos = bounded_rot = 0
    for ep in ep_indices:
        lo, hi = episode_slice(meta, ep)
        rows = cols[lo:hi]
        actions = np.asarray(rows["action"], dtype=np.float64)
        states = np.asarray(rows["observation.state"], dtype=np.float64)
        task = args.task or _source_task(source, meta, lo)
        if args.mode == "ee_pose":
            out = to_ee_pose_actions(actions, states, cfg)
        else:
            out, n_pos, n_rot = bound_deltas(actions, args.max_delta_pos_m,
                                             args.max_delta_rot_rad)
            bounded_pos += n_pos
            bounded_rot += n_rot
            out[:, GRIP] = states[:, NUM_JOINTS]     # same gripper-semantics fix
        episodes.append(dict(index=ep, actions=out, states=states, task=task))
    total = sum(len(e["actions"]) for e in episodes)
    logger.info("converted %d episodes, %d frames", len(episodes), total)
    if args.mode == "delta" and (args.max_delta_pos_m or args.max_delta_rot_rad):
        logger.info("bound applied: %d/%d frames scaled on translation (cap %s m), "
                    "%d/%d on rotation (cap %s rad)",
                    bounded_pos, total, args.max_delta_pos_m,
                    bounded_rot, total, args.max_delta_rot_rad)

    if args.dry_run:
        for e in episodes[:5]:
            a = e["actions"]
            logger.info("  ep %3d  %4d frames  goal_pos range %s",
                        e["index"], len(a), np.round(np.ptp(a[:, POS], axis=0), 4))
        logger.info("dry run: arm untouched")
        return 0

    # Before touching the arm: a partial output dir from an earlier failed run makes
    # LeRobotDataset.create build on top of stale metadata. Caught here rather than
    # after 35 minutes of replay.
    out_dir = Path(args.output_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            p.error(f"{out_dir} exists and is not empty. Pass --overwrite to replace it, "
                    f"or choose a different --output-dir.")
        logger.warning("removing existing %s", out_dir)
        shutil.rmtree(out_dir)

    robot = make_robot_from_config(cfg)
    _, _, robot_obs_proc = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=make_default_processors()[0],
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True),
    )
    n_cams = len(robot.cameras)
    dataset = LeRobotDataset.create(
        args.target_repo_id, fps, root=out_dir, robot_type=robot.name,
        features=features, use_videos=True, image_writer_processes=0,
        image_writer_threads=4 * n_cams, batch_encoding_size=1, vcodec="auto",
        streaming_encoding=True, encoder_queue_maxsize=8, encoder_threads=2,
    )

    robot.connect()
    try:
        with VideoEncodingManager(dataset):
            for e in episodes:
                _replay_episode(robot, dataset, e, fps, args)
    finally:
        robot.disconnect()

    if args.push_to_hub:
        logger.info("pushing %s ...", args.target_repo_id)
        dataset.push_to_hub(private=args.private)
    logger.info("done: %d episodes at %s", dataset.num_episodes, args.output_dir)
    return 0


def _replay_episode(robot, dataset, ep, fps, args) -> None:
    """Home to the episode's own start configuration, then step its actions."""
    import time

    actions, states = ep["actions"], ep["states"]
    q0 = states[0, :NUM_JOINTS]
    if not robot.home(home_q_left=None, home_q_right=q0,
                      gripper_norm=float(states[0, NUM_JOINTS]),
                      max_time_s=args.home_max_time_s, fps=fps):
        logger.warning("ep %d: homing did not converge; replaying anyway", ep["index"])

    period = 1.0 / fps
    keys = list(robot.action_features)
    drift = []
    for t, a in enumerate(actions):
        tick = time.perf_counter()
        obs = robot.get_observation()
        action = {k: float(v) for k, v in zip(keys, a)}
        robot.send_action(action)

        # The task rides INSIDE the frame dict; add_frame takes no task kwarg.
        frame = {**build_dataset_frame(dataset.features, obs, prefix="observation"),
                 **build_dataset_frame(dataset.features, action, prefix="action"),
                 "task": ep["task"]}
        dataset.add_frame(frame)

        # How far the arm is from the trajectory the source recorded. EE_POS goals
        # are absolute and never re-anchored, so this is the number that says
        # whether the replay is still tracking the original run.
        q = np.array([obs[f"{_ARM}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)])
        drift.append(float(np.max(np.abs(q - states[t, :NUM_JOINTS]))))

        elapsed = time.perf_counter() - tick
        if elapsed < period:
            time.sleep(period - elapsed)

    dataset.save_episode()
    logger.info("ep %d: %d frames, joint drift vs source max %.4f rad, mean %.4f rad",
                ep["index"], len(actions), max(drift), float(np.mean(drift)))


if __name__ == "__main__":
    raise SystemExit(main())
