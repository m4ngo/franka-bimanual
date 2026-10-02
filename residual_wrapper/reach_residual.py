"""The reach task on the real arm under the analytic base policy plus a FAST
residual -- multi-fast's SB3 checkpoint (ft_policy_<N>_steps.zip / final.zip).
run_residual.py dispatches here when its --residual-policy is that .zip.

The loop is eval_fast.py's. The checkpoint's own `predict_diffused` draws the
base chunk and composes the residual onto it exactly as it did in sim; this
module supplies what the sim env did around that call: the observation dict
(ReachObservationWrapper, in sim's world frame) and the execution of a chunk of
chunk-start-relative targets against the measured pose (ActionChunkWrapper's
two stages). RealReach owns the curve, the cursor, success and the wire.

Each episode can run the base alone, the residual on it, or both in succession
on the same curve (`policy`). Everything lands in one <out>/<timestamp>/:
episodes.hdf5 (base frame, EPISODE_HDF5.md; one group per episode and variant)
and one viz.py HTML per episode and variant -- the base and total chunk
forecasts, gains, gripper and residual rotation, with the reach curve drawn
for reference and, on a residual episode that followed a base one, that base
run's actual trail.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
for _p in ("scripts", "multi-fast", "multi-fast/stable-baselines3", "."):
    sys.path.insert(0, str(_ROOT / _p))

import franka_config as fc  # noqa: E402
from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot_robot_bimanual_franka import (  # noqa: E402
    ControlMode, SingleArmFrankaConfig, SingleArmRightConfig,
)
from lerobot_robot_bimanual_franka.reach_record import ReachEpisodeRecorder  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach import RealReach  # noqa: E402
from lerobot_robot_bimanual_franka.real_reach_geometry import (  # noqa: E402
    S, base_to_world, keep_out_sphere, safety_z_floor_world,
)
from real_reach_rollout import flush, run_metadata, warn_off_limits  # noqa: E402
from baselines.force_log import force_attrs, force_stats, note as force_note  # noqa: E402
from stable_baselines3 import FAST  # noqa: E402
from utils.base_policy_utils import load_base_policy  # noqa: E402
from viz import EpisodeRecorder, save_episode_html, save_rollout_html  # noqa: E402

PRODUCER = "residual_wrapper/run_residual.py"
# Which variants one episode runs, in order. "both" pairs them on one curve.
VARIANTS = {"residual": ("residual",), "base": ("base",), "both": ("base", "residual")}


def load_residual(path: str, device: str) -> FAST:
    """The checkpoint carries its training config, so the base policy it was
    trained against is rebuilt from it by multi-fast's own factory. The replay
    buffer that config would allocate (400k x 8 envs of dict obs) is shrunk to
    nothing; the warning is base_stats_path, a file on the training host."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FAST.load(path, device=device, custom_objects={"buffer_size": 1, "n_envs": 1})
    # Position-only on this side: the orientation curve is not ported, and the
    # base's orientation branch needs robosuite, which the workstation lacks.
    cfg = OmegaConf.merge(model.cfg, {"reach": {"orient_delta_max_deg": 0.0}})
    model.diffusion_policy = load_base_policy(cfg)
    return model


def check_contract(cfg, env: RealReach) -> None:
    """The checkpoint's task numbers against what RealReach read from
    config/reach.yaml. A residual trained against one curve density or delta
    envelope is meaningless against another, and nothing downstream would notice."""
    pairs = {
        "reach.n_waypoints": (cfg.reach.n_waypoints, env.n_waypoints),
        "reach.n_segments": (cfg.reach.n_segments, fc.section("reach")["curve"]["n_segments"]),
        "reach.advance_threshold": (cfg.reach.advance_threshold, env.advance_threshold),
        "reach.success_threshold": (cfg.reach.success_threshold, env.success_threshold),
        "env.max_episode_steps": (cfg.env.max_episode_steps, env.max_episode_steps),
        "reach.control_freq": (cfg.reach.get("control_freq", 20.0), fc.control_fps()),
        "base_policy.osc_output_max": (cfg.base_policy.osc_output_max, env.osc_output_max),
        "base_policy.osc_rot_output_max": (cfg.base_policy.osc_rot_output_max, env.osc_rot_output_max),
    }
    bad = {k: v for k, v in pairs.items() if not np.isclose(v[0], v[1])}
    if bad:
        raise ValueError(f"checkpoint (first) disagrees with config/reach.yaml (second): {bad}")
    if cfg.base_policy.goal_mode != "target":
        raise ValueError(f"this executor inverts chunk-start-relative targets; the checkpoint's "
                         f"base runs goal_mode={cfg.base_policy.goal_mode!r}")
    if env.include_orient:
        raise ValueError("reach.orientation.delta_max_deg is on, but the base runs "
                         "position-only here; set it to 0 for this runner")
    # Distribution shift, not a contract break: say it once and run.
    if cfg.reach.orient_delta_max_deg > 0.0:
        print(f"NOTE: residual trained with a {cfg.reach.orient_delta_max_deg:g} deg "
              f"orientation curve; this run holds the homed orientation")


def policy_obs(env: RealReach, obs: dict, gains: np.ndarray, n_upcoming: int) -> dict:
    """RealReach's obs -> the dict the checkpoint trained on, batched.

    Sim's world frame is the base frame shifted by its Franka mount, so every
    position moves by FRANKA_BASE_POS; orientations and velocities are the same
    in both. `controller_state` is [damping, kp] normalised: the gains in effect
    when the obs was read, i.e. the last ones sent.
    """
    waypoints = obs["waypoints"] + S.FRANKA_BASE_POS
    idx = env.next_waypoint_idx
    out = {
        **obs,
        "robot0_eef_pos": obs["robot0_eef_pos"] + S.FRANKA_BASE_POS,
        "goal": obs["goal"] + S.FRANKA_BASE_POS,
        "waypoints": waypoints,
        "upcoming_waypoints": waypoints[idx:idx + n_upcoming],
        "controller_state": gains[::-1],
        "robot0_eef_vel": env.robots[0].recent_ee_vel.current,
    }
    return {k: np.asarray(v, dtype=np.float32)[None] for k, v in out.items()}


def chunk_poses(chunk: np.ndarray, anchor_pos: np.ndarray, anchor_rot: Rotation,
                env: RealReach) -> np.ndarray:
    """(K+1, 7) [xyz, xyzw] poses a chunk of slots [damping, kp, pos(3), rot(3),
    grip] names, anchor first. Base frame, metres: the targets the viz draws."""
    pos = anchor_pos + chunk[:, 2:5] * env.osc_output_max
    rot = Rotation.from_rotvec(chunk[:, 5:8] * env.osc_rot_output_max) * anchor_rot
    return np.vstack([np.concatenate([anchor_pos, anchor_rot.as_quat()]),
                      np.hstack([pos, rot.as_quat()])])


def target_to_delta(target_pose: np.ndarray, obs: dict, env: RealReach) -> np.ndarray:
    """A chunk-start-relative target pose -> the normalised OSC delta that lands
    on it from the pose measured now (ActionChunkWrapper's second stage, position
    and rotation). The clip is scale_action's own saturation: a target further
    than one step away is approached at full step, not a new limit layer."""
    goal_rot = Rotation.from_quat(target_pose[3:7])
    ee_rot = Rotation.from_quat(obs["robot0_eef_quat"])
    return np.concatenate([
        np.clip((target_pose[:3] - obs["robot0_eef_pos"]) / env.osc_output_max, -1.0, 1.0),
        np.clip((goal_rot * ee_rot.inv()).as_rotvec() / env.osc_rot_output_max, -1.0, 1.0),
    ])


def run_episode(model: FAST, env: RealReach, recorder: ReachEpisodeRecorder,
                viz_rec: EpisodeRecorder, ep: int, seed: int, base_only: bool,
                floor: float, keep_out) -> dict:
    """One episode, chunk by chunk; returns the last step's info."""
    n_upcoming = int(model.observation_space["upcoming_waypoints"].shape[0])
    # Seeded per episode, so the base and residual variants of one episode draw
    # the same curve, and so do two runs with the same --seed.
    obs = env.reset(seed=(seed, ep))
    gains = np.zeros(2)          # normalised [a_kp, a_kd] in effect
    done, info = False, {}
    recorder.begin_reach(env, ep)
    viz_rec.record_reach_target(env.waypoints_world(), base_to_world(env.arm, env.goal))
    print(f"\nepisode {ep} ({'base' if base_only else 'residual'}): goal(base) "
          f"{np.round(env.goal, 3)}  world {np.round(base_to_world(env.arm, env.goal), 3)}")
    while not done:
        out = model.predict_diffused(policy_obs(env, obs, gains, n_upcoming),
                                     deterministic=True, sample_base=base_only)
        base = out["base_action"][0].reshape(-1, 9)
        final = out["final_action"][0].reshape(-1, 9)
        # One anchor per chunk: the pose the base planned from.
        anchor_pos = obs["robot0_eef_pos"].astype(np.float64)
        anchor_rot = Rotation.from_quat(obs["robot0_eef_quat"])
        base_poses = chunk_poses(base, anchor_pos, anchor_rot, env)
        final_poses = chunk_poses(final, anchor_pos, anchor_rot, env)
        viz_rec.record_chunk(step=len(viz_rec), ee_pos=anchor_pos,
                             base_traj=base_poses[:, :3], total_traj=final_poses[:, :3],
                             base_traj_pose=base_poses, total_traj_pose=final_poses)
        for k, (b, f) in enumerate(zip(base, final), start=1):
            seen = obs["robot0_eef_pos"]
            gains = np.array([f[1], f[0]])
            obs, _, done, info = env.step(target_to_delta(final_poses[k], obs, env),
                                          gain_action=gains)
            warn_off_limits(recorder.record_reach(seen, obs, info), floor, keep_out)
            viz_rec.record(q=info["qpos"], actual_ee_pos=obs["robot0_eef_pos"],
                           base_desired_pos=base_poses[k, :3], total_desired_pos=final_poses[k, :3],
                           kp=f[1], kd=f[0], gripper=env.gripper_norm, res_gripper=f[8] - b[8],
                           res_rotvec=(f[5:8] - b[5:8]) * env.osc_rot_output_max)
            if done:
                break
    print(f"  {'SUCCESS' if info.get('success') else 'timeout'} in {info['episode_steps']} "
          f"steps, cursor {info['next_waypoint_idx']}/{env._curve_len}")
    return info


def save_html(viz_rec: EpisodeRecorder, path: Path, arm: str, base_only: bool, title: str,
              stride: int, reference: np.ndarray | None) -> None:
    """viz.py's animated HTML, the arm placed by its own base pose. Base-only
    episodes get the rollout figure, as the LeRobot path does."""
    base = fc.robot_base_in_world(arm)
    kw = dict(title=title, frame_stride=stride, fps=fc.control_fps(),
              robot_base_in_world_translation=tuple(base.translation),
              robot_base_in_world_quat_wxyz=tuple(base.quat_wxyz),
              cam_in_world_rotation=None, cam_in_world_translation=None)
    if base_only:
        save_rollout_html(viz_rec, str(path), **kw)
    else:
        save_episode_html(viz_rec, str(path), reference_trail=reference,
                          reference_name="base-only actual EE", **kw)
    print(f"viz -> {path}")


def run(checkpoint: str, episodes: int, seed: int, arm: str, policy: str,
        device: str, out: str, viz_stride: int, no_viz: bool) -> None:
    variants = VARIANTS[policy]
    # Before the arm: a bad checkpoint should fail here, not after homing.
    model = load_residual(checkpoint, device)
    cfg = model.cfg
    print(f"residual {checkpoint}: {model.num_timesteps} steps, "
          f"goal_mode={cfg.base_policy.goal_mode}, residual_mag={model.residual_mag}, "
          f"gains_mag={model.gains_mag}; running {' then '.join(variants)} per episode")

    if not fc.robot_base_in_world_verified(arm):
        print(f"WARNING: robot_base_in_world({arm!r}) is unverified in config/world.yaml.\n"
              f"         The worktable floor derives from it. Verify before running near the table.")
    # Kinematics only: no reason to wait on six GigE connects for a task that
    # never reads a camera.
    cls = SingleArmFrankaConfig if arm == "left" else SingleArmRightConfig
    robot = make_robot_from_config(cls(control_mode=ControlMode.EE_DELTA,
                                      cameras={}, depth_cam={}, depth=False))
    robot.connect()
    try:
        env = RealReach(robot, arm=arm, arm_key="r")
        check_contract(cfg, env)
        floor = safety_z_floor_world(arm)
        keep_out = keep_out_sphere(arm)
        if keep_out is not None:
            print(f"keep-out: {np.round(keep_out[0], 3)} (world) r={keep_out[1]:.3f} m -- "
                  f"CONFIRM the other arm is where this assumes before proceeding.")

        run_dir = Path(out) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {**run_metadata(arm, seed), "residual": {
            "checkpoint": str(Path(checkpoint).resolve()),
            "num_timesteps": int(model.num_timesteps), "policy": policy,
            "cfg": {k: OmegaConf.to_container(cfg[k]) for k in ("base_policy", "policy", "reach")},
        }}
        results = []
        trace = flush(run_dir, results, meta, PRODUCER)

        recorder = ReachEpisodeRecorder(arm, seed)
        for ep in range(episodes):
            reference = None      # the base variant's trail, for the residual's figure
            for variant in variants:
                base_only = variant == "base"
                viz_rec = EpisodeRecorder()
                info = run_episode(model, env, recorder, viz_rec, ep, seed, base_only,
                                   floor, keep_out)
                name, arrays, attrs, curve = recorder.finish_reach(info)
                results.append((f"{name}_{variant}", arrays,
                                {**attrs, **force_attrs(arrays), "policy": variant}, curve))
                if "ee_force" in arrays:
                    print(f"  {variant} episode {ep}{force_note(force_stats(arrays['ee_force']))}")
                trace = flush(run_dir, results, meta, PRODUCER)
                if not no_viz:
                    save_html(viz_rec, run_dir / f"episode_{ep:03d}_{variant}.html", arm,
                              base_only, f"{variant} -- episode {ep}", viz_stride, reference)
                if base_only:
                    reference = base_to_world(arm, arrays["eef_pos"])

        for variant in variants:
            done = [r for r in results if r[2]["policy"] == variant]
            print(f"{variant}: {sum(r[2]['success'] for r in done)}/{len(done)} succeeded")
        print(f"traces -> {trace}")
    finally:
        robot.disconnect()
