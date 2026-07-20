"""Entry point for running and recording residual-policy episodes on the Franka."""

import argparse
import json
import logging
import os
import select
import sys
import termios
import tty
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import env_wrapper
from viz import EpisodeRecorder, save_episode_html, save_rollout_html, save_policy_pcd_npz
from viz import _propagate_pose_traj
from env_wrapper import (
    ee_pose_to_world,
    _ACTION_KEYS,
    _CHUNK_EXEC,
    _RESIDUAL_HORIZON,
    _STATE_OBS_KEYS,
    _POS_SCALE,
    _ROT_SCALE,
    build_action,
    current_ee_pose,
    extract_point_cloud,
    measured_ee_twist_world,
    process_chunk,
    split_gripper,
    strip_depth,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from policy_wrapper import BasePolicy, ResidualPolicy, Trajectory

logger = logging.getLogger(__name__)

_POSES_DIR = Path(__file__).resolve().parent.parent / "home_poses"
from env_wrapper import DEFAULT_HOME_Q as _DEFAULT_HOME_Q  # noqa: E402


def _stdin_key_pressed() -> bool:
    """Return True if a key has been pressed on stdin (non-blocking)."""
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _read_key() -> str:
    """Read one keypress from stdin (caller must be in raw mode).

    Returns 'right' for right-arrow, 'ctrl_c' for Ctrl-C, or '' for anything else.

    Uses os.read exclusively (never sys.stdin.read) so Python's text-mode buffer
    cannot swallow the CSI tail bytes before we inspect them.
    """
    # Sleep briefly so the full escape sequence has time to arrive, then read
    # all pending bytes in one syscall.
    time.sleep(0.03)
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):
        return "right"
    return ""


def _wait_for_right_arrow() -> None:
    """Block until the right-arrow key is pressed. Raises KeyboardInterrupt on Ctrl-C."""
    old_term = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = _read_key()
                if key == "right":
                    return
                if key == "ctrl_c":
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_dataset(args, controller) -> LeRobotDataset:
    """Create or resume a LeRobotDataset for rollout recording."""
    cam_features = {
        f"observation.images.{cam_name}": {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }
        for cam_name, cam in controller.cameras.items()
    }
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(_STATE_OBS_KEYS),),
            "names": [list(_STATE_OBS_KEYS)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(_ACTION_KEYS),),
            "names": [list(_ACTION_KEYS)],
        },
        **cam_features,
    }
    n_cams = len(controller.cameras)
    common = dict(
        batch_encoding_size=1,
        vcodec="auto",
        streaming_encoding=True,
        encoder_queue_maxsize=8,
        encoder_threads=2,
    )
    if args.resume:
        return LeRobotDataset.resume(
            args.repo_id,
            root=args.output_dir,
            image_writer_processes=0,
            image_writer_threads=4 * n_cams,
            **common,
        )
    return LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=args.output_dir,
        robot_type=controller.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * n_cams,
        **common,
    )


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def _run_episode(
    controller,
    base_policy: BasePolicy,
    residual: "ResidualPolicy | None",
    dataset: "LeRobotDataset | None",
    episode_time_s: "float | None",
    fps: float = 20.0,
    task: str = "",
    recorder: "EpisodeRecorder | None" = None,
    replaying: bool = False,
    proprio_frame: str = "world",
    sim_proprio_convention: bool = True,
) -> None:
    """Run one episode of the policy loop.

    Args:
        controller: connected SingleArmFranka instance.
        base_policy: loaded BasePolicy.
        residual: loaded ResidualPolicy, or None when --no-residual is set.
        dataset: open LeRobotDataset to record into, or None to skip recording.
        episode_time_s: stop after this many seconds when recording; None runs forever.
        fps: target control frequency; each step sleeps for the remainder of 1/fps.
        task: task description string included in every recorded frame.
        recorder: optional EpisodeRecorder; when provided, per-step state is
            appended so save_episode_html can be called after the episode.
    """
    base_policy.reset()

    base_chunk: np.ndarray = np.empty((0, 10))
    res_chunk: np.ndarray = np.empty((0, 9))
    chunk_used = _CHUNK_EXEC   # triggers immediate inference on first step
    prev_kp = 0.0
    prev_kd = 0.0

    dt = 1.0 / fps
    t_start = time.perf_counter()

    old_term = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    fps_frames = 0
    fps_window_start = time.perf_counter()
    busy_ms_window: list[float] = []        # per-step busy time (pre-sleep), current window
    chunk_busy_ms_window: list[float] = []  # subset: steps that ran inference
    # Fixed-cadence deadline: sleeping to an absolute clock lets the idle slack
    # of ordinary steps absorb the few ms that inference steps overrun, so the
    # loop averages the target rate instead of accumulating per-step deficits.
    t_deadline = time.perf_counter() + dt
    try:
        while True:
            t_step = time.perf_counter()
            if episode_time_s is not None and t_step - t_start >= episode_time_s:
                break
            if _stdin_key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "right":
                    print("\r\nearly stop requested", flush=True)
                    break
            
            # times = []
            # times.append(time.perf_counter())
            obs = controller.get_observation()
            # times.append(time.perf_counter())
            ee_pose = current_ee_pose(obs, sim_convention=sim_proprio_convention)
            # times.append(time.perf_counter())
            obs_no_depth = strip_depth(obs)
            # times.append(time.perf_counter())

            if chunk_used >= _CHUNK_EXEC:
                base_chunk = base_policy.infer(obs_no_depth)
                # times.append(time.perf_counter())
                chunk_used = 0

                if residual is not None:
                    kin = controller.kin
                    if kin is None:
                        vel = np.zeros(6)
                    else:
                        # Measured twist (J @ dq), in the same frame as the proprio pose.
                        r_w = (controller._r_robot_in_world if proprio_frame == "world"
                               else np.eye(3))
                        vel = measured_ee_twist_world(kin['r'], r_w)
                    point_cloud = extract_point_cloud(obs)
                    # The cloud is world-frame; franka_fk is robot-frame. In world
                    # mode, map the proprio pose into world so center_on_eef
                    # subtracts a point in the same frame as the cloud.
                    if proprio_frame == "world":
                        proprio_pose = ee_pose_to_world(
                            ee_pose,
                            controller._r_robot_in_world,
                            controller._t_robot_in_world,
                        )
                    else:
                        proprio_pose = ee_pose
                    # base_chunk = np.repeat(base_chunk, 2, axis=0)
                    processed_chunk = process_chunk(base_chunk)
                    residual_obs = {
                        "action_chunk": processed_chunk[:_RESIDUAL_HORIZON],
                        "proprio": np.concatenate([
                            split_gripper(proprio_pose).astype(np.float32),
                            # Sim controller_state convention: [damping_norm, kp_norm].
                            np.array([prev_kd, prev_kp], dtype=np.float32),
                            np.asarray(vel, dtype=np.float32),
                        ]),
                        "point_cloud": point_cloud,
                    }
                    res_chunk = residual.infer(residual_obs)
                    if recorder is not None and residual.last_network_pcd is not None:
                        recorder.record_policy_pcd(len(recorder), residual.last_network_pcd)

                if recorder is not None:
                    ee3 = ee_pose[:3].astype(np.float32)
                    ee_pose_xyzw = ee_pose[:7].astype(np.float32)
                    # Raw unnormalised position deltas (metres) from postprocessor.
                    # Cumsum from current EE gives the commanded delta trajectory.
                    base_deltas = base_chunk[:_RESIDUAL_HORIZON, :3].astype(np.float32)
                    base_rotvecs = np.array([
                        Rotation.from_quat(step[3:7]).as_rotvec() for step in base_chunk[:_RESIDUAL_HORIZON]
                    ], dtype=np.float32)
                    base_traj = np.vstack([ee3, ee3 + np.cumsum(base_deltas, axis=0)])
                    base_traj_pose = _propagate_pose_traj(ee_pose_xyzw, base_deltas, base_rotvecs)
                    if residual is not None and len(res_chunk) > 0:
                        K_res = min(len(res_chunk), _RESIDUAL_HORIZON)
                        # Same headroom clip as the execution path so plots show
                        # what actually runs.
                        total_deltas = base_deltas.copy()
                        total_deltas[:K_res] = np.clip(
                            base_deltas[:K_res] / _POS_SCALE + res_chunk[:K_res, 2:5], -1.0, 1.0
                        ).astype(np.float32) * _POS_SCALE
                        total_rotvecs = base_rotvecs.copy()
                        total_rotvecs[:K_res] = np.clip(
                            base_rotvecs[:K_res] / _ROT_SCALE + res_chunk[:K_res, 5:8], -1.0, 1.0
                        ).astype(np.float32) * _ROT_SCALE
                        total_traj = np.vstack([ee3, ee3 + np.cumsum(total_deltas, axis=0)])
                        total_traj_pose = _propagate_pose_traj(ee_pose_xyzw, total_deltas, total_rotvecs)
                    else:
                        total_traj = base_traj.copy()
                        total_traj_pose = base_traj_pose.copy()
                    recorder.record_chunk(
                        step=len(recorder),
                        ee_pos=ee3,
                        base_traj=base_traj,
                        total_traj=total_traj,
                        base_traj_pose=base_traj_pose,
                        total_traj_pose=total_traj_pose,
                    )

            if residual is not None:
                res = res_chunk[chunk_used]
                # Sim executes clip(base + residual, -1, 1) per normalized channel;
                # clip the residual to the base's remaining headroom to match.
                b_pos = base_chunk[chunk_used, :3].astype(np.float64) / _POS_SCALE
                b_rot = Rotation.from_quat(base_chunk[chunk_used, 3:7]).as_rotvec() / _ROT_SCALE
                dpos = (np.clip(b_pos + res[2:5], -1.0, 1.0) - b_pos) * _POS_SCALE
                drot = (np.clip(b_rot + res[5:8], -1.0, 1.0) - b_rot) * _ROT_SCALE
                if replaying:
                    # residual is visualized only; base/trajectory gains drive execution
                    kp = float(base_chunk[chunk_used, 8])
                    kd = float(base_chunk[chunk_used, 9])
                else:
                    # Residual layout is [damping, stiffness, ...] (multi-fast convention).
                    kp = float(res[1])
                    kd = float(res[0])
            else:
                dpos = np.zeros(3, dtype=np.float32)
                drot = np.zeros(3, dtype=np.float32)
                kp = float(base_chunk[chunk_used, 8])
                kd = float(base_chunk[chunk_used, 9])

            if not replaying:
                controller.cache_delta(dpos, drot)
            # print('base grip', base_chunk[chunk_used][8])
            # print('res grip', res[8])
            action = build_action(base_chunk[chunk_used], kp=kp, kd=kd)
            # print(action)
            if residual is not None:
                action["r_gripper"] = float(np.clip(action["r_gripper"] + res[8], -1.0, 1.0))
            controller.send_action(action)

            if recorder is not None:
                q = np.array([obs[f"r_joint_{i}"] for i in range(1, 8)])
                # In delta mode, base_chunk[:3] is a position delta, not an absolute
                # target.  Add it to the current EE position so both trail fields stay
                # in world-space metres for meaningful 3D visualization.
                base_desired_pos = ee_pose[:3] + base_chunk[chunk_used, :3]
                recorder.record(
                    q=q,
                    actual_ee_pos=ee_pose[:3],
                    base_desired_pos=base_desired_pos,
                    total_desired_pos=base_desired_pos + dpos,
                    kp=kp,
                    kd=kd,
                    gripper=action["r_gripper"],
                    res_gripper=res[8] if residual is not None else 0,
                    point_cloud=controller.last_full_point_cloud,
                    # point_cloud=point_cloud,
                    res_rotvec=drot,
                )

            if dataset is not None:
                frame: dict = {
                    "observation.state": np.array([obs[k] for k in _STATE_OBS_KEYS], dtype=np.float32),
                    "action": np.array([action[k] for k in _ACTION_KEYS], dtype=np.float32),
                    "task": task,
                }
                for cam_name in controller.cameras:
                    img = obs_no_depth.get(cam_name)
                    if isinstance(img, np.ndarray) and img.ndim == 3:
                        frame[f"observation.images.{cam_name}"] = img
                dataset.add_frame(frame)

            prev_kp = kp
            prev_kd = kd
            chunk_used += 1


            elapsed = time.perf_counter() - t_step
            busy_ms_window.append(elapsed * 1000.0)
            if chunk_used == 1:  # this iteration ran inference
                chunk_busy_ms_window.append(elapsed * 1000.0)

            fps_frames += 1
            now = time.perf_counter()
            window_s = now - fps_window_start
            if window_s >= 1.0:
                actual_fps = fps_frames / window_s
                chunk_avg = (sum(chunk_busy_ms_window) / len(chunk_busy_ms_window)
                             if chunk_busy_ms_window else 0.0)
                logger.info(
                    "loop fps: %.2f target: %.2f busy avg/max: %.1f/%.1f ms (chunk-step avg: %.1f ms)",
                    actual_fps, fps,
                    sum(busy_ms_window) / len(busy_ms_window), max(busy_ms_window), chunk_avg,
                )
                fps_window_start = now
                fps_frames = 0
                busy_ms_window.clear()
                chunk_busy_ms_window.clear()

            sleep_s = t_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_deadline += dt
            # After a large stall (episode-start model warmup, operator pause),
            # resync instead of racing to repay an unpayable debt.
            if t_deadline < time.perf_counter():
                t_deadline = time.perf_counter() + dt
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_viz(
    recorder: EpisodeRecorder,
    path: str,
    residual: "ResidualPolicy | None",
    title: str,
    frame_stride: int,
    fps: float,
) -> None:
    """Dispatch to save_rollout_html (base only) or save_episode_html (residual)."""
    if residual is None:
        save_rollout_html(recorder, path, title=f"base policy — {title}",
                          frame_stride=frame_stride, fps=fps)
    else:
        save_episode_html(recorder, path, title=f"residual — {title}",
                          frame_stride=frame_stride, fps=fps)
    if recorder.policy_pcd_events:
        pcd_path = (path[:-len(".html")] if path.endswith(".html") else path) + "_policy_pcd.npz"
        centered = residual is not None and residual.center_on_eef
        save_policy_pcd_npz(recorder.policy_pcd_events, pcd_path, center_on_eef=centered, fps=fps)
        print(f"saved policy-input clouds to {pcd_path} (plot with plot_policy_pcd.py)")


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", required=False, help="Path to base policy checkpoint")
    parser.add_argument(
        "--residual-policy",
        default=str(Path(__file__).resolve().parent.parent / "best.pt"),
        help="Path to residual policy checkpoint (best.pt)",
    )
    parser.add_argument("--no-residual", action="store_true",
                        help="Disable the residual policy; run base policy only")
    parser.add_argument("--proprio-frame", choices=("robot", "world"), default="world",
                        help="Frame for the residual proprio pose: 'robot' = raw franka_fk "
                             "(current behavior), 'world' = transformed to the world frame "
                             "the point cloud lives in")
    parser.add_argument("--raw-proprio", action="store_true",
                        help="A/B control: skip the sim-convention proprio correction "
                             "(45\u00b0 flange-vs-body quat + 6.9 mm TCP-vs-site pos; see "
                             "env_wrapper.current_ee_pose) and feed the legacy raw "
                             "franka_fk pose to the residual policy")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda/cpu)")
    parser.add_argument(
        "--home-pose-name",
        default="home_pose",
        help=f"Name of a saved pose JSON in {_POSES_DIR} (overrides --home-q)",
    )
    parser.add_argument(
        "--home-q", nargs=7, type=float, default=_DEFAULT_HOME_Q,
        help="7 joint angles (rad) for the right arm home pose",
    )
    parser.add_argument("--home-gripper", type=float, default=1.0)
    parser.add_argument("--home-max-time-s", type=float, default=3.0)
    parser.add_argument("--home-tol-rad", type=float, default=0.05)
    parser.add_argument("--home-tol-m", type=float, default=0.025)

    # Recording options (all optional; omitting --repo-id disables recording).
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace repo id for the recorded dataset; enables recording")
    parser.add_argument("--output-dir", default=None,
                        help="Local root for the dataset (required when --repo-id is set)")
    parser.add_argument("--task", default=None,
                        help="Single-task description stored with each episode")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to record (only used when recording)")
    parser.add_argument("--episode-time-s", type=float, default=60.0,
                        help="Duration of each episode in seconds (only used when recording)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Dataset fps (only used when creating a new dataset)")
    parser.add_argument("--push-to-hub", type=_str2bool, default=True,
                        help="Push dataset to HuggingFace Hub after recording")
    parser.add_argument("--resume", type=_str2bool, default=False,
                        help="Resume an existing dataset instead of creating a new one")
    parser.add_argument("--viz-dir", default=None,
                        help="Directory to write per-episode Plotly HTML visualizations")
    parser.add_argument("--viz-stride", type=int, default=1,
                        help="Animate every Nth step in the visualization (default 1)")
    parser.add_argument("--replay-dataset", default=None, help="HuggingFace id for the dataset to replay from")

    args = parser.parse_args()

    if args.repo_id and not args.output_dir:
        parser.error("--output-dir is required when --repo-id is set")
    if args.repo_id and not args.task:
        parser.error("--task is required when --repo-id is set")

    logging.basicConfig(level=logging.INFO, force=True)

    if args.home_pose_name:
        pose = json.loads((_POSES_DIR / f"{args.home_pose_name}.json").read_text())
        home_q = np.asarray(pose["r_q"], dtype=np.float64)
        home_gripper = float(pose.get("gripper", args.home_gripper))
    else:
        home_q = np.asarray(args.home_q, dtype=np.float64)
        home_gripper = args.home_gripper

    print("attempting connection to robot...")
    controller = env_wrapper.start_controller()
    print("robot initialized!")

    if args.replay_dataset is None:
        print(f"attempting to start base policy: {args.base_policy}")
        base_policy = BasePolicy(args.base_policy, device=args.device)
        print("base policy started!")
    else:
        print(f"attempting to fetch replay dataset: {args.replay_dataset}")
        base_policy = Trajectory(args.replay_dataset, device=args.device)
        print("replay dataset found!")

    residual: ResidualPolicy | None = None
    if args.no_residual:
        print("residual policy disabled (--no-residual)")
    else:
        print(f"attempting to start residual policy: {args.residual_policy}")
        residual = ResidualPolicy(args.residual_policy, device=args.device)
        print("residual policy started")

    home_kwargs = dict(
        home_q_left=None,
        home_q_right=home_q,
        gripper_norm=home_gripper,
        max_time_s=args.home_max_time_s,
        tol_rad=args.home_tol_rad,
        tol_pos_m=args.home_tol_m,
    )

    recording = args.repo_id is not None
    dataset: LeRobotDataset | None = None

    if not recording:
            print("homing...")
            if not controller.home(**home_kwargs):
                logger.warning("homing did not converge; proceeding anyway")
            recorder = EpisodeRecorder() if args.viz_dir else None
            try:
                _run_episode(
                    controller, base_policy, residual,
                    dataset=None, episode_time_s=None,
                    fps=args.fps, recorder=recorder,
                    replaying=args.replay_dataset is not None,
                    proprio_frame=args.proprio_frame,
                    sim_proprio_convention=not args.raw_proprio,
                )
            finally:
                if recorder is not None and len(recorder) > 0:
                    viz_path = os.path.join(args.viz_dir, "episode.html")
                    print(f"saving visualization to {viz_path}...")
                    _save_viz(recorder, viz_path, residual, "episode (free run)", args.viz_stride, args.fps)
                controller.disconnect()
            return

    # Multi-episode recording mode.
    dataset = _build_dataset(args, controller)
    try:
        with VideoEncodingManager(dataset):
            # Home once before the first episode.
            print(f"homing before episode {dataset.num_episodes}...")
            if not controller.home(**home_kwargs):
                logger.warning("homing did not converge; proceeding anyway")

            for ep_idx in range(args.num_episodes):
                print(f"press right arrow to start episode {dataset.num_episodes} / {args.num_episodes} "
                      f"({args.episode_time_s:.0f}s)...")
                _wait_for_right_arrow()

                print(f"recording episode {dataset.num_episodes} / {args.num_episodes} "
                      f"({args.episode_time_s:.0f}s)...")
                recorder = EpisodeRecorder() if args.viz_dir else None
                try:
                    _run_episode(
                        controller, base_policy, residual,
                        dataset=dataset,
                        episode_time_s=args.episode_time_s,
                        fps=args.fps,
                        task=args.task,
                        recorder=recorder,
                        replaying=args.replay_dataset is not None,
                        proprio_frame=args.proprio_frame,
                        sim_proprio_convention=not args.raw_proprio,
                    )
                finally:
                    if recorder is not None and len(recorder) > 0:
                        viz_path = os.path.join(
                            args.viz_dir, f"episode_{ep_idx:03d}.html"
                        )
                        print(f"saving visualization to {viz_path}...")
                        _save_viz(recorder, viz_path, residual, f"episode {ep_idx} — {args.task}", args.viz_stride, args.fps)
                dataset.save_episode()
                print(f"episode {dataset.num_episodes - 1} saved")

                if ep_idx < args.num_episodes - 1:
                    print("resetting environment — homing arm before next episode...")
                    if not controller.home(**home_kwargs):
                        logger.warning("homing did not converge; proceeding anyway")
    finally:
        if dataset is not None:
            dataset.finalize()
            if args.push_to_hub:
                try:
                    dataset.push_to_hub()
                except Exception:
                    logger.exception(
                        "push_to_hub failed; dataset is on disk at %s",
                        Path(args.output_dir).resolve(),
                    )
        controller.disconnect()


if __name__ == "__main__":
    main()
