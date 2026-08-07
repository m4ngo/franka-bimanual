"""Entry point for running and recording residual-policy episodes on the Franka."""

import argparse
import hashlib
import json
import logging
import os
import select
import sys
import termios
import tty
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import env_wrapper
from viz import EpisodeRecorder, save_episode_html, save_rollout_html, save_policy_pcd_npz
from viz import _propagate_pose_traj
from env_wrapper import (
    ee_pose_to_world,
    to_sim_world_points,
    to_sim_world_pose,
    to_sim_world_twist,
    _ACTION_KEYS,
    _CHUNK_EXEC,
    _RESIDUAL_HORIZON,
    _STATE_OBS_KEYS,
    _RES_POS_GAIN,
    _RES_ROT_GAIN,
    _POS_SCALE,
    _ROT_SCALE,
    build_action,
    current_ee_pose,
    measured_ee_twist_world,
    process_chunk,
    split_gripper,
    strip_depth,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from policy_wrapper import BasePolicy, ResidualPolicy, Trajectory

from reach_base_policy import ReachBasePolicy

logger = logging.getLogger(__name__)

import franka_config as fc  # noqa: E402
from env_wrapper import default_home_q as _default_home_q  # noqa: E402

_POSES_DIR = fc.home_poses_dir()


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

def _write_video_frame(writers, video_dir, video_stem, fps, cam_name, img, step_idx):
    """Lazily create one mp4 writer per camera and append an annotated frame.

    Frame index == control-loop step index (first frame = first post-homing,
    first-inference step), so base/residual runs at the same fps are
    time-aligned by construction for side-by-side stitching.
    """
    w = writers.get(cam_name)
    if w is None:
        video_dir.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(str(video_dir / f"{video_stem}_{cam_name}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps,
                            (img.shape[1], img.shape[0]))
        writers[cam_name] = w
    frame = np.ascontiguousarray(img[:, :, ::-1])  # RGB->BGR; copy keeps the obs image pristine
    label = f"{step_idx:05d} {step_idx / fps:6.2f}s"
    cv2.putText(frame, label, (4, frame.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (4, frame.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    w.write(frame)


def _infer_chunk(
    controller,
    base_policy: BasePolicy,
    residual: "ResidualPolicy | None",
    obs_no_depth: dict,
    point_cloud: np.ndarray,
    ee_pose: np.ndarray,
    kin,
    prev_kp: float,
    prev_kd: float,
    proprio_frame: str,
    sim_proprio_convention: bool,
    dump_dir: "Path | None",
    infer_idx: int,
) -> dict:
    """One base (+ residual) inference pass over a caller-supplied snapshot.

    Every input is a value the control loop already read, so this runs off the
    loop thread without touching live robot state -- see _run_episode's
    prefetch, which overlaps it with the tail of the current chunk.
    """
    base_chunk = base_policy.infer(obs_no_depth)
    res_chunk: np.ndarray = np.empty((0, 9))
    network_pcd = None

    if residual is not None:
        if kin is None:
            vel = np.zeros(6)
        else:
            # Measured twist (J @ dq), in the same frame as the proprio pose.
            r_w = (controller._r_robot_in_world if proprio_frame == "world"
                   else np.eye(3))
            vel = measured_ee_twist_world(kin['r'], r_w)
        # The cloud is world-frame; franka_fk is robot-frame. In world
        # mode, map the proprio pose into world so center_on_eef
        # subtracts a point in the same frame as the cloud.
        if proprio_frame == "world":
            proprio_pose = ee_pose_to_world(
                ee_pose,
                controller._r_robot_in_world,
                controller._t_robot_in_world,
            )
            # F5: express all world-frame quantities in sim's world
            # convention (table z + yaw; see env_wrapper). Applied
            # to pose, twist, and cloud together so the modalities
            # stay mutually consistent. --raw-proprio disables.
            if sim_proprio_convention:
                proprio_pose = to_sim_world_pose(proprio_pose)
                vel = to_sim_world_twist(vel)
                point_cloud = to_sim_world_points(point_cloud)
        else:
            proprio_pose = ee_pose
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
        network_pcd = residual.last_network_pcd
        if dump_dir is not None:
            np.savez_compressed(
                dump_dir / f"obs_{infer_idx:05d}.npz",
                base_chunk_raw=base_chunk.astype(np.float32),
                action_chunk=residual_obs["action_chunk"],
                proprio=residual_obs["proprio"],
                point_cloud=residual_obs["point_cloud"],
                network_pcd=network_pcd,
                res_chunk=res_chunk.astype(np.float32),
            )

    return {
        "base_chunk": base_chunk,
        "res_chunk": res_chunk,
        "ee_pose": ee_pose,
        "network_pcd": network_pcd,
    }


def _warmup_policies(
    controller,
    base_policy: BasePolicy,
    residual: "ResidualPolicy | None",
    proprio_frame: str,
    sim_proprio_convention: bool,
) -> None:
    """Run the real inference path a few times before the episode starts.

    torch.compile's first call, cuDNN algorithm selection and the CUDA context
    together cost seconds; paying that on the first control step would stall the
    arm mid-trajectory. Warms on the loop thread and once on a worker, since the
    prefetch path calls the same compiled module from a different thread.
    """
    t0 = time.perf_counter()
    obs = controller.get_observation()
    args_ = (controller, base_policy, residual, strip_depth(obs),
             controller.last_full_point_cloud,
             current_ee_pose(obs, sim_convention=sim_proprio_convention),
             controller.kin, 0.0, 0.0, proprio_frame, sim_proprio_convention, None, 0)
    for _ in range(3):
        _infer_chunk(*args_)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_infer_chunk, *args_).result()
    base_policy.reset()
    # The warmup consumed the cached snapshot; the next send_action must not
    # anchor its delta goal on a pose this old.
    controller._cached_kin_state = None
    print(f"policy warmup done in {time.perf_counter() - t0:.1f}s")


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
    dump_dir: "Path | None" = None,
    video_dir: "Path | None" = None,
    video_cams: "list[str] | None" = None,
    video_stem: str = "episode",
    infer_lead: int = 1,
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
        infer_lead: how many steps ahead of a chunk's first execution the
            inference for it is started, on a worker thread. 1 disables the
            overlap (inference blocks the loop, the old behaviour).
    """
    base_policy.reset()

    base_chunk: np.ndarray = np.empty((0, 10))
    res_chunk: np.ndarray = np.empty((0, 9))
    chunk_used = _CHUNK_EXEC   # triggers immediate inference on first step
    prev_kp = 0.0
    prev_kd = 0.0
    infer_idx = 0
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    video_writers: dict[str, "cv2.VideoWriter"] = {}
    step_idx = 0

    dt = 1.0 / fps
    t_start = time.perf_counter()

    # Inference runs on a worker so it overlaps the tail of the chunk already
    # executing instead of stalling the loop between the state read and the goal
    # write. One worker only: the policies are stateful and must stay serialised.
    infer_lead = int(np.clip(infer_lead, 1, _CHUNK_EXEC))
    # Index within the chunk whose step submits the NEXT chunk's inference, so
    # the obs it uses sits `infer_lead` steps before that chunk's first action.
    submit_at = _CHUNK_EXEC - infer_lead
    infer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    pending: "Future | None" = None

    old_term = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    fps_frames = 0
    fps_window_start = time.perf_counter()
    busy_ms_window: list[float] = []        # per-step busy time (pre-sleep), current window
    chunk_busy_ms_window: list[float] = []  # subset: steps that ran inference
    wait_ms_window: list[float] = []        # blocked at the chunk swap (prefetch too late)
    send_gap_ms_window: list[float] = []    # interval between consecutive send_action calls
    t_prev_send = 0.0
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
            # Grab it here: send_action() consumes and clears the cache.
            kin_snapshot = controller.kin
            # Array channel, not obs scalars; a fresh array each get_observation,
            # so the prefetch worker's reference stays valid across steps.
            cloud_snapshot = controller.last_full_point_cloud

            if video_dir is not None:
                for cam_name in (video_cams or sorted(controller.cameras.keys())):
                    img = obs.get(cam_name)
                    if isinstance(img, np.ndarray) and img.ndim == 3:
                        _write_video_frame(video_writers, video_dir, video_stem,
                                           fps, cam_name, img, step_idx)
            step_idx += 1

            if chunk_used >= _CHUNK_EXEC:
                if pending is None:
                    # Cold start (and infer_lead=1): nothing was prefetched, so
                    # this one pass does block the loop.
                    result = _infer_chunk(
                        controller, base_policy, residual, obs_no_depth,
                        cloud_snapshot, ee_pose, kin_snapshot, prev_kp, prev_kd,
                        proprio_frame, sim_proprio_convention, dump_dir, infer_idx,
                    )
                else:
                    t_wait = time.perf_counter()
                    result = pending.result()
                    pending = None
                    wait_ms_window.append((time.perf_counter() - t_wait) * 1000.0)
                infer_idx += 1
                base_chunk = result["base_chunk"]
                res_chunk = result["res_chunk"]
                # The forecast is anchored on the pose the policy actually saw,
                # which with a prefetch is infer_lead steps behind this one.
                chunk_ee_pose = result["ee_pose"]
                chunk_used = 0

                if recorder is not None and result["network_pcd"] is not None:
                    recorder.record_policy_pcd(len(recorder), result["network_pcd"])

                if recorder is not None:
                    ee3 = chunk_ee_pose[:3].astype(np.float32)
                    ee_pose_xyzw = chunk_ee_pose[:7].astype(np.float32)
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
                dpos = (np.clip(b_pos + res[2:5], -1.0, 1.0) - b_pos) * _POS_SCALE * _RES_POS_GAIN
                drot = (np.clip(b_rot + res[5:8], -1.0, 1.0) - b_rot) * _ROT_SCALE * _RES_ROT_GAIN
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
            t_send = time.perf_counter()
            controller.send_action(action)
            if t_prev_send:
                send_gap_ms_window.append((t_send - t_prev_send) * 1000.0)
            t_prev_send = t_send

            # Kick off the next chunk's inference only after this step's goal is
            # on the wire -- the whole point is to keep it out of the
            # state-read -> goal-write path.
            if chunk_used == submit_at and pending is None and infer_lead > 1:
                # prev_kp/prev_kd, not this step's: the residual's proprio
                # carries the controller state in effect when obs was READ, and
                # this step's gains only take effect from the send_action above.
                pending = infer_pool.submit(
                    _infer_chunk, controller, base_policy, residual, obs_no_depth,
                    cloud_snapshot, ee_pose, kin_snapshot, prev_kp, prev_kd,
                    proprio_frame, sim_proprio_convention, dump_dir, infer_idx,
                )

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
                # send-gap max is the number that matters for smoothness: it is
                # how long the OSC loop sat on one goal, and a spike there is
                # exactly the visible hitch at the chunk boundary.
                logger.info(
                    "loop fps: %.2f target: %.2f busy avg/max: %.1f/%.1f ms "
                    "(chunk-step avg: %.1f ms) send-gap avg/max: %.1f/%.1f ms "
                    "prefetch-wait avg/max: %.1f/%.1f ms",
                    actual_fps, fps,
                    sum(busy_ms_window) / len(busy_ms_window), max(busy_ms_window), chunk_avg,
                    (sum(send_gap_ms_window) / len(send_gap_ms_window)) if send_gap_ms_window else 0.0,
                    max(send_gap_ms_window) if send_gap_ms_window else 0.0,
                    (sum(wait_ms_window) / len(wait_ms_window)) if wait_ms_window else 0.0,
                    max(wait_ms_window) if wait_ms_window else 0.0,
                )
                fps_window_start = now
                fps_frames = 0
                busy_ms_window.clear()
                chunk_busy_ms_window.clear()
                wait_ms_window.clear()
                send_gap_ms_window.clear()

            sleep_s = t_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_deadline += dt
            # After a large stall (episode-start model warmup, operator pause),
            # resync instead of racing to repay an unpayable debt.
            if t_deadline < time.perf_counter():
                t_deadline = time.perf_counter() + dt
    finally:
        # Drain before shutdown: the in-flight pass holds the policies and may
        # still be writing a dump npz.
        if pending is not None:
            try:
                pending.result(timeout=5.0)
            except Exception:
                logger.exception("in-flight inference failed during teardown")
        infer_pool.shutdown(wait=True)
        stale = getattr(controller, "_kin_cache_stale", 0)
        if stale:
            logger.info("send_action re-read the kin snapshot %d time(s) (cache too old)", stale)
        for w in video_writers.values():
            w.release()
        if video_writers:
            logger.info("saved %d camera video(s) to %s", len(video_writers), video_dir)
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


def _record_reach_target(recorder: "EpisodeRecorder | None", base_policy) -> None:
    """Hand the episode's reach curve to the recorder so the viz can draw it.

    Called after the episode: the curve is anchored lazily on the first infer(),
    so it does not exist yet when the recorder is created.
    """
    if recorder is None or not isinstance(base_policy, ReachBasePolicy):
        return
    recorder.record_reach_target(base_policy.waypoints_world, base_policy.goal_pos_world)


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
    parser.add_argument("--save-videos", action="store_true",
                        help="Write one time-aligned mp4 per camera into --viz-dir "
                             "(frame index == control step; see stitch_videos.py)")
    parser.add_argument("--video-cams", nargs="+", default=None,
                        help="Camera obs keys to record (default: all connected cameras)")
    parser.add_argument("--dump-obs-dir", default=None,
                        help="Dump every residual_obs bundle (+ residual output) to npz under "
                             "this dir, one timestamped run subdir per invocation; feeds the "
                             "Tier 1/2 checks in STUDENT_INPUT_PARITY.md")
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
    parser.add_argument("--infer-lead", type=int, default=1,
                        help="Steps ahead of a chunk's first action to start its inference "
                             "on a worker thread, so it overlaps the tail of the chunk "
                             f"already executing (max {_CHUNK_EXEC}). Buys a uniform send "
                             "cadence at the cost of the policy seeing an observation that "
                             "many steps old. 1 = no prefetch: the chunk is inferred on the "
                             "step that executes it, which fits the 20 Hz budget once the "
                             "base policy is sped up (see --base-amp / --base-compile)")
    parser.add_argument("--base-amp", choices=("fp16", "bf16", "none"), default="fp16",
                        help="Autocast dtype for the base policy. Its cost is GPU compute, "
                             "not launch overhead, so precision is the lever that shrinks it: "
                             "fp16 takes the diffusion policy 33.0 -> 25.6 ms, and the "
                             "commanded delta moves by at most 0.04 mm / 0.008 deg against a "
                             "+/-50 mm, +/-28.6 deg envelope. 'none' restores fp32")
    parser.add_argument("--base-compile", choices=("default", "max-autotune", "none"),
                        default="default",
                        help="torch.compile the denoising UNet. With fp16: 'default' 23.4 ms "
                             "(~6 s startup), 'max-autotune' 21.8 ms (~20 s startup). Cost is "
                             "paid by the pre-episode warmup, not the first control step")
    parser.add_argument(
        "--home-pose-name",
        default=fc.default_home_pose_name(),
        help=f"Name of a saved pose JSON in {_POSES_DIR} (overrides --home-q)",
    )
    parser.add_argument(
        "--home-q", nargs=7, type=float, default=None,
        help="7 joint angles (rad) overriding the saved home pose",
    )
    parser.add_argument("--home-gripper", type=float, default=fc.control("homing.gripper_norm"))
    parser.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    parser.add_argument("--home-tol-rad", type=float, default=fc.control("homing.tol_rad"))

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
    parser.add_argument("--fps", type=int, default=fc.control_fps(),
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
    parser.add_argument("--reach-goal", nargs=3, type=float, default=None,
                        help="Goal EE position [x y z] (world/robot frame per "
                             "--proprio-frame) for the analytic Reach base policy; "
                             "enables the reach task in place of --base-policy")
    parser.add_argument("--reach-n-waypoints", type=int, default=128)
    parser.add_argument("--reach-control-offset", type=float, default=0.15)
    parser.add_argument("--reach-advance-threshold", type=float, default=0.02)
    parser.add_argument("--reach-lookahead-k", type=int, default=5)

    args = parser.parse_args()

    if args.repo_id and not args.output_dir:
        parser.error("--output-dir is required when --repo-id is set")
    if args.save_videos and not args.viz_dir:
        parser.error("--viz-dir is required when --save-videos is set")
    if args.repo_id and not args.task:
        parser.error("--task is required when --repo-id is set")

    logging.basicConfig(level=logging.INFO, force=True)

    if args.home_q is not None:
        home_q = np.asarray(args.home_q, dtype=np.float64)
        home_gripper = args.home_gripper
    else:
        pose = fc.load_home_pose(args.home_pose_name)
        home_q = _default_home_q(args.home_pose_name)
        home_gripper = float(pose.get("gripper", args.home_gripper))

    print("attempting connection to robot...")
    controller = env_wrapper.start_controller()
    print("robot initialized!")

    if args.reach_goal is not None:
        print(f"attempting to start analytic reach base policy toward "
              f"{args.reach_goal} (world frame)")
        base_policy = ReachBasePolicy(
            goal_pos=np.asarray(args.reach_goal, dtype=np.float64),
            r_robot_in_world=controller._r_robot_in_world,
            t_robot_in_world=controller._t_robot_in_world,
            n_waypoints=args.reach_n_waypoints,
            control_offset=args.reach_control_offset,
            advance_threshold=args.reach_advance_threshold,
            lookahead_k=args.reach_lookahead_k,
            sim_proprio_convention=not args.raw_proprio,
        )
        print("reach base policy started!")
    elif args.replay_dataset is None:
        print(f"attempting to start base policy: {args.base_policy}")
        base_policy = BasePolicy(args.base_policy, device=args.device,
                                 amp=args.base_amp, compile_mode=args.base_compile)
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

    _warmup_policies(controller, base_policy, residual,
                     args.proprio_frame, not args.raw_proprio)

    if isinstance(base_policy, ReachBasePolicy):
        base_policy.reset_curve()   # discard the warmup-anchored curve

    dump_root: "Path | None" = None
    if args.dump_obs_dir:
        if residual is None:
            print("--dump-obs-dir ignored: residual_obs only exists with a residual policy")
        else:
            dump_root = Path(args.dump_obs_dir).expanduser() / time.strftime("%Y%m%d_%H%M%S")
            dump_root.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with open(args.residual_policy, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            (dump_root / "meta.json").write_text(json.dumps({
                "residual_policy": str(Path(args.residual_policy).resolve()),
                "residual_policy_sha256": h.hexdigest(),
                "base_policy": args.base_policy or args.replay_dataset,
                "proprio_frame": args.proprio_frame,
                "raw_proprio": bool(args.raw_proprio),
                "fps": args.fps,
                "argv": sys.argv,
            }, indent=2))
            print(f"dumping residual obs bundles to {dump_root}")

    home_kwargs = dict(
        home_q_left=None,
        home_q_right=home_q,
        gripper_norm=home_gripper,
        max_time_s=args.home_max_time_s,
        tol_rad=args.home_tol_rad,
    )

    recording = args.repo_id is not None
    dataset: LeRobotDataset | None = None

    if not recording:
            print("homing...")
            if isinstance(base_policy, ReachBasePolicy):
                base_policy.reset_curve()   # anchor to the true post-homing pose
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
                    dump_dir=dump_root / "ep000" if dump_root else None,
                    video_dir=Path(args.viz_dir) if args.save_videos else None,
                    video_cams=args.video_cams,
                    infer_lead=args.infer_lead,
                )
            finally:
                _record_reach_target(recorder, base_policy)
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
            if isinstance(base_policy, ReachBasePolicy):
                base_policy.reset_curve()   # anchor to the true post-homing pose
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
                        dump_dir=dump_root / f"ep{dataset.num_episodes:03d}" if dump_root else None,
                        video_dir=Path(args.viz_dir) if args.save_videos else None,
                        video_cams=args.video_cams,
                        video_stem=f"episode_{ep_idx:03d}",
                        infer_lead=args.infer_lead,
                    )
                finally:
                    _record_reach_target(recorder, base_policy)
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
                    if isinstance(base_policy, ReachBasePolicy):
                        base_policy.reset_curve() 
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
