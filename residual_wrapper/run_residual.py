"""Entry point for running and recording residual-policy episodes on the Franka.

Two residual families, told apart by the checkpoint --residual-policy names:

  best.pt (torch)        the point-cloud residual on a LeRobot base policy: the
                         loop in this file, recorded as a LeRobotDataset under a
                         run directory. The base emits ABSOLUTE poses (EE_POS; a
                         delta-trained checkpoint is refused) and is executed the
                         way the reach path below executes its analytic base:
                         the chunk is taken relative to the pose the base planned
                         from, the residual is summed there, and each resulting
                         target is dispatched as the one-step EE_DELTA from the
                         pose measured at that step (env_wrapper, "Chunk-start-
                         relative targets").
  ft_policy_*.zip (SB3)  multi-fast's FAST residual on the analytic reach base:
                         reach_residual.run, recorded as episodes.hdf5 plus the
                         same viz.py HTML per episode (--policy both runs the
                         base alone and then base + residual on one curve).

  python residual_wrapper/run_residual.py --base-policy <ckpt> --residual-policy best.pt
  python residual_wrapper/run_residual.py --residual-policy ft_policy_400000_steps.zip --num-episodes 5
"""

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

import env_wrapper
from viz import EpisodeRecorder, save_episode_html, save_rollout_html, save_policy_pcd_npz
from env_wrapper import (
    ee_pose_to_world,
    to_sim_world_points,
    to_sim_world_pose,
    to_sim_world_twist,
    _ACTION_KEYS,
    _ARM_KEY,
    _CHUNK_EXEC,
    _RESIDUAL_HORIZON,
    _STATE_OBS_KEYS,
    _ROT_SCALE,
    chunk_to_relative,
    compose_chunk,
    current_ee_pose,
    delta_action,
    measured_ee_twist_world,
    relative_to_poses,
    split_gripper,
    strip_depth,
    target_to_delta,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from policy_wrapper import BasePolicy, ResidualPolicy, Trajectory

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.force_log import ForceLog, WrenchTrace, note as force_note  # noqa: E402
from baselines.rollout_common import Episode, frames_in_progress  # noqa: E402

# This method's name in the shared outputs tree, alongside sail and bspline.
METHOD = "multifast"
from env_wrapper import default_home_q as _default_home_q  # noqa: E402

_POSES_DIR = fc.home_poses_dir()


def _stdin_key_pressed() -> bool:
    """Return True if a key has been pressed on stdin (non-blocking)."""
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _read_key() -> str:
    """Read one keypress from stdin (caller must be in raw mode).

    Returns 'right' for right-arrow, 'left' for left-arrow, 'ctrl_c' for Ctrl-C,
    or '' for anything else.

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
    if data.startswith(b"\x1b[D") or data.startswith(b"\x1bOD"):
        return "left"
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
    # No resume branch: each run gets a fresh directory, so there is never an
    # existing dataset at this root to continue into.
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

    # The anchor: the measured base-frame pose at the observation the base
    # planned from -- the same snapshot send_action anchors its deltas on, and
    # the frame an EE_POS recording's targets were compared against.
    if kin is None:
        raise RuntimeError("no kinematic snapshot to anchor the chunk on; "
                           "get_observation() must precede inference")
    anchor_pos = np.asarray(kin[_ARM_KEY][3], dtype=np.float64).copy()
    anchor_quat = np.asarray(kin[_ARM_KEY][4], dtype=np.float64).copy()
    base_rel = chunk_to_relative(base_chunk, anchor_pos, anchor_quat)

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
        residual_obs = {
            "action_chunk": base_rel[:_RESIDUAL_HORIZON],
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
                anchor_pos=anchor_pos.astype(np.float32),
                anchor_quat=anchor_quat.astype(np.float32),
                action_chunk=residual_obs["action_chunk"],
                proprio=residual_obs["proprio"],
                point_cloud=residual_obs["point_cloud"],
                network_pcd=network_pcd,
                res_chunk=res_chunk.astype(np.float32),
            )

    # Composed once per chunk, as predict_diffused composes the reach chunk; the
    # bound is the chunk length, the bound chunk-start-relative targets get.
    total_rel = compose_chunk(base_rel, res_chunk, bound=float(len(base_rel)))
    return {
        "base_chunk": base_chunk,
        "base_rel": base_rel,
        "res_chunk": res_chunk,
        "total_rel": total_rel,
        "anchor_pos": anchor_pos,
        "anchor_quat": anchor_quat,
        "base_poses": relative_to_poses(base_rel, anchor_pos, anchor_quat),
        "total_poses": relative_to_poses(total_rel, anchor_pos, anchor_quat),
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
    ep: "object | None" = None,
    wrench: "WrenchTrace | None" = None,
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
        wrench: optional WrenchTrace; sampled after every goal sent.
    """
    base_policy.reset()
    # The residual is composed into the targets below; nothing rides on the
    # robot's own cached offset, as on the reach path.
    controller.cache_delta(np.zeros(3), np.zeros(3))

    base_rel: np.ndarray = np.empty((0, 9))
    res_chunk: np.ndarray = np.empty((0, 9))
    exec_rel: np.ndarray = np.empty((0, 9))
    base_poses: np.ndarray = np.empty((0, 7))
    exec_poses: np.ndarray = np.empty((0, 7))
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
                if ep is not None:
                    ep.verdict, ep.success = "timeout", False
                    ep.wall_time_s = t_step - t_start
                break
            if _stdin_key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key in ("right", "left"):
                    # The operator's verdict, not just a stop: without it there
                    # is no time-to-success to compare this method against the
                    # baselines on. Right = success, left = failure, matching
                    # baselines/rollout_common.py's Stopper.
                    verdict = "success" if key == "right" else "failure"
                    if ep is not None:
                        ep.verdict, ep.success = verdict, key == "right"
                        ep.wall_time_s = t_step - t_start
                    print(f"\r\n{verdict}\r", flush=True)
                    break
            obs = controller.get_observation()
            ee_pose = current_ee_pose(obs, sim_convention=sim_proprio_convention)
            obs_no_depth = strip_depth(obs)
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
                if ep is not None:
                    ep.inferences = infer_idx
                base_rel = result["base_rel"]
                res_chunk = result["res_chunk"]
                base_poses = result["base_poses"]
                # One anchor per chunk: the pose the base planned from, which with
                # a prefetch is infer_lead steps behind this one. When replaying
                # a recording the residual is visualised only; the recording's
                # own targets and gains drive the arm.
                exec_rel = base_rel if replaying else result["total_rel"]
                exec_poses = base_poses if replaying else result["total_poses"]
                chunk_used = 0

                if recorder is not None and result["network_pcd"] is not None:
                    recorder.record_policy_pcd(len(recorder), result["network_pcd"])

                if recorder is not None:
                    recorder.record_chunk(
                        step=len(recorder),
                        ee_pos=result["anchor_pos"].astype(np.float32),
                        base_traj=base_poses[:, :3].astype(np.float32),
                        total_traj=result["total_poses"][:, :3].astype(np.float32),
                        base_traj_pose=base_poses.astype(np.float32),
                        total_traj_pose=result["total_poses"].astype(np.float32),
                    )

            # Stage two of the executor: the target this step aims at was fixed at
            # the chunk's composition; the delta that lands on it is re-expressed
            # against the pose measured now and clipped to one step.
            pos_now = np.asarray(kin_snapshot[_ARM_KEY][3], dtype=np.float64)
            quat_now = np.asarray(kin_snapshot[_ARM_KEY][4], dtype=np.float64)
            target = exec_poses[chunk_used + 1]
            row = exec_rel[chunk_used]
            kp, kd = float(row[7]), float(row[8])
            action = delta_action(target_to_delta(target, pos_now, quat_now),
                                  gripper=float(row[6]), kp=kp, kd=kd)
            res = res_chunk[chunk_used] if chunk_used < len(res_chunk) else None
            drot = (exec_rel[chunk_used, 3:6] - base_rel[chunk_used, 3:6]) * _ROT_SCALE
            t_send = time.perf_counter()
            controller.send_action(action)
            if wrench is not None:
                wrench.sample(controller)
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
                # Targets and the measured pose in one frame (base, metres): the
                # anchor's, not the sim-convention proprio pose the residual reads.
                recorder.record(
                    q=q,
                    actual_ee_pos=pos_now.astype(np.float32),
                    base_desired_pos=base_poses[chunk_used + 1, :3].astype(np.float32),
                    total_desired_pos=target[:3].astype(np.float32),
                    kp=kp,
                    kd=kd,
                    gripper=action[f"{_ARM_KEY}_gripper"],
                    res_gripper=float(res[8]) if res is not None else 0.0,
                    point_cloud=controller.last_full_point_cloud,
                    res_rotvec=drot.astype(np.float32),
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
            if ep is not None:
                ep.steps = step_idx


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


def _base_policy_dataset(base_policy: str | None) -> str | None:
    """The LeRobot dataset a base-policy checkpoint was trained on.

    LeRobot writes `train_config.json` next to `pretrained_model/`, and it
    carries `dataset.repo_id`. Best-effort: an older or hand-assembled
    checkpoint simply has none, and the rollout asks for --train-dataset.
    """
    if not base_policy:
        return None
    here = Path(base_policy).expanduser()
    for candidate in (here / "train_config.json",
                      here.parent / "train_config.json",
                      here.parent.parent / "train_config.json"):
        if not candidate.is_file():
            continue
        try:
            cfg = json.loads(candidate.read_text())
        except Exception:
            continue
        repo_id = (cfg.get("dataset") or {}).get("repo_id")
        if repo_id:
            return repo_id
    return None


def _policy_record(args, base_action_space: str) -> dict:
    """Both halves of this method are policies; both are provenance."""
    return {
        "base_policy": rr.file_provenance(args.base_policy),
        "residual_policy": (None if args.no_residual
                            else rr.file_provenance(args.residual_policy)),
        "residual_enabled": not args.no_residual,
        "replay_dataset": args.replay_dataset,
        "device": args.device,
        "base_action_space": base_action_space,
        "control_mode": "EE_DELTA",
        "control_mode_source": "fixed",
        "control_mode_reason": "the base policy emits absolute EE_POS targets; each is "
                               "executed as the one-step delta from the pose measured at "
                               "that step, as the reach executor runs its base",
    }


def _parameter_record(args) -> dict:
    """Every knob this runner resolved, including the normalisation contract the
    checkpoints were trained against -- changing any of it invalidates them."""
    return {
        "exec_fps": float(args.fps),
        "obs_fps": float(args.fps),
        "num_episodes": args.num_episodes,
        "episode_time_s": args.episode_time_s,
        "task": args.task,
        "infer_lead": args.infer_lead,
        "proprio_frame": args.proprio_frame,
        "sim_proprio_convention": not args.raw_proprio,
        "base_amp": args.base_amp,
        "base_compile": args.base_compile,
        "residual": {
            "chunk_exec": fc.policy("residual.chunk_exec"),
            "horizon": fc.policy("residual.horizon"),
            "pos_scale_m": fc.policy("residual.pos_scale_m"),
            "rot_scale_rad": fc.policy("residual.rot_scale_rad"),
            "gains_mag": fc.policy("residual.gains_mag"),
            "residual_mag": fc.policy("residual.residual_mag"),
            "residual_trans_mag": fc.policy("residual.residual_trans_mag"),
            "residual_rot_mag": fc.policy("residual.residual_rot_mag"),
            "res_pos_gain": fc.policy("residual.res_pos_gain"),
            "res_rot_gain": fc.policy("residual.res_rot_gain"),
        },
    }


def _environment_record(args, controller) -> dict:
    """The same shape the baseline bridges record, so the three are comparable.

    Imported rather than restated: rollout_common.environment resolves the rig
    profile, the physical arm behind the `r_` prefix, and the torque/tuning
    blocks a sim-real comparison turns on.
    """
    from baselines.rollout_common import environment
    shim = argparse.Namespace(
        rig=controller.config.rig_profile, home_pose_name=args.home_pose_name,
        home_q=args.home_q, dry_run=False,
    )
    return environment(shim, controller)


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", required=False, help="Path to base policy checkpoint")
    parser.add_argument("--rig", choices=sorted(env_wrapper._RIGS), default=env_wrapper._PROFILE,
                        help="rig profile for a LeRobot base; which physical arm it drives "
                             "is in config/rig.yaml (the reach path uses --arm)")
    parser.add_argument(
        "--residual-policy",
        default=str(Path(__file__).resolve().parent.parent / "best.pt"),
        help="Residual checkpoint: best.pt (point-cloud residual) or a FAST .zip (reach)",
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

    # Recording. The run directory owns every output path, so there is no
    # --output-dir: a run is one directory you can archive or delete whole.
    parser.add_argument("--repo-id", default=None,
                        help="repo id for the LeRobotDataset recorded during the run; "
                             "default <train-dataset>-<method>-<timestamp>")
    parser.add_argument("--task", default="multifast rollout",
                        help="Single-task description stored with each episode")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--episode-time-s", type=float, default=60.0,
                        help="Per-episode timeout in seconds; a timeout is a failure")
    parser.add_argument("--fps", type=int, default=fc.control_fps(),
                        help="Control and dataset rate")
    parser.add_argument("--push-to-hub", type=_str2bool, default=False,
                        help="Push dataset to HuggingFace Hub after recording")
    parser.add_argument("--viz-dir", default=None,
                        help="Plotly HTML output; defaults to the run directory")
    parser.add_argument("--viz-stride", type=int, default=1,
                        help="Animate every Nth step in the visualization (default 1)")
    parser.add_argument("--replay-dataset", default=None, help="HuggingFace id for the dataset to replay from")
    parser.add_argument("--train-dataset", default=None,
                        help="repo id of the dataset the base policy was TRAINED on; "
                             "names the output directory. Optional when the checkpoint's "
                             "train_config.json carries it")
    parser.add_argument("--outputs-root", default=None,
                        help=f"default {rr.DEFAULT_ROOT}")
    parser.add_argument("--no-record", action="store_true",
                        help="skip the LeRobotDataset (manifest and episodes are always written)")

    reach = parser.add_argument_group("reach residual (FAST .zip checkpoint only; also "
                                      "reads --num-episodes, --device, --viz-stride)")
    reach.add_argument("--seed", type=int, default=0, help="curve sampling seed")
    reach.add_argument("--arm", default="left", choices=("left", "right"),
                       help="physical arm to drive; the key prefix stays r_ either way")
    reach.add_argument("--policy", default="residual", choices=("residual", "base", "both"),
                       help="per episode: the residual on its base, the base alone, or both "
                            "in succession on the same curve")
    reach.add_argument("--out", default=str(Path.home() / "franka_data" / "reach_residual"),
                       help="run directories go under here")
    reach.add_argument("--no-viz", action="store_true",
                       help="write episodes.hdf5 only, no episode HTML")

    args = parser.parse_args()
    if Path(args.residual_policy).suffix == ".zip":
        # A different base, loop and record entirely; nothing below applies.
        import reach_residual
        reach_residual.run(args.residual_policy, args.num_episodes, args.seed, args.arm,
                           args.policy, args.device, args.out, args.viz_stride, args.no_viz)
        return
    # --output-dir is gone; _build_dataset still reads it, and main() points it
    # at the run directory's own dataset/ once that exists.
    args.output_dir = None

    logging.basicConfig(level=logging.INFO, force=True)

    if args.home_q is not None:
        home_q = np.asarray(args.home_q, dtype=np.float64)
        home_gripper = args.home_gripper
    else:
        pose = fc.load_home_pose(args.home_pose_name)
        home_q = _default_home_q(args.home_pose_name)
        home_gripper = float(pose.get("gripper", args.home_gripper))

    print("attempting connection to robot...")
    controller = env_wrapper.start_controller(rig=args.rig)
    print(f"robot initialized: {controller.config.rig_profile} -> physical arm "
          f"{controller.config.arm_name(_ARM_KEY)!r} at {controller.config.r_robot_ip}, "
          f"cameras {sorted(controller.cameras)}")

    if args.replay_dataset is None:
        print(f"attempting to start base policy: {args.base_policy}")
        base_policy = BasePolicy(args.base_policy, device=args.device,
                                 amp=args.base_amp, compile_mode=args.base_compile)
        print("base policy started!")
    else:
        print(f"attempting to fetch replay dataset: {args.replay_dataset}")
        base_policy = Trajectory(args.replay_dataset, device=args.device)
        print("replay dataset found!")

    # The executor below turns absolute targets into per-step deltas; a base that
    # already emits deltas would have every 5 cm step read as a pose next to the
    # base origin. Decided from the checkpoint's own normalisation stats.
    try:
        base_action_space = base_policy.action_space()
    except (FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"cannot verify the base policy's action space: {exc}") from exc
    if base_action_space != "EE_POS":
        raise SystemExit(
            f"{args.base_policy or args.replay_dataset}: trained on {base_action_space} actions, "
            "and this runner executes absolute EE_POS targets. Train the base on an EE_POS "
            "recording (or relabel one with scripts/replay_dataset.py --mode ee_pose)."
        )
    print(f"base policy action space: {base_action_space}")

    # The single-arm rigs expose different cameras; refuse before homing rather
    # than KeyError inside the preprocessor mid-episode.
    base_cfg = getattr(getattr(base_policy, "policy", None), "config", None)
    missing = [k for k in (getattr(base_cfg, "image_features", None) or {})
               if k.removeprefix("observation.images.") not in controller.cameras]
    if missing:
        controller.disconnect()
        raise SystemExit(
            f"{args.base_policy}: needs {missing}, and rig {args.rig} has cameras "
            f"{sorted(controller.cameras)}. Pass the --rig it was recorded on.")

    residual: ResidualPolicy | None = None
    if args.no_residual:
        print("residual policy disabled (--no-residual)")
    else:
        print(f"attempting to start residual policy: {args.residual_policy}")
        residual = ResidualPolicy(args.residual_policy, device=args.device)
        print("residual policy started")

    _warmup_policies(controller, base_policy, residual,
                     args.proprio_frame, not args.raw_proprio)

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

    train_dataset = rr.resolve_train_dataset(
        args.train_dataset, _base_policy_dataset(args.base_policy), None)
    run_dir = rr.RunDir(train_dataset["repo_id"], METHOD,
                        root=args.outputs_root or rr.DEFAULT_ROOT)
    record = rr.RunRecord(run_dir, METHOD, train_dataset)
    print(f"run directory: {run_dir.path}")

    if args.repo_id is None:
        args.repo_id = f"{Path(train_dataset['repo_id']).name}-{run_dir.run_id}"
    args.output_dir = str(run_dir.dataset_dir)
    if args.viz_dir is None:
        args.viz_dir = str(run_dir.path)

    record.set("policy", **_policy_record(args, base_action_space))
    record.set("parameters", **_parameter_record(args))
    record.set("environment", **_environment_record(args, controller))
    forces = ForceLog(run_dir.force_profiles_path)
    record.set("outputs", force_profiles=str(run_dir.force_profiles_path))

    dataset = None
    encoder = None
    status, reason = "completed", None
    try:
        if not args.no_record:
            dataset = _build_dataset(args, controller)
            encoder = VideoEncodingManager(dataset)
            encoder.__enter__()
        record.set("outputs", dataset={
            "recorded": dataset is not None,
            "repo_id": args.repo_id if dataset is not None else None,
            "path": str(run_dir.dataset_dir) if dataset is not None else None,
            "fps": args.fps,
        })

        print("homing...")
        homed = bool(controller.home(**home_kwargs))
        if not homed:
            logger.warning("homing did not converge; proceeding anyway")

        for ep_idx in range(args.num_episodes):
            print(f"\r\nepisode {ep_idx + 1}/{args.num_episodes}: place the scene, "
                  f"then press RIGHT ARROW to start\r", flush=True)
            _wait_for_right_arrow()
            print(f"\r\nrunning ({args.episode_time_s:.0f}s max). "
                  f"RIGHT = success, LEFT = failure, Ctrl-C = abort\r", flush=True)

            ep = Episode(episode=ep_idx, homed=homed, started_at=rr.stamp(),
                         exec_fps=float(args.fps))
            recorder = EpisodeRecorder() if args.viz_dir else None
            wrench = WrenchTrace(_ARM_KEY)
            t0 = time.perf_counter()
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
                    dump_dir=dump_root / f"ep{ep_idx:03d}" if dump_root else None,
                    video_dir=run_dir.video_dir if args.save_videos else None,
                    video_cams=args.video_cams,
                    video_stem=f"episode_{ep_idx:03d}",
                    infer_lead=args.infer_lead,
                    ep=ep,
                    wrench=wrench,
                )
            finally:
                if not ep.wall_time_s:
                    ep.wall_time_s = time.perf_counter() - t0
                ep.ended_at = rr.stamp()
                # Anything that reached here without a verdict did not end on
                # the operator's say-so or the clock -- a Ctrl-C, an exception.
                ep.verdict = ep.verdict or "incomplete"
                if ep.wall_time_s > 0:
                    ep.achieved_fps = round(ep.steps / ep.wall_time_s, 2)
                ep.ee_force_n = forces.add_trace(ep_idx, wrench)
                if dataset is not None:
                    ep.frames_recorded = frames_in_progress(dataset)
                    ep.dataset_episode_index = dataset.num_episodes
                if recorder is not None and len(recorder) > 0:
                    viz_path = os.path.join(args.viz_dir, f"episode_{ep_idx:03d}.html")
                    print(f"saving visualization to {viz_path}...")
                    _save_viz(recorder, viz_path, residual,
                              f"episode {ep_idx} — {args.task}", args.viz_stride, args.fps)
                    ep.notes["viz"] = os.path.basename(viz_path)
                record.add_episode(ep)

            print(f"\r\nepisode {ep_idx}: {ep.verdict.upper()} "
                  f"in {ep.wall_time_s:.2f}s, {ep.steps} steps{force_note(ep.ee_force_n)}\r", flush=True)
            if dataset is not None:
                dataset.save_episode()
            if ep_idx < args.num_episodes - 1:
                print("resetting environment — homing arm before next episode...")
                homed = bool(controller.home(**home_kwargs))
                if not homed:
                    logger.warning("homing did not converge; proceeding anyway")
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt at the robot"
        print("\r\ninterrupted\r", flush=True)
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if dataset is not None:
            if encoder is not None:
                encoder.__exit__(None, None, None)
            dataset.finalize()
            record.set("outputs", dataset={
                "recorded": True, "repo_id": args.repo_id,
                **rr.describe_lerobot_dataset(run_dir.dataset_dir),
            })
            if args.push_to_hub:
                try:
                    dataset.push_to_hub()
                    record.set("outputs", pushed_to_hub=True)
                except Exception:
                    logger.exception("push_to_hub failed; dataset is on disk at %s",
                                     run_dir.dataset_dir)
                    record.set("outputs", pushed_to_hub=False)
        if args.save_videos and run_dir.video_dir.is_dir():
            record.set("outputs",
                       videos=sorted(v.name for v in run_dir.video_dir.glob("*.mp4")))
        record.finish(status, reason)
        print(f"\r\nrun written to {run_dir.path}\r", flush=True)
        controller.disconnect()


if __name__ == "__main__":
    main()
