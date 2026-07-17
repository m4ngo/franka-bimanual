"""System-ID trajectory replay: drive the robot along a reference trajectory and record response.

Usage
-----
python sysid/sysid.py <traj.hdf5> [--output sysid/data_replayed.hdf5] [--fps 20] [--kp 0] [--kd 0]

The input HDF5 must have the structure produced by the sim:
    f[group][episode][field] → (T, D) dataset

Fields consumed from the input:
    action  (T, 7) – [dx_norm, dy_norm, dz_norm, rx_norm, ry_norm, rz_norm, gripper]
                     Position deltas in units of _POS_SCALE (0.05 m); rotation as
                     axis-angle in units of _ROT_SCALE (0.5 rad).
    eef_pos (T, 3) – reference EE position (used for error visualisation only)
    qpos    (T, 7) – joint angles used to initialise the home pose

Fields recorded in the output HDF5 (same structure):
    action      (T, 7) – [dpos(3), drot_quat(4)] — position delta in metres and
                         rotation delta as unnormalized quaternion [drot/2, 1] (xyzw)
    eef_ang_vel (T, 3) – actual EE angular velocity from robot state
    eef_lin_vel (T, 3) – actual EE linear velocity from robot state
    eef_pos     (T, 3) – actual EE position from robot state
    eef_quat    (T, 4) – actual EE quaternion from robot state
    qpos        (T, 7) – actual joint angles
    qvel        (T, 7) – actual joint velocities
    t_sim       (T, 1) – wall-clock time since episode start
    tau_cmd     (T, 7) – zeros (not accessible via current RPC interface)

A comparison HTML visualization is also written alongside the HDF5.
One MP4 video per camera is written alongside the HDF5 (e.g. <stem>_cam_3_wrist.mp4).
"""

import argparse
import hashlib
import json
import logging
import os
import select
import socket
import sys
import termios
import tty
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "residual_wrapper"))

from env_wrapper import start_controller  # noqa: E402
from lerobot_robot_bimanual_franka import SingleArmFranka
from lerobot_robot_bimanual_franka import (  # noqa: E402  (constants snapshot for run.json)
    bimanual_franka as _bf_mod,
    franka_process as _fp_mod,
    osc_velocity_controller as _osc_mod,
    safety as _safety_mod,
)
from _viz import compute_trajectory_errors, save_aggregate_html, save_comparison_html, save_errors_json  # noqa: E402

logger = logging.getLogger(__name__)

# Action kp/kd in [-1, 1]; send_action maps them via kp_gain = 10**kp.
# Default 0.0 → kp_gain = 1.0 (minimum, safest for an open-loop sysid replay).
_DEFAULT_KP = 0.0
_DEFAULT_KD = 0.0

# Camera read timeout used when capturing frames inside the step loop.
# 50 ms is generous for a buffered async_read at 20 Hz control rate.
_CAM_TIMEOUT_MS = 50.0


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def parse_num_traj(filename: str) -> int:
    with h5py.File(filename, "r") as f:
        group_key = list(f.keys())[0]
        group = f[group_key]
        return len(group.keys())

def parse_traj(filename: str, index: int) -> tuple[str, dict[str, np.ndarray]]:
    """Load the first episode from a sysid HDF5 file.

    Expected structure: f[group_key][episode_key][field] → (T, D) dataset.
    All datasets are copied into numpy arrays so the file can be closed.
    """
    traj: dict[str, np.ndarray] = {}
    key = ""
    with h5py.File(filename, "r") as f:
        group_key = list(f.keys())[0]
        group = f[group_key]
        episode_key = list(group.keys())[index]
        key = episode_key
        episode = group[episode_key]
        for field in episode:
            traj[field] = episode[field][:]
    return (key, traj)


# ---------------------------------------------------------------------------
# Keyboard helpers (identical pattern to run_residual.py)
# ---------------------------------------------------------------------------

def _stdin_key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _read_key() -> str:
    time.sleep(0.03)
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):
        return "right"
    return ""


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _save_videos(
    cam_frames: dict[str, list[np.ndarray]],
    video_dir: Path,
    stem: str,
    fps: float,
) -> None:
    """Write one MP4 per camera from lists of RGB frames captured during an episode."""
    video_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam_name, frames in cam_frames.items():
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        vpath = video_dir / f"{stem}_{cam_name}.mp4"
        writer = cv2.VideoWriter(str(vpath), fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame[:, :, ::-1])  # RGB → BGR for OpenCV
        writer.release()
        logger.info("saved video %s (%d frames)", vpath, len(frames))


# ---------------------------------------------------------------------------
# Episode replay loop
# ---------------------------------------------------------------------------

def _run_episode(
    controller: SingleArmFranka,
    traj: dict[str, np.ndarray],
    fps: float = 20.0,
    kp: float = _DEFAULT_KP,
    kd: float = _DEFAULT_KD,
    gripper_norm: float = 1.0,
    video_dir: Path | None = None,
    video_stem: str = "episode",
) -> dict[str, np.ndarray]:
    """Replay *traj* on the robot and record the kinematic response.

    At each step the robot kinematic state is read via a direct call to
    ``controller.robot_manager.current_kinematic_state_batch`` (which returns
    q, dq, jacobian, ee_pos, ee_quat, ee_vel_6d).  That snapshot is stored in
    ``controller._cached_kin_state`` so the subsequent ``send_action`` call
    consumes it without an extra RPyC round-trip.

    Camera frames are captured asynchronously at each step using the
    controller's thread pool.  After the loop, one MP4 is written per camera
    into *video_dir* (skipped if *video_dir* is None or there are no cameras).

    Press right-arrow to end early, Ctrl-C to abort.

    Returns a dict of stacked numpy arrays (one row per step).
    """
    action_all = traj["action"]  # (T, 7): [dx_norm, dy_norm, dz_norm, rx_norm, ry_norm, rz_norm, gripper]
    n_steps = len(action_all)

    _POS_SCALE = 0.05  # metres per normalised unit (must match env_wrapper._POS_SCALE)
    _ROT_SCALE = 0.5   # radians per normalised unit (must match env_wrapper._ROT_SCALE)

    buf: dict[str, list] = {k: [] for k in (
        "action", "action_norm", "eef_ang_vel", "eef_lin_vel", "eef_pos",
        "eef_quat", "fault_count", "qpos", "qvel", "t_sim", "tau_cmd",
    )}

    record_video = video_dir is not None and bool(controller.cameras)
    cam_frames: dict[str, list[np.ndarray]] = {n: [] for n in controller.cameras} if record_video else {}

    dt = 1.0 / fps
    t_start = time.perf_counter()
    old_term = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        for step in range(n_steps):
            t_step = time.perf_counter()

            if _stdin_key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "right":
                    print("\r\nearly stop requested", flush=True)
                    break

            # --- read kinematic state ----------------------------------------
            # Store in _cached_kin_state so send_action re-uses it, avoiding
            # a redundant RPyC round-trip.
            kin = controller.robot_manager.current_kinematic_state_batch(["r"])
            controller._cached_kin_state = kin
            q, dq, _jac, ee_pos, ee_quat, ee_vel = kin["r"]
            # ee_vel layout: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]

            # --- build and send action ----------------------------------------
            dpos: np.ndarray = action_all[step][0:3].astype(np.float64) * _POS_SCALE
            drot: np.ndarray = action_all[step][3:6].astype(np.float64) * _ROT_SCALE

            # Convert axis-angle drot to unnormalized quaternion [drot/2, 1] (xyzw).
            # _ee_delta recovers the axis-angle via 2·arctan2(|v|, w) which gives
            # back drot exactly for small angles and is consistent for larger angles.
            drot_quat = np.array([drot[0] / 2, drot[1] / 2, drot[2] / 2, 1.0], dtype=np.float64)

            action = {
                "r_x":       float(dpos[0]),
                "r_y":       float(dpos[1]),
                "r_z":       float(dpos[2]),
                "r_qx":      float(drot_quat[0]),
                "r_qy":      float(drot_quat[1]),
                "r_qz":      float(drot_quat[2]),
                "r_qw":      float(drot_quat[3]),
                "r_gripper": float(gripper_norm),
                "kp":        kp,
                "kd":        kd,
            }
            controller.send_action(action)

            # --- submit camera reads (async, resolved before next sleep) ------
            if record_video:
                cam_futs = {
                    n: controller._camera_pool.submit(cam.async_read, _CAM_TIMEOUT_MS)
                    for n, cam in controller.cameras.items()
                }

            # --- record kinematic data ----------------------------------------
            t_now = time.perf_counter() - t_start
            buf["action"].append(np.concatenate([dpos, drot_quat]).astype(np.float32))
            buf["action_norm"].append(action_all[step].astype(np.float32))
            # Cumulative recoverable-error recoveries (reflexes etc.) so analysis
            # can flag ticks where tracking was interrupted. Local attribute read.
            buf["fault_count"].append(np.int32(controller.robot_manager.recovery_counts().get("r", 0)))
            buf["eef_ang_vel"].append(np.asarray(ee_vel[3:], dtype=np.float32))
            buf["eef_lin_vel"].append(np.asarray(ee_vel[:3], dtype=np.float32))
            buf["eef_pos"].append(np.asarray(ee_pos, dtype=np.float32))
            buf["eef_quat"].append(np.asarray(ee_quat, dtype=np.float32))
            buf["qpos"].append(np.asarray(q, dtype=np.float32))
            buf["qvel"].append(np.asarray(dq, dtype=np.float32))
            buf["t_sim"].append(np.array([t_now], dtype=np.float32))
            buf["tau_cmd"].append(np.zeros(7, dtype=np.float32))

            elapsed = time.perf_counter() - t_step
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            # --- collect camera frames after sleeping -------------------------
            if record_video:
                for n, fut in cam_futs.items():
                    try:
                        frame = fut.result(timeout=0.1)
                        cam_frames[n].append(frame)
                    except Exception as e:
                        logger.warning("Camera %s frame dropped at step %d: %s", n, step, e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    if record_video and cam_frames:
        _save_videos(cam_frames, video_dir, video_stem, fps)

    return {k: np.stack(v) for k, v in buf.items() if v}


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_sysid_hdf5(recorded: dict[str, np.ndarray], path: str, attrs: dict | None = None) -> None:
    """Write the recorded episode to an HDF5 file with the sim-compatible layout.

    ``attrs`` are stamped onto the episode group so each file stays
    interpretable when separated from its run directory (None values skipped).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        ep = f.create_group("data").create_group("episode_0")
        for field, arr in recorded.items():
            ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
        for key, val in (attrs or {}).items():
            if val is not None:
                ep.attrs[key] = val
    logger.info("saved %d steps to %s", next(iter(recorded.values())).shape[0], path)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

# Stack constants snapshotted into run.json, read live from the modules so the
# record can't drift from the code. getattr(..., None): a renamed constant
# shows up as null in the JSON instead of crashing the run.
_METADATA_CONSTANTS: dict[str, tuple[object, tuple[str, ...]]] = {
    "bimanual_franka": (_bf_mod, (
        "_EE_TRANSLATION_FUDGE_FACTOR", "_EE_ROTATION_FUDGE_FACTOR",
        "OSC_BASE_KP", "_KP_GAIN_BASE", "_KD_GAIN_BASE",
        "EE_PD_KP", "EE_PD_KD", "JOINT_PD_KP", "JOINT_PD_KD",
    )),
    "osc_velocity_controller": (_osc_mod, (
        "DEFAULT_KP", "DEFAULT_DAMPING_RATIO", "DEFAULT_NULLSPACE_KP",
        "DEFAULT_DLS_DAMPING", "DEFAULT_MAX_QDOT",
    )),
    "safety": (_safety_mod, (
        "JOINT_VELOCITY_MAX", "EE_LINEAR_VELOCITY_MAX", "EE_ANGULAR_VELOCITY_MAX",
        "WORKTABLE_HEIGHT", "WORKTABLE_DISTANCE_MIN", "WORKTABLE_MAX_DECEL",
        "CUSTOM_END_EFFECTOR_Z_EXTENSION",
    )),
    "franka_process": (_fp_mod, (
        "VELOCITY_COMMAND_DURATION_MS", "_JOINT_RELATIVE_DYNAMICS",
        "_EE_DELTA_RELATIVE_DYNAMICS", "_JOINT_STIFFNESS",
        "_TORQUE_THRESHOLD", "_FORCE_THRESHOLD",
    )),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_run_metadata(args: argparse.Namespace, episode_names: list[str]) -> dict:
    """Everything needed to interpret a run directory later, gathered up front."""
    constants = {
        mod_name: {name: getattr(mod, name, None) for name in names}
        for mod_name, (mod, names) in _METADATA_CONSTANTS.items()
    }
    kp_gain = 10.0 ** args.kp
    osc_base_kp = constants["bimanual_franka"]["OSC_BASE_KP"]
    return {
        "status": "running",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "args": vars(args),
        "traj_file": str(Path(args.traj_file).resolve()),
        "traj_file_sha256": _sha256(args.traj_file),
        "episodes": episode_names,
        "episodes_completed": [],
        "derived_gains": {
            "kp_gain": kp_gain,
            "effective_velocity_kp": kp_gain * osc_base_kp if osc_base_kp is not None else None,
            "kd_note": "kd action is inert on this stack (_KD_GAIN_BASE == 1.0; "
                       "OSCVelocityController's law has no kd term)",
        },
        "constants": constants,
    }


def _write_run_json(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a reference trajectory on the Franka and record the kinematic response."
    )
    parser.add_argument("traj_file", help="Input HDF5 trajectory file to replay")
    parser.add_argument("--fps", type=float, default=20.0, help="Control rate in Hz")
    parser.add_argument(
        "--kp", type=float, default=_DEFAULT_KP,
        help="EE PD kp in [-1, 1]; maps to gain 10**kp (default 0 → gain 1.0)",
    )
    parser.add_argument(
        "--kd", type=float, default=_DEFAULT_KD,
        help="EE PD kd in [-1, 1] (default 0)",
    )
    parser.add_argument("--gripper-norm", type=float, default=1.0,
                        help="Gripper openness [0, 1] held constant during replay")
    parser.add_argument("--home-max-time-s", type=float, default=5.0,
                        help="Maximum seconds allowed for the homing move")
    parser.add_argument("--home-tol-rad", type=float, default=0.005,
                        help="Joint-angle convergence tolerance (rad) for homing")
    parser.add_argument("--home-tol-m", type=float, default=0.005,
                        help="EE position convergence tolerance (m) for homing")
    parser.add_argument("--viz-stride", type=int, default=1,
                        help="Animate every Nth step in the visualization (use 2-4 for long episodes)")
    parser.add_argument("--out-root", default="~/sysid/outputs",
                        help="Parent directory for per-run output directories")
    parser.add_argument("--tag", default=None,
                        help="Run-directory suffix; defaults to the reference dataset's "
                             "parent directory name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG)

    # Run directory: <out_root>/<timestamp>_<tag>. The tag defaults to the
    # reference dataset's parent directory name — datasets live one per
    # directory, so the directory name carries the sim condition
    # (e.g. kp_actn0.50_damp_actn0.50).
    tag = args.tag or Path(args.traj_file).resolve().parent.name
    out_root = Path(args.out_root).expanduser()
    run_dir = out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"

    with h5py.File(args.traj_file, "r") as f:
        episode_names = list(f[list(f.keys())[0]].keys())

    meta = _collect_run_metadata(args, episode_names)
    _write_run_json(run_dir, meta)
    logger.info("run directory: %s", run_dir)

    # Per-episode HDF5 attrs: enough to interpret a file separated from run.json.
    base_attrs = {
        "traj_file": meta["traj_file"],
        "kp": args.kp,
        "kd": args.kd,
        "fps": args.fps,
        "gripper_norm": args.gripper_norm,
        "kp_gain": 10.0 ** args.kp,
        "ee_translation_fudge_factor": getattr(_bf_mod, "_EE_TRANSLATION_FUDGE_FACTOR", None),
        "ee_rotation_fudge_factor": getattr(_bf_mod, "_EE_ROTATION_FUDGE_FACTOR", None),
        "osc_base_kp": getattr(_bf_mod, "OSC_BASE_KP", None),
        "max_qdot": getattr(_osc_mod, "DEFAULT_MAX_QDOT", None),
    }

    controller = None
    all_errors: list[dict] = []
    episode_pairs: list[tuple[str, dict, dict]] = []
    try:
        # Connect robot
        logger.info("connecting to robot...")
        controller = start_controller()
        logger.info("robot connected")

        for i in range(0, len(episode_names)):
            # Load reference trajectory
            logger.info("loading trajectory from %s", args.traj_file)
            name, traj = parse_traj(args.traj_file, i)
            output = "record_" + name
            n_steps = len(traj["eef_pos"])
            logger.info(f"recording trajectory: {name} with {n_steps} steps, keys: {sorted(traj.keys())}")

            # Home to first trajectory position (uses qpos from trajectory if available)
            if "qpos" in traj:
                home_q = traj["qpos"][0].astype(np.float64)
                logger.info("homing to first trajectory qpos: %s", np.round(home_q, 3))
            else:
                logger.warning("trajectory has no 'qpos'; skipping pose-specific homing")
                home_q = None

            if home_q is not None:
                converged = controller.home(
                    home_q_left=None,
                    home_q_right=home_q,
                    gripper_norm=args.gripper_norm,
                    max_time_s=args.home_max_time_s,
                    tol_rad=args.home_tol_rad,
                    tol_pos_m=args.home_tol_m,
                )
                if not converged:
                    logger.warning("homing did not converge; proceeding anyway")

            # Replay and record
            logger.info(
                "replaying %d steps at %.1f Hz (kp=%.2f → gain=%.2f) — press right-arrow to stop early",
                n_steps, args.fps, args.kp, 10.0 ** args.kp,
            )
            video_stem = f"{i}_{output}"
            recorded = _run_episode(
                controller=controller,
                traj=traj,
                fps=args.fps,
                kp=args.kp,
                kd=args.kd,
                gripper_norm=args.gripper_norm,
                video_dir=run_dir,
                video_stem=video_stem,
            )

            if not recorded:
                logger.warning("no steps recorded; stopping sweep")
                break

            # Save HDF5
            save_sysid_hdf5(recorded, str(run_dir / f"{i}_{output}.hdf5"), attrs={
                **base_attrs,
                "reference_episode": name,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            })

            # Visualization
            viz_out = str(run_dir / f"{i}_{output}.html")
            save_comparison_html(traj, recorded, viz_out, fps=args.fps, frame_stride=args.viz_stride)

            # Accumulate errors; rewrite the summary after every episode so an
            # aborted run still keeps stats for what it completed.
            all_errors.append(compute_trajectory_errors(traj, recorded, name=output))
            save_errors_json(all_errors, str(run_dir / "errors.json"))
            meta["episodes_completed"].append(name)
            episode_pairs.append((name, traj, recorded))

        if episode_pairs:
            try:
                save_aggregate_html(episode_pairs, str(run_dir / "aggregate.html"), fps=args.fps)
            except Exception:
                logger.exception("aggregate visualization failed")

        meta["status"] = "completed"
    except BaseException:
        meta["status"] = "aborted"
        raise
    finally:
        _write_run_json(run_dir, meta)
        if controller is not None:
            controller.disconnect()


if __name__ == "__main__":
    main()
