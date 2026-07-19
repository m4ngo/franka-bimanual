"""System-ID collection: drive the robot and record its response (SYSID_UPDATE.md in multi-fast).

Two modes
---------
replay (default): open-loop delta replay of a sim-generated sweep file.
    python sysid/sysid.py <traj.hdf5> [--fps 20] [--kp 0] [--kd 0]

track: closed-loop reference tracking — a pose-space reference path is
    anchored at the measured start pose and each tick commands
    ``delta = (ref[t] - measured) / fudge`` so the pursued goal lands exactly
    on the reference. Both sim (multi-fast absolute replay) and real then
    track the same exogenous goal stream — the well-posed sysid input.
    python sysid/sysid.py --mode track --track-spec spec.json [--hold-s 5]

    spec.json: {"init_qpos": [7], "ramp_s": 2.0,
                "tracks": [{"kind": "sine", "axes": 1, "amp": 0.03,
                            "freq_hz": 0.5, "duration_s": 10.0}, ...]}
    (axes 0-2: position offsets in m; 3-5: rotation rotvec offsets in rad;
     circles take an [u, v] pair. --hold-s prepends a constant-reference HOLD
     episode for static offset calibration.)

--dry-run runs either mode against a kinematic mock (no hardware, no
lerobot/franky imports) to verify the loop, logging, and file outputs.

Replay input HDF5 (sim layout): f[group][episode][field] → (T, D):
    action  (T, 7) – [dx_norm, dy_norm, dz_norm, rx_norm, ry_norm, rz_norm, gripper]
                     Position deltas in units of _POS_SCALE (0.05 m); rotation as
                     axis-angle in units of _ROT_SCALE (0.5 rad).
    eef_pos (T, 3) – reference EE position (used for error visualisation only)
    qpos    (T, 7) – joint angles used to initialise the home pose

Fields recorded in the output HDF5 (both modes):
    action        (T, 7) – [dpos(3), drot_quat(4)] — position delta in metres
                           (pre-fudge, as sent) and rotation delta quaternion
                           (xyzw), EXACT axis-angle encoding (legacy data used
                           the small-angle [drot/2, 1]; see the
                           ``quat_encoding`` attr)
    action_norm   (T, 7) – replay mode only: the normalized sim action replayed
    eef_goal_pos  (T, 3) – goal position the controller pursued (post-fudge)
    eef_goal_quat (T, 4) – goal orientation the controller pursued (xyzw)
    eef_ang_vel   (T, 3) – actual EE angular velocity from robot state
    eef_lin_vel   (T, 3) – actual EE linear velocity from robot state
    eef_pos       (T, 3) – actual EE position from robot state
    eef_quat      (T, 4) – actual EE quaternion from robot state
    fault_count   (T, 1) – cumulative recoverable-error recoveries
    qpos          (T, 7) – actual joint angles
    qvel          (T, 7) – actual joint velocities
    t_sim         (T, 1) – wall-clock time since episode start
    tau_cmd       (T, 7) – zeros (not accessible via current RPC interface)

Episodes are flushed incrementally (atomic tmp+rename, --flush-every steps)
so a crash or Ctrl-C mid-episode keeps the data collected so far.
A comparison HTML visualization is written alongside the HDF5 (replay mode).
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

from types import SimpleNamespace  # noqa: E402

from _viz import compute_trajectory_errors, save_aggregate_html, save_comparison_html, save_errors_json  # noqa: E402

logger = logging.getLogger(__name__)

# Robot-stack modules are imported lazily so --dry-run works on machines
# without lerobot/franky (e.g. the multi-fast dev box). _robot_stack() returns
# the namespace, or None when unavailable and allow_missing is set.
_ROBOT_STACK: SimpleNamespace | None = None


def _robot_stack(allow_missing: bool = False) -> SimpleNamespace | None:
    global _ROBOT_STACK
    if _ROBOT_STACK is not None:
        return _ROBOT_STACK
    try:
        from env_wrapper import start_controller
        from lerobot_robot_bimanual_franka import SingleArmFranka  # noqa: F401
        from lerobot_robot_bimanual_franka import (
            bimanual_franka as bf,
            franka_process as fp,
            osc_velocity_controller as osc,
            safety as sf,
        )
    except ImportError:
        if allow_missing:
            return None
        raise
    _ROBOT_STACK = SimpleNamespace(
        start_controller=start_controller, bf=bf, fp=fp, osc=osc, safety=sf,
    )
    return _ROBOT_STACK

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
# Quaternion / reference-path math (pure numpy; unit-tested off-robot)
# ---------------------------------------------------------------------------

def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """Exact axis-angle → unit quaternion (xyzw). Replaces the legacy
    small-angle [aa/2, 1] encoding (~2% angle shortfall at 0.5 rad); output
    files carry quat_encoding="exact" so analysis can tell datasets apart."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = aa / angle
    return np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, both xyzw."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _rotvec_between(q_goal: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Axis-angle (rad, 3-vector) rotation taking q_cur → q_goal, shortest path."""
    g = np.asarray(q_goal, dtype=np.float64)
    c = np.asarray(q_cur, dtype=np.float64)
    g = g / max(float(np.linalg.norm(g)), 1e-12)
    c = c / max(float(np.linalg.norm(c)), 1e-12)
    q_err = _quat_mul(g, c * np.array([-1.0, -1.0, -1.0, 1.0]))
    if q_err[3] < 0.0:
        q_err = -q_err
    v = q_err[:3]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-9:
        return 2.0 * v
    return (v / v_norm) * 2.0 * np.arctan2(v_norm, float(np.clip(q_err[3], -1.0, 1.0)))


def _amplitude_ramp(n_steps: int, dt: float, ramp_s: float) -> np.ndarray:
    """Ease-in/out scale (0→1→0) so episodes start and end at rest."""
    if ramp_s <= 0.0:
        return np.ones(n_steps)
    t = np.arange(n_steps) * dt
    dur = n_steps * dt
    return np.minimum(1.0, np.minimum(t / ramp_s, np.maximum(0.0, (dur - t) / ramp_s)))


def _reference_offsets(kind: str, axes, amp: float, freq_hz: float,
                       duration_s: float, dt: float, ramp_s: float = 2.0) -> dict:
    """Pose-space reference offsets from the start pose.

    Keep in sync with multi-fast utils/sysid/sweeps.py reference_pose_offsets
    (exact parity is not load-bearing: the pursued goals are dual-logged, so
    the fit consumes the recorded reference, not a regenerated one).
    Sines are zero at t=0; circles use (cos-1, sin) anchoring so the path
    starts AND ends at the start pose. axes 0-2 position (m), 3-5 rotation
    (rad rotvec); circles take a (u, v) pair.
    """
    n = int(round(duration_s / dt))
    t = np.arange(n) * dt
    scale = _amplitude_ramp(n, dt, ramp_s)
    theta = 2.0 * np.pi * freq_hz * t
    pos = np.zeros((n, 3))
    rot = np.zeros((n, 3))

    def _target(axis):
        return (pos, axis) if axis < 3 else (rot, axis - 3)

    if kind == "sine":
        arr, col = _target(int(axes))
        arr[:, col] = amp * scale * np.sin(theta)
    elif kind == "circle":
        u, v = (int(a) for a in axes)
        arr_u, col_u = _target(u)
        arr_v, col_v = _target(v)
        arr_u[:, col_u] = amp * scale * (np.cos(theta) - 1.0)
        arr_v[:, col_v] = amp * scale * np.sin(theta)
    elif kind == "hold":
        pass  # zero offsets: constant reference at the start pose
    else:
        raise ValueError(f"unknown reference kind {kind!r}")
    return {"pos_offsets": pos, "rotvec_offsets": rot}


def _load_track_spec(path: str, fps: float, hold_s: float) -> tuple[np.ndarray | None, list[tuple[str, dict]]]:
    """Parse a track-spec JSON into (init_qpos, [(name, ref_dict), ...]).

    A --hold-s > 0 prepends a constant-reference HOLD episode (static offset
    calibration tier).
    """
    with open(path) as fh:
        spec = json.load(fh)
    dt = 1.0 / fps
    ramp_s = float(spec.get("ramp_s", 2.0))
    init_qpos = (
        np.asarray(spec["init_qpos"], dtype=np.float64)
        if "init_qpos" in spec else None
    )
    episodes: list[tuple[str, dict]] = []
    if hold_s > 0.0:
        episodes.append(("hold", _reference_offsets("hold", 0, 0.0, 0.0, hold_s, dt, 0.0)))
    for tr in spec["tracks"]:
        kind = tr["kind"]
        axes = tr["axes"]
        name = tr.get("name") or (
            f"{kind}_ax{axes if np.isscalar(axes) else ''.join(str(a) for a in axes)}"
            f"_a{tr['amp']:g}_f{tr['freq_hz']:g}"
        )
        episodes.append((name, _reference_offsets(
            kind, axes, float(tr["amp"]), float(tr["freq_hz"]),
            float(tr["duration_s"]), dt, float(tr.get("ramp_s", ramp_s)),
        )))
    return init_qpos, episodes


# ---------------------------------------------------------------------------
# Dry-run mock (kinematics-only; no hardware / lerobot / franky)
# ---------------------------------------------------------------------------

class _MockController:
    """Duck-typed SingleArmFranka stand-in: first-order EE tracking toward
    each tick's goal, mirroring the real stack's goal semantics (incl. the
    translation fudge). Exercises the full episode loop, logging, flushes and
    file outputs off-robot."""

    cameras: dict = {}

    def __init__(self, trans_fudge: float = 1.2, rate: float = 0.3):
        self._fudge = trans_fudge
        self._rate = rate
        self._q = np.zeros(7)
        self._pos = np.array([0.4, 0.0, 0.4])
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._cached_kin_state = None
        outer = self

        class _RM:
            def current_kinematic_state_batch(self, arms):
                snap = (outer._q.copy(), np.zeros(7), np.zeros((6, 7)),
                        outer._pos.copy(), outer._quat.copy(), np.zeros(6))
                return {a: snap for a in arms}

            def recovery_counts(self):
                return {"r": 0}

        self.robot_manager = _RM()

    def home(self, home_q_left=None, home_q_right=None, **kwargs):
        if home_q_right is not None:
            self._q = np.asarray(home_q_right, dtype=np.float64).copy()
        return True

    def send_action(self, action: dict):
        dpos = np.array([action["r_x"], action["r_y"], action["r_z"]])
        dquat = np.array([action[f"r_q{ax}"] for ax in ("x", "y", "z", "w")])
        goal_pos = self._pos + self._fudge * dpos
        dquat = dquat / max(float(np.linalg.norm(dquat)), 1e-12)
        goal_quat = _quat_mul(dquat, self._quat)
        self._pos = self._pos + self._rate * (goal_pos - self._pos)
        rot_err = _rotvec_between(goal_quat, self._quat)
        self._quat = _quat_mul(_aa_to_quat(self._rate * rot_err), self._quat)
        self._quat /= max(float(np.linalg.norm(self._quat)), 1e-12)
        return action

    def disconnect(self):
        pass


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
    controller,
    traj: dict[str, np.ndarray] | None,
    fps: float = 20.0,
    kp: float = _DEFAULT_KP,
    kd: float = _DEFAULT_KD,
    gripper_norm: float = 1.0,
    video_dir: Path | None = None,
    video_stem: str = "episode",
    ref: dict[str, np.ndarray] | None = None,
    trans_fudge: float = 1.0,
    flush_path: str | None = None,
    flush_attrs: dict | None = None,
    flush_every: int = 100,
    track_abort_m: float = 0.15,
) -> dict[str, np.ndarray]:
    """Run one episode on the robot and record the kinematic response.

    Replay mode (``traj`` given): open-loop delta replay of the sim action
    sequence, using the EXACT axis-angle→quat encoding (legacy runs used the
    small-angle [drot/2, 1]).

    Track mode (``ref`` given): closed-loop reference tracking. The reference
    offsets are anchored at the measured pose of the FIRST tick; each tick
    commands ``dpos = (ref_pos - measured) / trans_fudge`` so the goal the
    controller pursues (post-fudge) lands exactly on the reference. Aborts if
    the position error exceeds ``track_abort_m`` (tracking-runaway guard).

    Both modes dual-log the pursued goal (``eef_goal_pos``/``eef_goal_quat``)
    and flush the buffers atomically to ``flush_path`` every ``flush_every``
    steps so partial episodes survive crashes.

    At each step the robot kinematic state is read via a direct call to
    ``controller.robot_manager.current_kinematic_state_batch`` (which returns
    q, dq, jacobian, ee_pos, ee_quat, ee_vel_6d).  That snapshot is stored in
    ``controller._cached_kin_state`` so the subsequent ``send_action`` call
    consumes it without an extra RPyC round-trip.

    Press right-arrow to end early, Ctrl-C to abort.

    Returns a dict of stacked numpy arrays (one row per step).
    """
    assert (traj is None) != (ref is None), "exactly one of traj / ref"
    track = ref is not None
    if track:
        n_steps = len(ref["pos_offsets"])
    else:
        action_all = traj["action"]  # (T, 7): [d*_norm x6, gripper]
        n_steps = len(action_all)

    _POS_SCALE = 0.05  # metres per normalised unit (must match env_wrapper._POS_SCALE)
    _ROT_SCALE = 0.5   # radians per normalised unit (must match env_wrapper._ROT_SCALE)

    buf: dict[str, list] = {k: [] for k in (
        "action", "action_norm", "eef_goal_pos", "eef_goal_quat",
        "eef_ang_vel", "eef_lin_vel", "eef_pos",
        "eef_quat", "fault_count", "qpos", "qvel", "t_sim", "tau_cmd",
    )}

    record_video = video_dir is not None and bool(controller.cameras)
    cam_frames: dict[str, list[np.ndarray]] = {n: [] for n in controller.cameras} if record_video else {}

    dt = 1.0 / fps
    t_start = time.perf_counter()
    start_pos = start_quat = None  # track-mode anchor, set on the first tick
    # Keyboard early-stop only when stdin is a real terminal (dry runs under
    # pipes/CI and nohup'd sessions have no tty to put in raw mode).
    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setraw(sys.stdin)
    try:
        for step in range(n_steps):
            t_step = time.perf_counter()

            if interactive and _stdin_key_pressed():
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
            ee_pos64 = np.asarray(ee_pos, dtype=np.float64)
            ee_quat64 = np.asarray(ee_quat, dtype=np.float64)

            # --- build the delta command --------------------------------------
            if track:
                if start_pos is None:
                    start_pos, start_quat = ee_pos64.copy(), ee_quat64.copy()
                goal_pos = start_pos + ref["pos_offsets"][step]
                goal_quat = _quat_mul(_aa_to_quat(ref["rotvec_offsets"][step]), start_quat)
                pos_err = goal_pos - ee_pos64
                if float(np.linalg.norm(pos_err)) > track_abort_m:
                    logger.error(
                        "tracking error %.3f m exceeds --track-abort-m %.3f at step %d; aborting episode",
                        float(np.linalg.norm(pos_err)), track_abort_m, step,
                    )
                    break
                # Divide by the fudge so the post-fudge goal the controller
                # pursues is exactly the reference.
                dpos = pos_err / trans_fudge
                drot_quat = _aa_to_quat(_rotvec_between(goal_quat, ee_quat64))
            else:
                dpos = action_all[step][0:3].astype(np.float64) * _POS_SCALE
                drot = action_all[step][3:6].astype(np.float64) * _ROT_SCALE
                drot_quat = _aa_to_quat(drot)
                # Pursued goal (dual log): what _qdot_ee_delta derives from
                # this action — measured pose (+/x) fudged delta.
                goal_pos = ee_pos64 + trans_fudge * dpos
                goal_quat = _quat_mul(drot_quat, ee_quat64)

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
            if not track:
                buf["action_norm"].append(action_all[step].astype(np.float32))
            buf["eef_goal_pos"].append(goal_pos.astype(np.float32))
            gq = goal_quat / max(float(np.linalg.norm(goal_quat)), 1e-12)
            buf["eef_goal_quat"].append(gq.astype(np.float32))
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

            # --- incremental flush (atomic tmp + rename) ----------------------
            if flush_path is not None and (step + 1) % flush_every == 0:
                save_sysid_hdf5(
                    {k: np.stack(v) for k, v in buf.items() if v},
                    flush_path, attrs=flush_attrs, quiet=True,
                )

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
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    if record_video and cam_frames:
        _save_videos(cam_frames, video_dir, video_stem, fps)

    return {k: np.stack(v) for k, v in buf.items() if v}


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_sysid_hdf5(recorded: dict[str, np.ndarray], path: str, attrs: dict | None = None,
                    quiet: bool = False) -> None:
    """Write the recorded episode to an HDF5 file with the sim-compatible layout.

    Atomic (tmp + rename), so it doubles as the mid-episode incremental flush
    — a crash leaves either the previous flush or the new one, never a torn
    file. ``attrs`` are stamped onto the episode group so each file stays
    interpretable when separated from its run directory (None values skipped).
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        ep = f.create_group("data").create_group("episode_0")
        for field, arr in recorded.items():
            ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
        for key, val in (attrs or {}).items():
            if val is not None:
                ep.attrs[key] = val
    os.replace(tmp, path)
    if not quiet:
        logger.info("saved %d steps to %s", next(iter(recorded.values())).shape[0], path)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

# Stack constants snapshotted into run.json, read live from the modules so the
# record can't drift from the code. getattr(..., None): a renamed constant
# shows up as null in the JSON instead of crashing the run. Module attrs on
# the lazily-imported robot stack; all-null under --dry-run off-workstation.
_METADATA_CONSTANT_NAMES: dict[str, tuple[str, ...]] = {
    "bf": (
        "_EE_TRANSLATION_FUDGE_FACTOR", "_EE_ROTATION_FUDGE_FACTOR",
        "OSC_BASE_KP", "_KP_GAIN_BASE", "_KD_GAIN_BASE",
        "EE_PD_KP", "EE_PD_KD", "JOINT_PD_KP", "JOINT_PD_KD",
    ),
    "osc": (
        "DEFAULT_KP", "DEFAULT_DAMPING_RATIO", "DEFAULT_NULLSPACE_KP",
        "DEFAULT_DLS_DAMPING", "DEFAULT_MAX_QDOT",
    ),
    "safety": (
        "JOINT_VELOCITY_MAX", "EE_LINEAR_VELOCITY_MAX", "EE_ANGULAR_VELOCITY_MAX",
        "WORKTABLE_HEIGHT", "WORKTABLE_DISTANCE_MIN", "WORKTABLE_MAX_DECEL",
        "CUSTOM_END_EFFECTOR_Z_EXTENSION",
    ),
    "fp": (
        "VELOCITY_COMMAND_DURATION_MS", "_JOINT_RELATIVE_DYNAMICS",
        "_EE_DELTA_RELATIVE_DYNAMICS", "_JOINT_STIFFNESS",
        "_TORQUE_THRESHOLD", "_FORCE_THRESHOLD",
    ),
}
_METADATA_MODULE_LABELS = {
    "bf": "bimanual_franka", "osc": "osc_velocity_controller",
    "safety": "safety", "fp": "franka_process",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_run_metadata(args: argparse.Namespace, episode_names: list[str],
                          stack: SimpleNamespace | None) -> dict:
    """Everything needed to interpret a run directory later, gathered up front."""
    constants = {
        _METADATA_MODULE_LABELS[key]: {
            name: getattr(getattr(stack, key, None), name, None) for name in names
        }
        for key, names in _METADATA_CONSTANT_NAMES.items()
    }
    kp_gain = 10.0 ** args.kp
    osc_base_kp = constants["bimanual_franka"]["OSC_BASE_KP"]
    input_file = args.traj_file if args.mode == "replay" else args.track_spec
    return {
        "status": "running",
        "mode": args.mode,
        "quat_encoding": "exact",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "args": vars(args),
        "input_file": str(Path(input_file).resolve()),
        "input_file_sha256": _sha256(input_file),
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
        description="Drive the Franka (open-loop replay or closed-loop reference tracking) and record the response."
    )
    parser.add_argument("traj_file", nargs="?", default=None,
                        help="Input HDF5 trajectory file to replay (replay mode)")
    parser.add_argument("--mode", choices=("replay", "track"), default="replay",
                        help="replay: open-loop sim-action replay; track: closed-loop "
                             "reference tracking from --track-spec (see module docstring)")
    parser.add_argument("--track-spec", default=None,
                        help="JSON spec of reference tracks (track mode)")
    parser.add_argument("--hold-s", type=float, default=0.0,
                        help="Track mode: prepend a constant-reference HOLD episode of "
                             "this duration (static-offset calibration)")
    parser.add_argument("--track-abort-m", type=float, default=0.15,
                        help="Track mode: abort an episode if position tracking error exceeds this (m)")
    parser.add_argument("--flush-every", type=int, default=100,
                        help="Steps between atomic mid-episode HDF5 flushes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run against a kinematic mock instead of the robot "
                             "(no hardware or lerobot/franky needed)")
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

    if args.mode == "replay" and not args.traj_file:
        parser.error("replay mode requires a traj_file")
    if args.mode == "track" and not args.track_spec:
        parser.error("track mode requires --track-spec")

    stack = _robot_stack(allow_missing=args.dry_run)
    if stack is None:
        logger.warning("robot stack unavailable — dry run against defaults")
    bf = getattr(stack, "bf", None)
    trans_fudge = float(getattr(bf, "_EE_TRANSLATION_FUDGE_FACTOR", 1.2))

    # Run directory: <out_root>/<timestamp>_<tag>. The tag defaults to the
    # input's parent directory name (replay: datasets live one per directory,
    # so it carries the sim condition, e.g. kp_actn0.50_damp_actn0.50) or the
    # spec stem (track).
    if args.mode == "replay":
        tag = args.tag or Path(args.traj_file).resolve().parent.name
        with h5py.File(args.traj_file, "r") as f:
            episode_names = list(f[list(f.keys())[0]].keys())
        track_init_qpos, track_episodes = None, []
    else:
        tag = args.tag or Path(args.track_spec).resolve().stem
        track_init_qpos, track_episodes = _load_track_spec(
            args.track_spec, args.fps, args.hold_s
        )
        episode_names = [name for name, _ in track_episodes]
    out_root = Path(args.out_root).expanduser()
    run_dir = out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"

    meta = _collect_run_metadata(args, episode_names, stack)
    _write_run_json(run_dir, meta)
    logger.info("run directory: %s", run_dir)

    # Per-episode HDF5 attrs: enough to interpret a file separated from run.json.
    base_attrs = {
        "input_file": meta["input_file"],
        "mode": args.mode,
        "quat_encoding": "exact",
        "kp": args.kp,
        "kd": args.kd,
        "fps": args.fps,
        "gripper_norm": args.gripper_norm,
        "kp_gain": 10.0 ** args.kp,
        "ee_translation_fudge_factor": trans_fudge,
        "ee_rotation_fudge_factor": getattr(bf, "_EE_ROTATION_FUDGE_FACTOR", None),
        "osc_base_kp": getattr(bf, "OSC_BASE_KP", None),
        "max_qdot": getattr(getattr(stack, "osc", None), "DEFAULT_MAX_QDOT", None),
        "dry_run": bool(args.dry_run),
    }

    controller = None
    all_errors: list[dict] = []
    episode_pairs: list[tuple[str, dict, dict]] = []
    try:
        # Connect robot (or mock)
        if args.dry_run:
            logger.info("dry run: using kinematic mock controller")
            controller = _MockController(trans_fudge=trans_fudge)
        else:
            logger.info("connecting to robot...")
            controller = stack.start_controller()
            logger.info("robot connected")

        for i in range(0, len(episode_names)):
            # Load the episode: replay parses the sim file, track uses the spec.
            if args.mode == "replay":
                logger.info("loading trajectory from %s", args.traj_file)
                name, traj = parse_traj(args.traj_file, i)
                ref = None
                n_steps = len(traj["eef_pos"])
                home_q = (
                    traj["qpos"][0].astype(np.float64) if "qpos" in traj else None
                )
            else:
                name, ref = track_episodes[i]
                traj = None
                n_steps = len(ref["pos_offsets"])
                home_q = track_init_qpos
            output = "record_" + name
            logger.info("recording episode %s (%d steps, mode=%s)", name, n_steps, args.mode)

            if home_q is not None:
                logger.info("homing to init qpos: %s", np.round(home_q, 3))
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
            else:
                logger.warning("no init qpos for episode %s; skipping pose-specific homing", name)

            episode_attrs = {
                **base_attrs,
                "reference_episode": name,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            out_path = str(run_dir / f"{i}_{output}.hdf5")
            logger.info(
                "running %d steps at %.1f Hz (kp=%.2f → gain=%.2f) — press right-arrow to stop early",
                n_steps, args.fps, args.kp, 10.0 ** args.kp,
            )
            recorded = _run_episode(
                controller=controller,
                traj=traj,
                fps=args.fps,
                kp=args.kp,
                kd=args.kd,
                gripper_norm=args.gripper_norm,
                video_dir=run_dir,
                video_stem=f"{i}_{output}",
                ref=ref,
                trans_fudge=trans_fudge,
                flush_path=out_path,
                flush_attrs=episode_attrs,
                flush_every=max(int(args.flush_every), 1),
                track_abort_m=args.track_abort_m,
            )

            if not recorded:
                logger.warning("no steps recorded; stopping sweep")
                break

            # Save HDF5 (final atomic write over any mid-episode flush)
            save_sysid_hdf5(recorded, out_path, attrs=episode_attrs)

            # Visualization + error stats (replay mode only: they compare
            # against the sim reference trajectory; track-mode analysis lives
            # in multi-fast's fit pipeline, which consumes the logged goals).
            if args.mode == "replay":
                viz_out = str(run_dir / f"{i}_{output}.html")
                save_comparison_html(traj, recorded, viz_out, fps=args.fps, frame_stride=args.viz_stride)
                all_errors.append(compute_trajectory_errors(traj, recorded, name=output))
                save_errors_json(all_errors, str(run_dir / "errors.json"))
                episode_pairs.append((name, traj, recorded))
            meta["episodes_completed"].append(name)

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
