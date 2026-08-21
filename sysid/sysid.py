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

Replay input can also be a LeRobotDataset (see --lerobot-repo-id below). Such
a dataset observes joints only, so its reference eef_pos/eef_quat -- the EE
path and rotation series every comparison plot and error stat is measured
against -- are forward kinematics of the reference qpos.

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
    tau_cmd       (T, 7) – joint torque the controller actually wrote, post
                           clamp and rate limit (matches sim's tau_cmd)
    tau_measured  (T, 7) – link-side measured joint torque (state.tau_J)
    tau_ext       (T, 7) – libfranka's estimated external torque, i.e. what the
                           dynamic model cannot account for. Commanded high but
                           measured low with the joint stationary means the
                           command is being absorbed mechanically, not a
                           controller fault -- the distinction sim cannot show.

Episodes are flushed incrementally (atomic tmp+rename, --flush-every steps)
so a crash or Ctrl-C mid-episode keeps the data collected so far.
A comparison HTML visualization is written alongside the HDF5 (replay mode).
One MP4 video per camera is written alongside the HDF5 (e.g. <stem>_cam_3_wrist.mp4).

Replay mode additionally writes a single aggregated ``aggregate_sim_format.hdf5``
alongside the per-episode files. For sim-HDF5 input it is structured EXACTLY
like the input file (``f[group_key][episode_key][field]`` → (T, D), same
group key and episode keys as the source). For LeRobotDataset input, which
has no single source group key to mirror, the group key is fixed to
``"data"`` instead (the ``aggregate_sim_format_group_key`` run.json field
records which). Its ``action`` field is a verbatim copy of the replayed
episode's ``action_norm`` — the actual normalized action the controller
received each tick, not the realized metric displacement — so the file can
be fed back into sim as a real-data sweep for further sysid.
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
import franka_config as fc
import h5py
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "residual_wrapper"))

from types import SimpleNamespace  # noqa: E402

from _viz import (  # noqa: E402
    compute_trajectory_errors,
    ee_path_from_qpos,
    save_aggregate_html,
    save_comparison_html,
    save_errors_json,
)

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
        from lerobot_robot_bimanual_franka import SingleArmFrankaConfig as cfg
        from lerobot_robot_bimanual_franka import (
            bimanual_franka as bf,
            franka_process as fp,
            osc_torque_controller as osc,
            safety as sf,
        )
    except ImportError:
        if allow_missing:
            return None
        raise
    _ROBOT_STACK = SimpleNamespace(
        start_controller=start_controller, bf=bf, fp=fp, osc=osc, safety=sf, cfg=cfg,
    )
    return _ROBOT_STACK

# Action kp/kd in [-1, 1]; send_action maps them via kp_gain = 10**kp.
# Default 0.0 → kp_gain = 1.0 (minimum, safest for an open-loop sysid replay).
_DEFAULT_KP = fc.policy("sysid.default_kp")
_DEFAULT_KD = fc.policy("sysid.default_kd")

# Rig profile the sysid loop drives, and the arm key it exposes.
_PROFILE = "single_arm_franka"
_ARM_KEY = fc.profile(_PROFILE).depth_center_arm

# Camera read timeout used when capturing frames inside the step loop.
_CAM_TIMEOUT_MS = fc.policy("sysid.camera_read_timeout_ms")


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
# LeRobotDataset trajectory source
# ---------------------------------------------------------------------------
#
# A LeRobotDataset episode is loaded and reshaped into exactly the dict shape
# parse_traj() returns from a sim HDF5 (an "action" (T,7) normalized delta
# array, plus "qpos" for homing and "eef_pos"/"eef_quat" for error
# visualisation), so _run_episode() and the rest of the replay path need no
# branching on the trajectory source.
#
# lerobot is only imported inside __init__ (like the robot stack, via
# _robot_stack) so --dry-run and HDF5-replay runs keep working on machines
# without lerobot installed. The whole sweep is fetched there in one go;
# __getitem__ then only slices the loaded table.

_EEF_POS_KEYS = ("observation.eef_pos", "eef_pos", "observation.ee_pos")
_EEF_QUAT_KEYS = ("observation.eef_quat", "eef_quat", "observation.ee_quat")


class LeRobotTrajSource:
    """Duck-types the (filename, index) parse_traj() interface over a
    LeRobotDataset so main()'s replay loop is agnostic to the input format.

    ``state_key``/``action_key`` select which dataset feature maps onto the
    sysid "qpos"/"action" fields; both default to the LeRobot convention
    ("observation.state", "action"). If the dataset's action is not already
    in the normalized [dx,dy,dz,rx,ry,rz,gripper] convention _run_episode
    expects (units of _POS_SCALE / _ROT_SCALE), pass --lerobot-action-scale
    to rescale it (see _load_lerobot_episode).

    The reference "eef_pos"/"eef_quat" come from an EE feature when the dataset
    carries one and from FK of the reference qpos otherwise -- see
    _viz.ee_path_from_qpos for the frame.
    """

    def __init__(self, repo_id: str, root: str | None, episodes: list[int] | None,
                state_key: str, action_key: str, action_is_normalized: bool):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError as e:
            raise ImportError(
                "LeRobotDataset replay requires the `lerobot` package "
                "(pip install lerobot)."
            ) from e
        self.repo_id = repo_id
        self.root = root
        self.state_key = state_key
        self.action_key = action_key
        self.action_is_normalized = action_is_normalized

        # Enumerate from meta/ alone: constructing a LeRobotDataset here would
        # fetch the frames too, before the episode filter is known.
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.episode_indices = (episodes if episodes is not None
                                else list(range(meta.total_episodes)))
        self.fps = float(meta.fps)

        # One download up front covering every episode of the sweep. Per-episode
        # construction re-ran snapshot_download between trajectories, and its
        # cache check counts the videos, so each one refetched mp4s that sysid
        # never reads (it replays with_cameras=False).
        logger.info("loading %d episode(s) of %s (videos skipped)",
                    len(self.episode_indices), repo_id)
        self._ds = LeRobotDataset(repo_id, root=root, episodes=self.episode_indices,
                                  download_videos=False)
        # Columns straight off the Arrow table: indexing the dataset per frame
        # decodes every camera stream to reach state/action (~58 ms/frame here).
        self._hf = self._ds.hf_dataset.with_format("numpy")
        self._episode_of_row = np.asarray(self._hf["episode_index"])

    def __len__(self) -> int:
        return len(self.episode_indices)

    def _episode_rows(self, ep_index: int) -> dict[str, np.ndarray]:
        """Every column of one episode, sliced out of the loaded table."""
        rows = np.flatnonzero(self._episode_of_row == ep_index)
        if len(rows) == 0:
            raise ValueError(f"episode {ep_index} of {self.repo_id!r} has no frames")
        return self._hf[int(rows[0]):int(rows[-1]) + 1]

    def __getitem__(self, index: int) -> tuple[str, dict[str, np.ndarray]]:
        ep_index = self.episode_indices[index]
        rows = self._episode_rows(ep_index)

        actions = np.asarray(rows[self.action_key], dtype=np.float64)
        n = len(actions)
        if actions.shape[1] < 7:
            # Pad a held-constant gripper column if the dataset action has no
            # gripper dim (e.g. 6-DoF pose-delta-only actions).
            pad = np.ones((n, 7 - actions.shape[1]), dtype=np.float64)
            actions = np.concatenate([actions, pad], axis=1)
        elif actions.shape[1] > 7:
            actions = actions[:, :7]

        if not self.action_is_normalized:
            actions = _normalize_lerobot_action(actions)

        state = np.asarray(rows[self.state_key], dtype=np.float64)
        # qpos (for homing) is the leading 7 state dims (or fewer, padded with
        # zeros) by convention.
        qpos = state[:, :7] if state.shape[1] >= 7 else np.pad(state, ((0, 0), (0, 7 - state.shape[1])))

        # Reference EE path and rotation series: a missing one silently blanks
        # the comparison plots and error stats rather than failing them.
        eef_pos = self._feature_series(rows, _EEF_POS_KEYS, 3)
        eef_quat = self._feature_series(rows, _EEF_QUAT_KEYS, 4)
        if eef_pos is None or eef_quat is None:
            if state.shape[1] >= 7:
                # This stack's datasets observe joints only; FK is the reference pose.
                fk_pos, fk_quat = ee_path_from_qpos(qpos)
                eef_pos = fk_pos if eef_pos is None else eef_pos
                eef_quat = fk_quat if eef_quat is None else eef_quat
            else:
                logger.warning(
                    "%s: state key %r has %d dims (<7 joints), so no reference EE "
                    "path or rotation error can be drawn for episode %d",
                    self.repo_id, self.state_key, state.shape[1], ep_index)
                eef_pos = np.zeros((n, 3)) if eef_pos is None else eef_pos

        name = f"episode_{ep_index:06d}"
        traj = {"action": actions.astype(np.float32), "qpos": qpos.astype(np.float32),
                "eef_pos": eef_pos.astype(np.float32)}
        if eef_quat is not None:
            traj["eef_quat"] = eef_quat.astype(np.float32)
        return name, traj

    @staticmethod
    def _feature_series(rows: dict, keys: tuple[str, ...], dim: int) -> np.ndarray | None:
        """First of ``keys`` the episode carries, as (T, dim); None if none."""
        for key in keys:
            if key in rows:
                arr = np.asarray(rows[key], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= dim:
                    return arr[:, :dim]
        return None


def _normalize_lerobot_action(actions: np.ndarray) -> np.ndarray:
    """Rescale a metric [dpos(3) m, drot(3) rad, gripper] action into the
    normalized [dpos/_POS_SCALE, drot/_ROT_SCALE, gripper] convention
    _run_episode expects in ``action_norm`` (see module docstring's
    _POS_SCALE/_ROT_SCALE). Only used when --lerobot-action-scale is passed,
    i.e. the dataset stores metric deltas rather than pre-normalized ones."""
    out = actions.copy()
    out[:, 0:3] = actions[:, 0:3] / 0.05   # _POS_SCALE
    out[:, 3:6] = actions[:, 3:6] / 0.5    # _ROT_SCALE
    return out


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
                return {_ARM_KEY: 0}

            def torque_snapshot(self, name):
                return (np.zeros(7), np.zeros(7), np.zeros(7))

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
        "tau_measured", "tau_ext",
    )}

    record_video = video_dir is not None and bool(controller.cameras)
    cam_frames: dict[str, list[np.ndarray]] = {n: [] for n in controller.cameras} if record_video else {}

    dt = 1.0 / fps
    t_start = time.perf_counter()
    start_pos = start_quat = None  # track-mode anchor, set on the first tick
    stop_reason = None  # None = ran to completion; else "early_stop" / "track_abort"
    # Keyboard early-stop only when stdin is a real terminal (dry runs under
    # pipes/CI and nohup'd sessions have no tty to put in raw mode).
    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        # cbreak, not raw: keypresses arrive unbuffered for the early-stop
        # poll, but native Ctrl-C (SIGINT even mid-sleep / mid-RPC) and newline
        # output processing are preserved, so the progress bar and any log
        # lines render cleanly.
        tty.setcbreak(sys.stdin)
    bar = tqdm(total=n_steps, desc=video_stem, unit="step", dynamic_ncols=True)
    try:
        for step in range(n_steps):
            t_step = time.perf_counter()

            if interactive and _stdin_key_pressed():
                key = _read_key()
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "right":
                    bar.write("early stop requested")
                    stop_reason = "early_stop"
                    break

            # --- read kinematic state ----------------------------------------
            # Store in _cached_kin_state so send_action re-uses it, avoiding
            # a redundant RPyC round-trip.
            kin = controller.robot_manager.current_kinematic_state_batch([_ARM_KEY])
            controller._cached_kin_state = kin
            q, dq, _jac, ee_pos, ee_quat, ee_vel = kin[_ARM_KEY]
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
                    stop_reason = "track_abort"
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
            buf["fault_count"].append(np.int32(controller.robot_manager.recovery_counts().get(_ARM_KEY, 0)))
            buf["eef_ang_vel"].append(np.asarray(ee_vel[3:], dtype=np.float32))
            buf["eef_lin_vel"].append(np.asarray(ee_vel[:3], dtype=np.float32))
            buf["eef_pos"].append(np.asarray(ee_pos, dtype=np.float32))
            buf["eef_quat"].append(np.asarray(ee_quat, dtype=np.float32))
            buf["qpos"].append(np.asarray(q, dtype=np.float32))
            buf["qvel"].append(np.asarray(dq, dtype=np.float32))
            buf["t_sim"].append(np.array([t_now], dtype=np.float32))
            # Torques come from the same state read as q/dq above, so they are
            # from one tick, not stitched across two.
            tau_cmd, tau_meas, tau_ext = controller.robot_manager.torque_snapshot("r")
            buf["tau_cmd"].append(np.asarray(tau_cmd, dtype=np.float32))
            buf["tau_measured"].append(np.asarray(tau_meas, dtype=np.float32))
            buf["tau_ext"].append(np.asarray(tau_ext, dtype=np.float32))

            # --- incremental flush (atomic tmp + rename) ----------------------
            if flush_path is not None and (step + 1) % flush_every == 0:
                save_sysid_hdf5(
                    {k: np.stack(v) for k, v in buf.items() if v},
                    flush_path, attrs=flush_attrs, quiet=True,
                )

            if track:
                bar.set_postfix_str(f"err={float(np.linalg.norm(pos_err)) * 1000:.1f}mm", refresh=False)
            bar.update(1)

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
        bar.close()
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    if record_video and cam_frames:
        _save_videos(cam_frames, video_dir, video_stem, fps)

    return {k: np.stack(v) for k, v in buf.items() if v}, stop_reason


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


def save_sim_format_hdf5(
    episode_pairs: list[tuple[str, dict[str, np.ndarray], dict]],
    path: str,
    group_key: str,
) -> None:
    """Write an aggregated HDF5 in the EXACT sim layout (group -> episode ->
    field), populated with REAL recorded data, so the file is a drop-in
    replacement for the sim-generated input file and can be fed back into sim
    for further sysid.

    Structure mirrors the input sim file precisely: f[group_key][episode_key][field],
    with the same episode keys the sim file used. For LeRobotDataset input the
    caller passes a synthetic group_key ("data") since there is no single
    source group key to mirror; episode keys still match what was replayed.

    Replay mode ONLY. The ``action`` field written here is a verbatim copy of
    each episode's recorded ``action_norm`` — the actual normalized action
    the controller received that tick (read straight from the sim input file,
    since replay is open-loop) — NOT the realized metric displacement that
    the flat per-episode files store under ``action``. This is deliberate:
    the aggregate's ``action`` must encode-match the sim file's ``action``
    field exactly, unit for unit.

    All other fields (eef_pos, eef_quat, qpos, qvel, eef_goal_pos, ...) are
    the real recorded robot state/response for that tick, carried through
    unchanged.

    episode_pairs: list of (episode_name, recorded_dict, episode_attrs).
        recorded_dict is exactly what _run_episode returns for a replay-mode
        episode (must contain "action_norm"). episode_attrs are stamped onto
        each episode group (mode, stop_reason, timestamps, gains, etc.) so
        the file is self-describing even if separated from run.json.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        grp = f.create_group(group_key)
        for name, recorded, attrs in episode_pairs:
            if "action_norm" not in recorded:
                raise ValueError(
                    f"episode {name!r} has no action_norm; save_sim_format_hdf5 "
                    "is replay-mode only"
                )
            ep = grp.create_group(name)
            for field, arr in recorded.items():
                if field in ("action", "action_norm"):
                    continue  # both folded into the explicit "action" (=action_norm) write below
                ep.create_dataset(field, data=arr, compression="gzip", compression_opts=4)
            ep.create_dataset("action", data=recorded["action_norm"],
                            compression="gzip", compression_opts=4)

            for key, val in (attrs or {}).items():
                if val is not None:
                    ep.attrs[key] = val
    os.replace(tmp, path)
    logger.info("saved aggregate sim-format dataset (%d episodes) to %s",
                len(episode_pairs), path)


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
        "OSC_BASE_KP", "OSC_BASE_DAMPING_RATIO", "_KP_GAIN_BASE", "_KD_GAIN_BASE",
        "JOINT_IMPEDANCE_KP", "HOME_IMPEDANCE_KP", "HOME_MAX_QDOT",
    ),
    "osc": (
        "IMPEDANCE_MODE",
        "DEFAULT_KP", "KP_LIMITS", "DEFAULT_DAMPING_RATIO", "DAMPING_RATIO_LIMITS",
        "KP_EXP_SCALE", "DAMPING_EXP_SCALE", "DEFAULT_NULLSPACE_KP",
        "DELTA_POS_MAX", "DELTA_ROT_MAX", "DEFAULT_JOINT_KP", "JOINT_TORQUE_LIMITS",
    ),
    "safety": (
        "WORKTABLE_HEIGHT", "WORKTABLE_DISTANCE_MIN", "EE_SPHERE",
    ),
    "fp": (
        "NUM_JOINTS", "RPYC_TIMEOUT_S", "FIRST_STATE_TIMEOUT_S",
    ),
}
_METADATA_MODULE_LABELS = {
    "bf": "bimanual_franka", "osc": "osc_torque_controller",
    "safety": "safety", "fp": "franka_process",
}

# The NUC-side law cannot be imported here (no pylibfranka), so pin the run to a
# controller revision by hashing the files the deploy script ships.
_CONTROL_STACK_FILES = ("pylibfranka_control.py", "osc_torque_controller.py",
                        "franka_jacobian.py")


def _control_stack_hashes(stack: SimpleNamespace | None) -> dict:
    osc_mod = getattr(stack, "osc", None)
    if osc_mod is None or not getattr(osc_mod, "__file__", None):
        return {}
    pkg = Path(osc_mod.__file__).resolve().parent
    out = {}
    for name in _CONTROL_STACK_FILES:
        try:
            out[name] = _sha256(str(pkg / name))
        except OSError:
            out[name] = None
    return out


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
    # env_wrapper passes no config overrides, so control.yaml's `tuning:` block
    # IS the plant; runs recorded under different values are not comparable.
    # Read from the config rather than scraped off the dataclass: every field
    # there is a default_factory, so the class carries no readable attribute.
    constants["control.yaml tuning"] = dict(fc.control("tuning"))
    kp_gain = 10.0 ** args.kp
    osc_base_kp = constants["bimanual_franka"]["OSC_BASE_KP"]
    if args.mode == "replay":
        input_file = args.lerobot_repo_id if args.lerobot_repo_id else args.traj_file
    else:
        input_file = args.track_spec
    # LeRobotDataset inputs are identified by repo_id (+ optional local root),
    # not a single local file, so they have no sha256 to pin the run to.
    input_file_sha256 = None
    input_file_resolved = input_file
    if args.mode != "replay" or not args.lerobot_repo_id:
        input_file_resolved = str(Path(input_file).resolve())
        input_file_sha256 = _sha256(input_file)
    return {
        "status": "running",
        "mode": args.mode,
        "quat_encoding": "exact",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "args": vars(args),
        "input_file": input_file_resolved,
        "input_file_sha256": input_file_sha256,
        "episodes": episode_names,
        "episodes_completed": [],
        "derived_gains": {
            "kp_gain": kp_gain,
            "effective_kp": kp_gain * osc_base_kp if osc_base_kp is not None else None,
            "kd_note": "kd action is the OSC damping_ratio: ratio = 10**kd, "
                       "kd = 2*sqrt(kp)*ratio (robosuite impedance_mode='variable')",
        },
        "constants": constants,
        # Full YAML config snapshot: the constants above are read live from the
        # modules, but the modules now read from config/, so record the source.
        "config": fc.all_sections(),
        "config_dir": str(fc.config_dir()),
        "control_stack_sha256": _control_stack_hashes(stack),
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
                        help="Input HDF5 trajectory file to replay (replay mode). "
                             "Omit and use --lerobot-repo-id instead to replay a LeRobotDataset.")
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
    parser.add_argument("--fps", type=float, default=float(fc.control_fps()), help="Control rate in Hz")
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
    parser.add_argument("--lerobot-repo-id", default=None,
                        help="Replay mode: repo_id (or local dataset name) of a LeRobotDataset "
                             "to replay instead of --traj_file, e.g. 'lerobot/aloha_sim_insertion'.")
    parser.add_argument("--lerobot-root", default=None,
                        help="Local root directory of the LeRobotDataset, if not on the Hub cache "
                             "(passed through to LeRobotDataset(root=...)).")
    parser.add_argument("--lerobot-episodes", default=None,
                        help="Comma-separated episode indices to replay, e.g. '0,2,5'. "
                             "Defaults to all episodes in the dataset.")
    parser.add_argument("--lerobot-state-key", default="observation.state",
                        help="LeRobotDataset feature key used for qpos/homing.")
    parser.add_argument("--lerobot-action-key", default="action",
                        help="LeRobotDataset feature key used for the replayed action.")
    parser.add_argument("--lerobot-action-scale", action="store_true",
                        help="Set if the LeRobotDataset action is metric (m, rad) rather than "
                             "already normalized to [-1, 1] in units of _POS_SCALE/_ROT_SCALE; "
                             "the loader will rescale it into the normalized convention "
                             "_run_episode expects.")
    args = parser.parse_args()

    # INFO: the only DEBUG emitters in this pipeline are the camera drivers
    # (frame-drain notices); everything sysid relies on is INFO/WARNING.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    if args.mode == "replay" and not (args.traj_file or args.lerobot_repo_id):
        parser.error("replay mode requires a traj_file or --lerobot-repo-id")
    if args.mode == "replay" and args.traj_file and args.lerobot_repo_id:
        parser.error("pass either traj_file or --lerobot-repo-id, not both")
    if args.mode == "track" and not args.track_spec:
        parser.error("track mode requires --track-spec")

    stack = _robot_stack(allow_missing=args.dry_run)
    if stack is None:
        logger.warning("robot stack unavailable — dry run against defaults")
    bf = getattr(stack, "bf", None)
    trans_fudge = float(getattr(bf, "_EE_TRANSLATION_FUDGE_FACTOR", 1.2))

    # Run directory: <out_root>/<timestamp>_<tag>. The tag defaults to the
    # input's parent directory name (replay from HDF5: datasets live one per
    # directory, so it carries the sim condition, e.g.
    # kp_actn0.50_damp_actn0.50; replay from LeRobotDataset: the repo_id's
    # final path component) or the spec stem (track).
    lerobot_traj_source = None
    sim_group_key = None  # only set (and aggregate written) for HDF5 replay input
    if args.mode == "replay" and args.lerobot_repo_id:
        episode_filter = None
        if args.lerobot_episodes:
            episode_filter = [int(x) for x in args.lerobot_episodes.split(",") if x.strip()]
        lerobot_traj_source = LeRobotTrajSource(
            args.lerobot_repo_id, args.lerobot_root, episode_filter,
            args.lerobot_state_key, args.lerobot_action_key,
            action_is_normalized=not args.lerobot_action_scale,
        )
        tag = args.tag or args.lerobot_repo_id.rstrip("/").split("/")[-1]
        episode_names = [f"episode_{i:06d}" for i in lerobot_traj_source.episode_indices]
        track_init_qpos, track_episodes = None, []
        # LeRobotDataset input has no single sim group key to mirror (unlike
        # HDF5 sim input), so use the same "data" key save_sysid_hdf5 already
        # writes per-episode files under, and note it wasn't the source's key.
        sim_group_key = "data"
    elif args.mode == "replay":
        tag = args.tag or Path(args.traj_file).resolve().parent.name
        with h5py.File(args.traj_file, "r") as f:
            sim_group_key = list(f.keys())[0]
            episode_names = list(f[sim_group_key].keys())
        track_init_qpos, track_episodes = None, []
    else:
        tag = args.tag or Path(args.track_spec).resolve().stem
        track_init_qpos, track_episodes = _load_track_spec(
            args.track_spec, args.fps, args.hold_s
        )
        episode_names = [name for name, _ in track_episodes]
        sim_group_key = None  # track mode never writes the sim-format aggregate
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
    sim_format_pairs: list[tuple[str, dict, dict]] = []
    try:
        # Connect robot (or mock)
        if args.dry_run:
            logger.info("dry run: using kinematic mock controller")
            controller = _MockController(trans_fudge=trans_fudge)
        else:
            logger.info("connecting to robot...")
            # No cameras: sysid consumes kinematics only, and skipping the rig
            # drops the per-tick reads, GigE traffic, and MP4 writes.
            controller = stack.start_controller(with_cameras=False)
            logger.info("robot connected")

        for i in range(0, len(episode_names)):
            # Load the episode: replay parses the sim file or a LeRobotDataset
            # episode, track uses the spec.
            if args.mode == "replay" and lerobot_traj_source is not None:
                logger.info("loading episode %d/%d from LeRobotDataset %s",
                           i + 1, len(lerobot_traj_source), args.lerobot_repo_id)
                name, traj = lerobot_traj_source[i]
                ref = None
                n_steps = len(traj["action"])
                home_q = traj["qpos"][0].astype(np.float64) if "qpos" in traj else None
            elif args.mode == "replay":
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
            recorded, stop_reason = _run_episode(
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

            # Per-episode sanity summary — makes run.json sufficient for the
            # post-run check (step counts, faults, aborts, tracking lag).
            n_rec = len(next(iter(recorded.values())))
            summary = {
                "episode": name,
                "steps": n_rec,
                "expected_steps": n_steps,
                "stop_reason": stop_reason,
                "max_fault_count": int(recorded["fault_count"].max()),
            }
            if "eef_goal_pos" in recorded:
                lag = np.linalg.norm(
                    recorded["eef_goal_pos"] - recorded["eef_pos"], axis=1
                )
                summary["lag_mm"] = {
                    "mean": round(float(lag.mean() * 1e3), 2),
                    "max": round(float(lag.max() * 1e3), 2),
                }
            if "eef_goal_quat" in recorded:
                gq = recorded["eef_goal_quat"].astype(np.float64)
                mq = recorded["eef_quat"].astype(np.float64)
                gq /= np.maximum(np.linalg.norm(gq, axis=1, keepdims=True), 1e-12)
                mq /= np.maximum(np.linalg.norm(mq, axis=1, keepdims=True), 1e-12)
                ang = 2.0 * np.arccos(np.clip(np.abs((gq * mq).sum(axis=1)), -1.0, 1.0))
                summary["lag_deg"] = {
                    "mean": round(float(np.degrees(ang.mean())), 2),
                    "max": round(float(np.degrees(ang.max())), 2),
                }
            ok = (stop_reason is None and n_rec == n_steps
                  and summary["max_fault_count"] == 0)
            summary["ok"] = ok
            meta.setdefault("episode_summaries", []).append(summary)
            _write_run_json(run_dir, meta)  # keep run.json current per episode
            log = logger.info if ok else logger.warning
            lag_str = ""
            if "lag_mm" in summary:
                lag_str += ", lag mean %.1f / max %.1f mm" % (
                    summary["lag_mm"]["mean"], summary["lag_mm"]["max"])
            if "lag_deg" in summary:
                lag_str += ", ori lag mean %.1f / max %.1f deg" % (
                    summary["lag_deg"]["mean"], summary["lag_deg"]["max"])
            log("episode %s: %d/%d steps, faults=%d, stop=%s%s",
                name, n_rec, n_steps, summary["max_fault_count"], stop_reason, lag_str)

            # Visualization + error stats (replay mode only: they compare
            # against the sim/LeRobot reference trajectory; track-mode
            # analysis lives in multi-fast's fit pipeline, which consumes the
            # logged goals).
            if args.mode == "replay":
                viz_out = str(run_dir / f"{i}_{output}.html")
                save_comparison_html(traj, recorded, viz_out, fps=args.fps, frame_stride=args.viz_stride)
                errors = compute_trajectory_errors(traj, recorded, name=output)
                all_errors.append(errors)
                save_errors_json(all_errors, str(run_dir / "errors.json"))
                episode_pairs.append((name, traj, recorded))

                # Real-vs-reference error in run.json, so a broken replay shows
                # without opening the HTML.
                pos_e = errors["position_error_m"]
                rot_e = errors["rotation_error_rad"]
                summary["ref_error"] = {
                    "position_mm": {k: round(pos_e[k] * 1e3, 2) for k in ("mean", "max")},
                    "rotation_deg": (None if rot_e is None else
                                     {k: round(np.degrees(rot_e[k]), 2) for k in ("mean", "max")}),
                }
                _write_run_json(run_dir, meta)
                ref_str = "ref err pos mean %.1f / max %.1f mm" % (
                    summary["ref_error"]["position_mm"]["mean"],
                    summary["ref_error"]["position_mm"]["max"])
                if summary["ref_error"]["rotation_deg"] is not None:
                    ref_str += ", rot mean %.1f / max %.1f deg" % (
                        summary["ref_error"]["rotation_deg"]["mean"],
                        summary["ref_error"]["rotation_deg"]["max"])
                else:
                    ref_str += ", rot n/a (reference has no eef_quat)"
                logger.info("episode %s: %s", name, ref_str)

                # Aggregate sim-format accumulator (replay-from-HDF5 only: it
                # mirrors a single sim group key, which a LeRobotDataset input
                # has no equivalent of).
                if sim_group_key is not None:
                    sim_format_pairs.append((name, recorded, {
                        **episode_attrs,
                        "stop_reason": stop_reason,
                        "steps": n_rec,
                        "expected_steps": n_steps,
                    }))
            meta["episodes_completed"].append(name)

        if episode_pairs:
            try:
                save_aggregate_html(episode_pairs, str(run_dir / "aggregate.html"), fps=args.fps)
            except Exception:
                logger.exception("aggregate visualization failed")

        if sim_format_pairs:
            try:
                save_sim_format_hdf5(
                    sim_format_pairs,
                    str(run_dir / "aggregate_sim_format.hdf5"),
                    group_key=sim_group_key,
                )
            except Exception:
                logger.exception("aggregate sim-format HDF5 write failed")
        elif args.mode == "replay" and lerobot_traj_source is not None and episode_pairs:
            logger.info(
                "skipping aggregate_sim_format.hdf5: input was a LeRobotDataset "
                "(--lerobot-repo-id), which has no single sim group key to mirror"
            )

        # End-of-run verdict line (also derivable from run.json episode_summaries).
        summaries = meta.get("episode_summaries", [])
        bad = [s for s in summaries if not s["ok"]]
        if bad:
            logger.warning("RUN CHECK: %d/%d episodes flagged: %s",
                           len(bad), len(summaries), [s["episode"] for s in bad])
        else:
            logger.info("RUN CHECK: all %d episodes clean (full length, no faults, no aborts)",
                        len(summaries))

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