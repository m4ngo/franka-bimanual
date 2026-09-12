"""Shared machinery for the two baseline hardware rollouts.

Everything here is common to `sail_bridge/rollout.py` and
`bspline_bridge/rollout.py`: robot construction, the observation both backends
are built from, the action dict, the rate-varying goal dispatcher, and the
recording/metrics side. No torch and no upstream imports -- the policy lives in
another process (see zmq_client).

The structural reference is residual_wrapper/run_residual.py; the pieces marked
as copied from it are noted at their definitions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import franka_config as fc  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402

from lerobot_robot_bimanual_franka import ControlMode  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.lerobot_source import EE_KEYS  # noqa: E402

from baselines.zmq_client import PolicyTimeout  # noqa: E402

# The rig -> config-class table, and the exposed key prefix, have exactly one
# definition in the tree; importing it costs the lerobot import we take anyway.
from teleop_single_arm import _ARM_KEY as ARM_KEY, _RIGS as RIGS  # noqa: E402

logger = logging.getLogger("baselines.rollout")

NUM_JOINTS = fc.num_joints()

# EE_KEYS is the EE_POS action schema itself (lerobot_source), which is what
# arm_prefix() validates a recording against -- not a restatement of it. Keeping
# these identical to run_residual.py's is what makes a baseline rollout readable
# by the same tooling as a residual one.
EE_ACTION_KEYS = tuple(f"{ARM_KEY}_{ax}" for ax in EE_KEYS)
ACTION_KEYS = (*EE_ACTION_KEYS, "kp", "kd")
STATE_OBS_KEYS = (
    *(f"{ARM_KEY}_joint_{i}" for i in range(1, NUM_JOINTS + 1)),
    f"{ARM_KEY}_gripper",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def obs_fps() -> int:
    """Observation/inference rate; null in policy.yaml means control_fps()."""
    return int(fc.policy("baselines.exec.obs_fps") or fc.control_fps())


def origin_time_scale() -> float:
    """B-Spline knot-index -> seconds, i.e. the rate the demos were RECORDED at.

    The spline's knots count frames, and `t` advances at this many index units
    per second, so this must be the recording fps. null in policy.yaml means
    control_fps(). Upstream defaults it to 10.0, their own rate; left there
    against 20 Hz data `t` advances at half the needed rate and every plan plays
    at 0.5x -- twice the wall clock, which reads as a sluggish controller rather
    than as a misconfiguration.
    """
    return float(fc.policy("baselines.bspline.origin_time_scale") or fc.control_fps())


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

def build_robot(rig: str, control_mode: ControlMode, depth: bool = False):
    """Connectable robot for one rig profile, in an explicit control mode.

    `control_mode` is passed rather than inherited: both single-arm profiles
    declare EE_DELTA in config/rig.yaml, and B-Spline needs EE_POS.

    depth=False by default -- neither baseline consumes a point cloud, so the
    FRAMOS cloud crop is per-tick cost for nothing.
    """
    if rig not in RIGS:
        raise ValueError(f"unknown rig {rig!r}; choose from {sorted(RIGS)}")
    arm_key = next(iter(fc.profile(rig).arms))
    if arm_key != ARM_KEY:
        # The guard teleop_single_arm.py has and the record script does not.
        raise ValueError(
            f"rig {rig!r} exposes key {arm_key!r}, not {ARM_KEY!r}; the action "
            "keys and gripper wiring assume the latter."
        )
    # Banner: RIGHT_ARM_RIG_HANDOFF.md's rule is that a wrong rig name here means
    # the flag did not take and the OTHER arm is about to move.
    logger.info("rig %s -> physical arm %r, control mode %s",
                rig, fc.profile(rig).arms[arm_key], control_mode.value)
    return make_robot_from_config(
        RIGS[rig](control_mode=control_mode, depth=depth)
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Measured:
    """One observation, in the units the baseline datasets were built in."""
    q: np.ndarray            # (NUM_JOINTS,) rad
    pos: np.ndarray          # (3,) m, O_T_EE
    quat_xyzw: np.ndarray    # (4,) O_T_EE
    gripper: float           # normalised [0, 1]
    images: dict[str, np.ndarray]   # camera name -> HWC uint8


def measure(obs: dict) -> Measured:
    """Robot observation -> Measured.

    The pose is O_T_EE via `eef_poses_from_qpos`, which is what
    baselines/common.py built every training observation from. NOT
    residual_wrapper's `current_ee_pose`: that applies a different correction
    (grip-site position, hand-body orientation) for sim-trained students, and
    feeding it here would put both baselines off-distribution by 45 degrees.
    """
    q = np.array([obs[f"{ARM_KEY}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)],
                 dtype=np.float64)
    pos, quat = eef_poses_from_qpos(q[None])
    return Measured(
        q=q,
        pos=pos[0],
        quat_xyzw=quat[0],
        # Reported in the same normalised [0, 1] the converter stored, so there
        # is no unit conversion on this channel in either direction.
        gripper=float(obs[f"{ARM_KEY}_gripper"]),
        images={k: v for k, v in obs.items()
                if isinstance(v, np.ndarray) and v.ndim == 3},
    )


def _images(m: Measured, shapes: dict) -> dict:
    """Camera frames under the policy's own `<cam>_image` keys, resized.

    The converter names them `obs/<cam>_image`, so the policy key for camera
    `cam_2` is `cam_2_image`. Sizes come from the checkpoint (the meta
    handshake), because SAIL's HDF5 carries full-resolution frames while its
    config expects 84x84 -- nothing else in the stack would catch the mismatch.
    Both servers want HWC uint8 and do their own scaling and transpose.
    """
    out = {}
    for key, shape in shapes.items():
        if not key.endswith("_image"):
            continue
        cam = key[: -len("_image")]
        img = m.images.get(cam)
        if img is None:
            # Never silently omitted -- check_camera_coverage refuses the run up
            # front, before the arm has homed.
            raise KeyError(
                f"the checkpoint wants {key!r} but the rig has no camera {cam!r}"
            )
        h, w = int(shape[-2]), int(shape[-1])
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        out[key] = np.ascontiguousarray(img, dtype=np.uint8)
    return out


def check_camera_coverage(meta: dict, controller, allow_missing: bool = False) -> dict:
    """Refuse a checkpoint whose camera keys the rig cannot supply.

    The two single-arm profiles expose DIFFERENT cameras -- `single_arm_franka`
    has cam_1/cam_5/cam_2 and `single_arm_right` has cam_3/cam_4/cam_2 -- so a
    checkpoint trained on one rig names keys the other does not have. Left to the
    observation builder this surfaces as a KeyError inside the policy server
    after the arm has already homed, blaming the wrong layer.

    Returns the shape map to hand to sail_obs/bspline_obs.
    """
    shapes = dict(meta["obs_key_shapes"])
    wanted = {k[: -len("_image")] for k in shapes if k.endswith("_image")}
    have = set(controller.cameras)
    missing = sorted(wanted - have)
    if missing and not allow_missing:
        raise ValueError(
            f"checkpoint needs camera(s) {missing} that this rig does not have "
            f"(it has {sorted(have)}). The single-arm profiles expose different "
            "cameras, so a checkpoint trained on one rig cannot roll out on the "
            "other unedited -- pick the matching --rig, or pass "
            "--allow-missing-cameras to send blank frames for the missing ones "
            "and accept that the policy is off-distribution."
        )
    for cam in missing:
        logger.warning("no camera %r on this rig; sending blank frames", cam)
        shapes.pop(f"{cam}_image", None)
    unused = sorted(have - wanted)
    if unused:
        logger.info("cameras %s are recorded but not fed to the policy", unused)
    return shapes


def sail_obs(m: Measured, shapes: dict) -> dict:
    """The obs keys baselines/sail_bridge/dataset.py writes."""
    return {
        "robot0_eef_pos": m.pos.astype(np.float32),
        "robot0_eef_quat": m.quat_xyzw.astype(np.float32),
        "robot0_joint_pos": m.q.astype(np.float32),
        "robot0_gripper_qpos": np.array([m.gripper], dtype=np.float32),
        **_images(m, shapes),
    }


def bspline_obs(m: Measured, shapes: dict) -> dict:
    """The obs keys baselines/bspline_bridge/dataset.py writes."""
    return {
        "arm_pos": m.pos.astype(np.float32),
        "arm_quat": m.quat_xyzw.astype(np.float32),
        "gripper_pos": np.array([m.gripper], dtype=np.float32),
        **_images(m, shapes),
    }


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

def _gains() -> dict:
    """kp/kd action channels.

    Normalised zero, which the exponential remap turns into
    torque.osc.default_kp -- neither baseline predicts gains.
    """
    return {"kp": float(fc.policy("sysid.default_kp")),
            "kd": float(fc.policy("sysid.default_kd"))}


def ee_pos_action(pos, quat_xyzw, gripper: float) -> dict:
    """Absolute OSC goal pose. `{arm}_gripper` is absolute in [0, 1]."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    vals = (*np.asarray(pos, dtype=np.float64), *q, float(gripper))
    return {**{k: float(v) for k, v in zip(EE_ACTION_KEYS, vals)}, **_gains()}


def ee_delta_action(dpos, drotvec, gripper: float) -> dict:
    """Per-step delta. Metres and a delta QUATERNION, which is what send_action
    reads -- a normalised value passed through unconverted reads as metres, gets
    clipped to torque.delta.pos_max_m, and looks like a tracking problem."""
    dq = Rotation.from_rotvec(np.asarray(drotvec, dtype=np.float64)).as_quat()
    vals = (*np.asarray(dpos, dtype=np.float64), *dq, float(gripper))
    return {**{k: float(v) for k, v in zip(EE_ACTION_KEYS, vals)}, **_gains()}


# ---------------------------------------------------------------------------
# Ported upstream helpers (numpy/scipy only)
# ---------------------------------------------------------------------------

def slowdown_mode(prev_acts, cur_act, future_acts, window: int) -> bool:
    """Port of SAIL/utils/dev_utils.py:get_slowdown_mode_from_model.

    The precision label is the LAST action column. A window of `window` centred
    on the current step; any label over 0.5 means execute this step slowly.
    """
    cur_act = np.asarray(cur_act)
    if window == 1:
        return bool(cur_act[-1] > 0.5)
    future_acts = np.asarray(future_acts)
    segment = window // 2
    left_n = min(len(prev_acts), segment)
    right_n = min(future_acts.shape[0], segment)
    left = np.asarray(prev_acts[-left_n:]) if left_n else np.empty((0,))
    left = left[..., -1] if left.shape[0] > 0 else np.zeros(1)
    return bool(np.any(np.concatenate(
        [left, cur_act[np.newaxis, -1], future_acts[:right_n, -1]]
    ) > 0.5))


def tracking_error_low(meas_pos, meas_quat_xyzw, desired_pos, desired_rotvec,
                       pos_teb: float, ori_teb: float) -> bool:
    """Port of run_trained_agent_receding_horizon.py:check_if_tracking_error_low.

    Upstream reads `controller.ee_pos` / `.ee_ori_mat` off robosuite's OSC
    object; the measured O_T_EE stands in for both. The error definitions are
    upstream's: inf-norm on position, arccos((trace-1)/2) on orientation.
    """
    pos_err = float(np.linalg.norm(
        np.asarray(meas_pos, dtype=np.float64) - np.asarray(desired_pos, dtype=np.float64),
        ord=np.inf,
    ))
    real_R = Rotation.from_quat(np.asarray(meas_quat_xyzw, dtype=np.float64)).as_matrix()
    desired_R = Rotation.from_rotvec(np.asarray(desired_rotvec, dtype=np.float64)).as_matrix()
    trace = float(np.clip(np.trace(desired_R.T @ real_R), -1.0, 3.0))
    ori_err = float(np.arccos((trace - 1.0) / 2.0))
    return pos_err < pos_teb and ori_err < ori_teb


def rot6d_to_quat_xyzw(rot6d) -> np.ndarray:
    """Port of policy_local_utils.py's rotation_6d_to_matrix + rot6d_to_quat_xyzw.

    Gram-Schmidt on the two 3-vectors, then a sign convention on w so successive
    samples of a spline do not flip hemisphere between ticks.
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    b1 = r[:3] / max(float(np.linalg.norm(r[:3])), 1e-12)
    b2 = r[3:] - float(np.dot(b1, r[3:])) * b1
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-12)
    mat = np.stack((b1, b2, np.cross(b1, b2)), axis=-2)
    quat = Rotation.from_matrix(mat).as_quat()
    if quat[3] < 0.0:
        np.negative(quat, out=quat)
    return quat


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class LeadExceeded(RuntimeError):
    """The commanded goal has run too far ahead of the arm."""


class Dispatcher:
    """Pushes OSC goals on an absolute-deadline clock at a per-step rate.

    The rate is an argument to `send`, not to the constructor: SAIL's precision
    modulation changes it step by step. Sleeping to an absolute clock (the
    pattern in run_residual.py) lets the idle slack of ordinary steps absorb the
    few ms an inference step overruns, so the loop averages its target instead of
    accumulating per-step deficits.
    """

    def __init__(self, controller, dry_run: bool = False) -> None:
        self.controller = controller
        self.dry_run = dry_run
        self.steps = 0
        self.last_action: dict | None = None
        self._deadline = time.perf_counter()
        self._prev_send = 0.0
        self._gaps: list[float] = []
        self._window_start = time.perf_counter()
        self._window_steps = 0

    def start(self) -> None:
        self._deadline = time.perf_counter()
        self._prev_send = 0.0
        self._window_start = time.perf_counter()
        self._window_steps = 0
        self._gaps.clear()

    def send(self, action: dict, hz: float) -> None:
        t_send = time.perf_counter()
        if not self.dry_run:
            self.controller.send_action(action)
        self.last_action = action
        self.steps += 1
        self._window_steps += 1
        if self._prev_send:
            self._gaps.append((t_send - self._prev_send) * 1000.0)
        self._prev_send = t_send

        self._log_window()
        # Ordering is run_residual.py's: sleep to THIS step's deadline, then
        # advance. Incrementing first instead delays the FIRST goal by a whole
        # period (measured 100 ms at 10 Hz) and makes the resync below land on
        # `now` rather than `now + dt`, double-counting a period after a stall.
        # Sub-period jitter is absorbed either way.
        dt = 1.0 / float(hz)
        sleep_s = self._deadline - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        self._deadline += dt
        if self._deadline < time.perf_counter():
            # Only a stall longer than one period lands here. After a large one
            # (episode-start warmup, an operator pause) resync rather than racing
            # to repay an unpayable debt.
            self._deadline = time.perf_counter() + dt

    def _log_window(self) -> None:
        now = time.perf_counter()
        span = now - self._window_start
        if span < 1.0 or not self._gaps:
            return
        # send-gap max is the number that matters: it is how long the OSC loop
        # sat on one goal, and a spike there is the visible hitch.
        logger.info("exec %.1f Hz over %.1fs  send-gap avg/max %.1f/%.1f ms",
                    self._window_steps / span, span,
                    sum(self._gaps) / len(self._gaps), max(self._gaps))
        self._window_start = now
        self._window_steps = 0
        self._gaps.clear()


def lead(goal_pos, goal_quat_xyzw, m: Measured) -> tuple[float, float]:
    """(position, orientation) divergence of the last absolute goal from the arm."""
    pos_err = float(np.linalg.norm(
        np.asarray(goal_pos, dtype=np.float64) - m.pos.astype(np.float64)))
    rel = (Rotation.from_quat(np.asarray(goal_quat_xyzw, dtype=np.float64))
           * Rotation.from_quat(m.quat_xyzw.astype(np.float64)).inv())
    return pos_err, float(np.linalg.norm(rel.as_rotvec()))


def check_lead(goal_pos, goal_quat_xyzw, m: Measured) -> tuple[float, float]:
    """Abort the episode when an absolute goal has outrun the arm.

    An ABORT, never a clamp. EE_POS has no delta envelope and that is faithful to
    osc.py; clamping the goal here would be the third limit layer CLAUDE.md
    forbids, and it would silently eat exactly the lead a sped-up plan is
    supposed to have. Checked at observation ticks only -- a 15 cm divergence
    does not appear and vanish inside one 50 ms period.

    Meaningless in EE_DELTA, where the goal is rebuilt from the measured pose
    every step and the lead is structurally one clipped delta.
    """
    pos_err, rot_err = lead(goal_pos, goal_quat_xyzw, m)
    max_m = float(fc.policy("baselines.exec.max_lead_m"))
    max_rad = float(fc.policy("baselines.exec.max_lead_rad"))
    if pos_err > max_m or rot_err > max_rad:
        raise LeadExceeded(
            f"goal leads the arm by {pos_err:.3f} m / {rot_err:.3f} rad, over "
            f"baselines.exec.max_lead_{{m,rad}} ({max_m} / {max_rad}). The arm is "
            "not tracking the plan; lower --speed-up-times or --exec-fps."
        )
    return pos_err, rot_err


# ---------------------------------------------------------------------------
# Keyboard  (copied from residual_wrapper/run_residual.py:55-92, plus left-arrow.
# Deliberate: importing run_residual pulls in viz, policy_wrapper, torch and
# multi-fast, none of which belong in a process whose policy is a ZMQ peer.)
# ---------------------------------------------------------------------------

def stdin_key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def read_key() -> str:
    """'right', 'left', 'ctrl_c', or ''. Caller must be in raw mode.

    os.read exclusively (never sys.stdin.read) so Python's text-mode buffer
    cannot swallow the CSI tail bytes before we inspect them.
    """
    time.sleep(0.03)
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):
        return "right"
    if data.startswith(b"\x1b[D") or data.startswith(b"\x1bOD"):
        return "left"
    return ""


def wait_for_right_arrow() -> None:
    old = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = read_key()
                if key == "right":
                    return
                if key == "ctrl_c":
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


class raw_stdin:
    """Raw mode for the duration of an episode, restored on any exit."""

    def __enter__(self):
        self._old = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
        return False


# ---------------------------------------------------------------------------
# Homing
# ---------------------------------------------------------------------------

def home_kwargs(args) -> dict:
    """`home_q` is keyed by the EXPOSED prefix, matching how the recordings this
    policy trained on were homed (lerobot_record_homed_single_arm.py), not by the
    physical arm as RealReach does."""
    if getattr(args, "home_q", None) is not None:
        q = np.asarray(args.home_q, dtype=np.float64)
        gripper = args.home_gripper
    else:
        pose = fc.load_home_pose(args.home_pose_name)
        q = fc.home_q(args.home_pose_name, key=ARM_KEY)
        gripper = float(pose.get("gripper", args.home_gripper))
    return dict(home_q_left=None, home_q_right=q, gripper_norm=gripper,
                max_time_s=args.home_max_time_s, tol_rad=args.home_tol_rad)


def home(controller, kwargs: dict) -> None:
    """Non-convergence warns and proceeds, as every other entrypoint here does."""
    if not controller.home(**kwargs):
        logger.warning("homing did not converge; proceeding anyway")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def dataset_fps(args) -> int:
    """The rate frames are WRITTEN at, which is the dispatch rate, not obs_fps.

    One frame per dispatched goal: the action stream is what a replay needs, and
    it advances faster than the observations. Images therefore repeat within an
    observation period -- they compress to nearly nothing and the alternative
    (one frame per observation) would drop most of the commanded actions.

    For SAIL this is the nominal FAST rate: precision modulation makes the true
    per-step rate vary, so no single fps is exactly right there. `slow_steps` in
    the metrics is how many steps ran at the slow rate.

    Integer, because the video encoder builds a Fraction from it and a float
    crashes the encoder thread mid-episode.
    """
    fps = float(getattr(args, "exec_fps", None) or fc.policy("baselines.exec.fast_fps"))
    if abs(fps - round(fps)) > 1e-9:
        logger.warning(
            "--exec-fps %.3f is not an integer; the dataset will be labelled %d Hz. "
            "Its timestamps will drift against the real dispatch rate.", fps, round(fps))
    return int(round(fps))


def build_dataset(args, controller):
    """Mirrors run_residual.py:_build_dataset, so a baseline rollout carries the
    same `observation.state` / `action` features as a residual one."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_OBS_KEYS),),
            "names": [list(STATE_OBS_KEYS)],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_KEYS),),
            "names": [list(ACTION_KEYS)],
        },
        **{
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
            }
            for name, cam in controller.cameras.items()
        },
    }
    n_cams = len(controller.cameras)
    common = dict(batch_encoding_size=1, vcodec="auto", streaming_encoding=True,
                  encoder_queue_maxsize=8, encoder_threads=2,
                  image_writer_processes=0, image_writer_threads=4 * n_cams)
    if args.resume:
        return LeRobotDataset.resume(args.repo_id, root=args.output_dir, **common)
    return LeRobotDataset.create(
        args.repo_id, dataset_fps(args), root=args.output_dir,
        robot_type=controller.name, features=features, use_videos=True, **common,
    )


def add_frame(dataset, obs: dict, action: dict, task: str, cameras) -> None:
    frame: dict = {
        "observation.state": np.array([obs[k] for k in STATE_OBS_KEYS], dtype=np.float32),
        "action": np.array([action[k] for k in ACTION_KEYS], dtype=np.float32),
        "task": task,
    }
    for name in cameras:
        img = obs.get(name)
        if isinstance(img, np.ndarray) and img.ndim == 3:
            frame[f"observation.images.{name}"] = img
    dataset.add_frame(frame)


def write_video_frame(writers, video_dir, stem, fps, cam, img, step_idx):
    """Copied from run_residual.py:155-175. Frame index == dispatch step, so two
    runs at the same rate are time-aligned for side-by-side stitching."""
    w = writers.get(cam)
    if w is None:
        video_dir.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(str(video_dir / f"{stem}_{cam}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps,
                            (img.shape[1], img.shape[0]))
        writers[cam] = w
    frame = np.ascontiguousarray(img[:, :, ::-1])  # RGB->BGR; copy keeps obs pristine
    label = f"{step_idx:05d} {step_idx / fps:6.2f}s"
    for colour, thick in (((0, 0, 0), 2), ((255, 255, 255), 1)):
        cv2.putText(frame, label, (4, frame.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, colour, thick, cv2.LINE_AA)
    w.write(frame)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One rollout attempt. `success` is operator-marked: right arrow ends the
    episode as a success, left arrow as a failure, and a timeout is a failure.
    Without that there is no time-to-success to compare the baselines on."""
    episode: int
    success: bool = False
    wall_time_s: float = 0.0
    steps: int = 0
    inferences: int = 0
    slow_steps: int = 0
    guided_inferences: int = 0
    aborted: str | None = None
    max_lead_m: float = 0.0
    max_lead_rad: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RunMetrics:
    policy: str
    header: dict = field(default_factory=dict)
    episodes: list[Episode] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n_ok = sum(1 for e in self.episodes if e.success)
        times = [e.wall_time_s for e in self.episodes if e.success]
        path.write_text(json.dumps({
            "policy": self.policy,
            **self.header,
            "summary": {
                "episodes": len(self.episodes),
                "successes": n_ok,
                "success_rate": (n_ok / len(self.episodes)) if self.episodes else 0.0,
                "mean_time_to_success_s": (sum(times) / len(times)) if times else None,
            },
            "episodes": [e.as_dict() for e in self.episodes],
        }, indent=2))
        logger.info("wrote metrics to %s (%d/%d succeeded)",
                    path, n_ok, len(self.episodes))


def sha256(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def propagate_pose(pos, quat_xyzw, deltas) -> tuple[np.ndarray, np.ndarray]:
    """Compose (dpos, drotvec) rows onto a pose, in EE_DELTA's own convention.

    `goal = measured + dpos` and `goal_rot = drot * measured_rot`, matching
    OSCGoalBuilder.from_delta -- so propagating an anchor through the deltas that
    were actually commanded gives the pose the arm WOULD be at under perfect
    tracking. That is what makes a cumulative tracking error well defined on the
    delta path, where each individual delta is relative to its own step's
    measured pose and so carries no tracking information at all.
    """
    p = np.asarray(pos, dtype=np.float64).copy()
    r = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
    for row in np.asarray(deltas, dtype=np.float64):
        p = p + row[:3]
        r = Rotation.from_rotvec(row[3:6]) * r
    return p, r.as_quat()


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def add_common_args(p, *, policy_name: str) -> None:
    """The argparse surface both entrypoints share, matching run_residual.py's."""
    p.add_argument("--rig", choices=sorted(RIGS), default="single_arm_franka",
                   help="rig profile; which physical arm it drives is in config/rig.yaml")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint path, recorded in the metrics for provenance; "
                        "the policy server is what actually loads it")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--exec-fps", type=float, default=None,
                   help="goal-push rate when not slowed; default baselines.exec.fast_fps")
    p.add_argument("--obs-fps", type=float, default=None,
                   help="observation/inference rate; default baselines.exec.obs_fps")
    p.add_argument("--dry-run", action="store_true",
                   help="connect, infer and log, but never send an action")
    p.add_argument("--allow-missing-cameras", action="store_true",
                   help="proceed when the rig lacks a camera the checkpoint wants "
                        "(the policy then runs off-distribution)")

    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--task", default=f"{policy_name} rollout")
    p.add_argument("--metrics", default=None,
                   help="output JSON; default <output-dir>/metrics.json, else "
                        "~/franka_data/baseline_eval/<policy>-<timestamp>.json")
    p.add_argument("--repo-id", default=None,
                   help="HuggingFace repo id for a recorded dataset; enables recording")
    p.add_argument("--output-dir", default=None,
                   help="local dataset root (required with --repo-id)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-videos", default=None,
                   help="directory for one time-aligned mp4 per camera")

    p.add_argument("--home-pose-name", default=fc.default_home_pose_name())
    p.add_argument("--home-q", nargs=NUM_JOINTS, type=float, default=None)
    p.add_argument("--home-gripper", type=float, default=fc.control("homing.gripper_norm"))
    p.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    p.add_argument("--home-tol-rad", type=float, default=fc.control("homing.tol_rad"))


def metrics_path(args, policy_name: str) -> Path:
    """Resolved once and cached on `args`.

    The fallback name carries a timestamp, so recomputing it would scatter each
    episode of one run into its own file -- and it is called after every episode
    precisely so an interrupted run keeps what finished.
    """
    cached = getattr(args, "_metrics_path", None)
    if cached is not None:
        return cached
    if args.metrics:
        path = Path(args.metrics).expanduser()
    elif args.output_dir:
        path = Path(args.output_dir).expanduser() / "metrics.json"
    else:
        path = (Path.home() / "franka_data" / "baseline_eval"
                / f"{policy_name}-{time.strftime('%Y%m%d_%H%M%S')}.json")
    args._metrics_path = path
    return path


class Stopper:
    """Episode termination: operator verdict, or a timeout counted as failure.

    Right arrow ends the episode as a SUCCESS, left arrow as a FAILURE. That
    verdict is the whole point -- run_residual.py has no success detection, and
    without one there is no time-to-success to compare the baselines on.
    """

    def __init__(self, episode_time_s: float | None) -> None:
        self.limit = episode_time_s
        self.t0 = time.perf_counter()
        self.verdict: str | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def check(self) -> str | None:
        """-> 'success' | 'failure' | 'timeout' | None. Raises on Ctrl-C.

        LATCHED, because a keypress is consumed by reading it: callers that poll
        more than once per step (the B-Spline loop polls between observations)
        would otherwise see the verdict once and then lose it.
        """
        if self.verdict is not None:
            return self.verdict
        if stdin_key_pressed():
            key = read_key()
            if key == "ctrl_c":
                raise KeyboardInterrupt
            if key in ("right", "left"):
                self.verdict = "success" if key == "right" else "failure"
        if self.verdict is None and self.limit is not None and self.elapsed() >= self.limit:
            self.verdict = "timeout"
        return self.verdict


def run_episodes(args, controller, metrics: RunMetrics, episode_fn) -> None:
    """Outer harness: home, wait for the operator, run, record, repeat.

    `episode_fn(controller, dispatcher, dataset, ep, stopper)` runs one episode
    and fills in `ep`. Shared so the two backends differ only in their loop.
    """
    kw = home_kwargs(args)
    dataset = None
    encoder = None
    try:
        if args.repo_id:
            from lerobot.datasets.video_utils import VideoEncodingManager
            dataset = build_dataset(args, controller)
            encoder = VideoEncodingManager(dataset)
            encoder.__enter__()

        home(controller, kw)
        for idx in range(args.num_episodes):
            print(f"\r\nepisode {idx + 1}/{args.num_episodes}: place the scene, "
                  f"then press RIGHT ARROW to start\r", flush=True)
            wait_for_right_arrow()
            print(f"\r\nrunning ({args.episode_time_s:.0f}s max). "
                  f"RIGHT = success, LEFT = failure, Ctrl-C = abort\r", flush=True)

            ep = Episode(episode=idx)
            dispatcher = Dispatcher(controller, dry_run=args.dry_run)
            stopper = Stopper(args.episode_time_s)
            try:
                with raw_stdin():
                    episode_fn(controller, dispatcher, dataset, ep, stopper)
            except (LeadExceeded, PolicyTimeout) as exc:
                # Both end THIS episode and neither ends the run: a lead abort is
                # a tracking verdict and a timeout means one inference was lost.
                # The operator can still place the scene and try again, and the
                # episode is recorded as a failure with the reason attached.
                ep.aborted = f"{type(exc).__name__}: {exc}"
                logger.error("episode %d aborted: %s", idx, exc)
            finally:
                ep.steps = dispatcher.steps
                if not ep.wall_time_s:
                    ep.wall_time_s = stopper.elapsed()
                metrics.episodes.append(ep)
                # Written after every episode, so a run interrupted at the robot
                # still leaves the episodes that did finish.
                metrics.write(metrics_path(args, metrics.policy))

            print(f"\r\nepisode {idx}: {'SUCCESS' if ep.success else 'failure'} "
                  f"in {ep.wall_time_s:.2f}s, {ep.steps} steps\r", flush=True)
            if dataset is not None:
                dataset.save_episode()
            if idx < args.num_episodes - 1:
                home(controller, kw)
    finally:
        if dataset is not None:
            if encoder is not None:
                encoder.__exit__(None, None, None)
            dataset.finalize()
            if args.push_to_hub:
                try:
                    dataset.push_to_hub()
                except Exception:
                    logger.exception("push_to_hub failed; dataset is on disk at %s",
                                     Path(args.output_dir).resolve())
