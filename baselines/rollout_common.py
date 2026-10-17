"""Shared machinery for the two baseline hardware rollouts.

Robot construction, the observation both backends are built from, the action
dict, the goal clock, and the recording/metrics side. No torch and no upstream
imports -- the policy lives in another process (see zmq_client).

Three threads run during an episode: the method's control loop (the caller),
`ObservationPump` reading the robot, and `Recorder` writing the dataset. The
camera reads block for a new frame, so keeping them off the control thread is
what lets goals go out on an even clock, as they do upstream.
"""

from __future__ import annotations

import logging
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

from baselines import run_record as rr  # noqa: E402
from baselines.force_log import ForceLog, WrenchTrace, note as force_note  # noqa: E402
from baselines.rollout_viz import CHUNKS_FILE, ChunkLog  # noqa: E402
from baselines.policy_math import (  # noqa: E402,F401  re-exported for the entrypoints
    rot6d_to_quat_xyzw, slowdown_mode, tracking_error_low,
)
from baselines.zmq_client import PolicyTimeout  # noqa: E402

# The rig -> config-class table, and the exposed key prefix, have exactly one
# definition in the tree; importing it costs the lerobot import we take anyway.
from teleop_single_arm import _ARM_KEY as ARM_KEY, _RIGS as RIGS  # noqa: E402

logger = logging.getLogger("baselines.rollout")

NUM_JOINTS = fc.num_joints()

# EE_KEYS is the EE_POS action schema itself (lerobot_source). Identical to
# run_residual.py's, so one set of tooling reads all three methods' runs.
EE_ACTION_KEYS = tuple(f"{ARM_KEY}_{ax}" for ax in EE_KEYS)
ACTION_KEYS = (*EE_ACTION_KEYS, "kp", "kd")
STATE_OBS_KEYS = (
    *(f"{ARM_KEY}_joint_{i}" for i in range(1, NUM_JOINTS + 1)),
    f"{ARM_KEY}_gripper",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def data_fps() -> float:
    """The rate the demonstrations were recorded at: upstream's `data_freq`."""
    return float(fc.control_fps())


def origin_time_scale() -> float:
    """B-Spline knot-index units per second, i.e. the recording rate; null -> data_fps()."""
    return float(fc.policy("baselines.bspline.origin_time_scale") or data_fps())


def record_fps(args) -> int:
    """Dataset frame rate. Integer: the video encoder builds a Fraction from it."""
    fps = getattr(args, "record_fps", None) or fc.policy("baselines.exec.record_fps") or data_fps()
    return int(round(float(fps)))


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

def build_robot(rig: str, control_mode: ControlMode, depth: bool = False):
    """Connectable robot for one rig profile, in an explicit control mode.

    Both single-arm profiles declare EE_DELTA in config/rig.yaml; both baselines
    need EE_POS, so the mode is passed rather than inherited.
    """
    if rig not in RIGS:
        raise ValueError(f"unknown rig {rig!r}; choose from {sorted(RIGS)}")
    arm_key = next(iter(fc.profile(rig).arms))
    if arm_key != ARM_KEY:
        raise ValueError(
            f"rig {rig!r} exposes key {arm_key!r}, not {ARM_KEY!r}; the action "
            "keys and gripper wiring assume the latter."
        )
    # A wrong rig name here means the flag did not take and the OTHER arm moves.
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

    The pose is `eef_poses_from_qpos`, the function baselines/common.py built
    every training observation from -- not residual_wrapper's current_ee_pose.
    """
    q = np.array([obs[f"{ARM_KEY}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)],
                 dtype=np.float64)
    pos, quat = eef_poses_from_qpos(q[None])
    return Measured(
        q=q,
        pos=pos[0],
        quat_xyzw=quat[0],
        gripper=float(obs[f"{ARM_KEY}_gripper"]),
        images={k: v for k, v in obs.items()
                if isinstance(v, np.ndarray) and v.ndim == 3},
    )


def fresh_pose(controller) -> tuple[np.ndarray, np.ndarray]:
    """(pos, quat_xyzw) from a state read made now, not from the last camera-paced observation."""
    kin = controller.robot_manager.current_kinematic_state_batch([ARM_KEY])
    pos, quat = eef_poses_from_qpos(np.asarray(kin[ARM_KEY][0], dtype=np.float64)[None])
    return pos[0], quat[0]


def _images(m: Measured, shapes: dict) -> dict:
    """Camera frames under the policy's `<cam>_image` keys, resized to the checkpoint's shapes."""
    out = {}
    for key, shape in shapes.items():
        if not key.endswith("_image"):
            continue
        cam = key[: -len("_image")]
        img = m.images.get(cam)
        if img is None:
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

    The single-arm profiles expose different cameras, so a checkpoint trained on
    one rig can name keys the other lacks. Returns the shape map for
    sail_obs/bspline_obs.
    """
    shapes = dict(meta["obs_key_shapes"])
    wanted = {k[: -len("_image")] for k in shapes if k.endswith("_image")}
    have = set(controller.cameras)
    missing = sorted(wanted - have)
    if missing and not allow_missing:
        raise ValueError(
            f"checkpoint needs camera(s) {missing} that this rig does not have "
            f"(it has {sorted(have)}). Pick the matching --rig, or pass "
            "--allow-missing-cameras to send blank frames and accept that the "
            "policy is off-distribution."
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


@dataclass
class Snapshot:
    """One completed `get_observation`, stamped when its arm-state read began."""
    seq: int
    t: float
    obs: dict
    m: Measured
    # A kinematic read made with it: the anchor an EE_POS goal is sent with, so
    # streaming goals costs one RPyC round trip, not two.
    kin: dict | None = None


class ObservationPump:
    """Reads the robot on its own thread; `latest()` never blocks.

    Upstream's loops read a fresh state and the camera's newest frame every
    control tick. Here a camera read blocks until the next frame and
    `get_observation` waits for every camera (~15-20 Hz measured), so the read
    lives on this thread and the control loop takes the newest snapshot.
    """

    _MIN_PERIOD_S = 0.002   # a read that returns at once must not spin a core

    def __init__(self, controller) -> None:
        self._controller = controller
        self._cond = threading.Condition()
        self._snap: Snapshot | None = None
        self._history: deque[Snapshot] = deque(maxlen=32)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, timeout_s: float = 5.0) -> Snapshot:
        self._stop.clear()
        self._snap, self._error = None, None
        self._history.clear()
        self._thread = threading.Thread(target=self._run, name="obs-pump", daemon=True)
        self._thread.start()
        return self.wait_newer(0, timeout_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                obs = self._controller.get_observation()
                # Its own read: get_observation's is consumed by the next send_action.
                kin = self._controller.read_kinematic_state()
                m = measure(obs)
            except BaseException as exc:  # surfaced to the control loop by latest()
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()
                return
            seq += 1
            with self._cond:
                self._snap = Snapshot(seq, t, obs, m, kin)
                self._history.append(self._snap)
                self._cond.notify_all()
            spare = self._MIN_PERIOD_S - (time.perf_counter() - t)
            if spare > 0:
                time.sleep(spare)

    def latest(self) -> Snapshot:
        with self._cond:
            if self._error is not None:
                raise RuntimeError("observation pump failed") from self._error
            return self._snap

    def peek(self) -> Snapshot | None:
        """The newest snapshot, or None; never raises."""
        return self._snap

    def window(self, n: int, spacing_s: float) -> list[Snapshot]:
        """The policy's observation window, oldest first, ending at the newest snapshot.

        Each earlier entry is the distinct snapshot nearest `spacing_s` before the
        next one -- the training frames' spacing. Seeded with copies of the first
        when the episode is younger than the window, as FrameStackWrapper does.
        """
        with self._cond:
            if self._error is not None:
                raise RuntimeError("observation pump failed") from self._error
            snaps = list(self._history)
        out = [snaps[-1]]
        older = snaps[:-1]
        while len(out) < n and older:
            target = out[0].t - spacing_s
            i = min(range(len(older)), key=lambda j: abs(older[j].t - target))
            out.insert(0, older[i])
            older = older[:i]
        while len(out) < n:
            out.insert(0, out[0])
        return out

    def wait_newer(self, seq: int, timeout_s: float) -> Snapshot:
        deadline = time.perf_counter() + timeout_s
        with self._cond:
            while self._error is None and (self._snap is None or self._snap.seq <= seq):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"no observation within {timeout_s:.1f}s")
                self._cond.wait(remaining)
            if self._error is not None:
                raise RuntimeError("observation pump failed") from self._error
            return self._snap


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

def _gains() -> dict:
    """Normalised zero on both gain channels: torque.osc.default_kp at the default damping ratio."""
    return {"kp": float(fc.policy("sysid.default_kp")),
            "kd": float(fc.policy("sysid.default_kd"))}


def gain_action(kp: float | None, damping_ratio: float | None = None) -> dict:
    """kp/kd action channels for stiffness `kp` and `damping_ratio`; None keeps the default.

    The inverse of resolve_gains' remap (kp = default_kp * base ** a_kp), so the
    arm's kp_limits still bind.
    """
    base = float(fc.control("torque.osc.gain_exp_base"))

    def channel(value: float, default: float, name: str) -> float:
        a = math.log(float(value) / default, base)
        if not -1.0 <= a <= 1.0:
            raise ValueError(f"osc {name} {value} is outside the gain channel's reach "
                             f"[{default / base:g}, {default * base:g}]")
        return a

    gains = _gains()
    if kp is not None:
        gains["kp"] = channel(kp, float(fc.control("torque.osc.default_kp")), "kp")
    if damping_ratio is not None:
        gains["kd"] = channel(damping_ratio, float(fc.control("torque.osc.default_damping_ratio")),
                              "damping ratio")
    return gains


def stock_gains() -> dict:
    """The controller multi-fast runs and the demonstrations were recorded under."""
    return _gains()


def sail_kp() -> float:
    """SAIL's stiffness: baselines.sail.osc_kp_scale times the stock kp."""
    return float(fc.policy("baselines.sail.osc_kp_scale")) * float(fc.control("torque.osc.default_kp"))


def sail_gains() -> dict:
    return gain_action(sail_kp(), float(fc.policy("baselines.sail.osc_damping_ratio")))


def osc_lag(gains: dict, controller=None) -> np.ndarray:
    """(6,) kd/kp per axis, as the arm resolves these gain channels.

    The OSC's ramp-following lag in seconds: a goal moved this far along its own
    velocity turns kp*e_p - kd*v into kp*e_p + kd*(v_goal - v).
    """
    from lerobot_robot_bimanual_franka.osc_torque_controller import resolve_gains

    cfg = getattr(controller, "config", None)
    scales = {k: getattr(cfg, k, None) for k in
              ("kp_ori_scale", "kd_ori_scale", "kp_pos_scale", "kd_pos_scale")}
    scales = {k: fc.control(f"tuning.{k}") if v is None else v for k, v in scales.items()}
    kp6, kd6 = resolve_gains(gains["kp"], gains["kd"], **scales)
    return np.asarray(kd6, dtype=np.float64) / np.asarray(kp6, dtype=np.float64)


def bspline_lead(speed: float, controller=None) -> np.ndarray:
    """(6,) s of goal lead that holds the stock OSC's 1x lag at `speed`; 0 at 1x.

    Upstream speeds its servo up with the plan for B-Spline rollouts
    (rollout_local_policy.py, set_env_speed_scale -> yam_server.set_ik_dt_scale),
    so the arm trails a sped-up plan by the distance it trailed it at 1x. The
    stiffness stays stock, as upstream's motor gains do, and the goal is led along
    the plan by (kd/kp)(1 - 1/speed) instead. baselines.bspline.goal_lead false
    sends the plan as sampled.
    """
    if not fc.policy("baselines.bspline.goal_lead"):
        return np.zeros(6)
    return osc_lag(stock_gains(), controller) * (1.0 - 1.0 / float(speed))


def ee_pos_action(pos, quat_xyzw, gripper: float, gains: dict | None = None) -> dict:
    """Absolute OSC goal pose. `{arm}_gripper` is absolute in [0, 1]."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    vals = (*np.asarray(pos, dtype=np.float64), *q, float(gripper))
    return {**{k: float(v) for k, v in zip(EE_ACTION_KEYS, vals)}, **(gains or _gains())}


# ---------------------------------------------------------------------------
# Clock and dispatch
# ---------------------------------------------------------------------------

def sleep_until(deadline: float) -> None:
    """Coarse sleep, then upstream's 0.1 ms spin for the last 2 ms."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(remaining - 0.002 if remaining > 0.003 else 0.0001)


class Dispatcher:
    """Sends goals to the arm and keeps the per-episode dispatch bookkeeping.

    `send` is upstream B-Spline's `env.step(action)`: the caller owns the clock.
    """

    def __init__(self, controller, dry_run: bool = False) -> None:
        self.controller = controller
        self.dry_run = dry_run
        self.wrench = WrenchTrace(ARM_KEY)
        self.steps = 0
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.last_action: dict | None = None
        # run_episodes points this at the recorder: rollout_viz draws chunks against recorded frames.
        self.frame_index = lambda: self.steps
        self._prev_send = 0.0
        self._all_gaps: list[float] = []
        self._window_gaps: list[float] = []
        self._window_start = time.perf_counter()
        self._window_steps = 0

    def send(self, action: dict, anchor: dict | None = None) -> None:
        """`anchor`: a kinematic snapshot for send_action to compose on instead of
        reading the arm again. An EE_POS goal is absolute, so the read only
        feeds diagnostics, and skipping it halves a goal's RPyC cost. The wrench
        is then the one from that snapshot's read."""
        t_send = time.perf_counter()
        if not self.dry_run:
            if anchor is None:
                self.controller.send_action(action)
            else:
                self.controller.send_action(action, anchor=anchor)
            self.wrench.sample(self.controller)
        self.last_action = action
        self.steps += 1
        self._window_steps += 1
        if self._prev_send:
            gap = (t_send - self._prev_send) * 1000.0
            self._all_gaps.append(gap)
            self._window_gaps.append(gap)
        self._prev_send = t_send
        self._log_window()

    def chunk(self, poses) -> None:
        """Log a new plan, (N, 7) base-frame [xyz, quat_xyzw]."""
        self.chunks.append((int(self.frame_index()), np.asarray(poses, dtype=np.float32)))

    def gap_stats(self) -> tuple[float | None, float | None]:
        """(mean, max) ms between consecutive goals over the episode."""
        if not self._all_gaps:
            return None, None
        return (round(sum(self._all_gaps) / len(self._all_gaps), 3),
                round(max(self._all_gaps), 3))

    def _log_window(self) -> None:
        now = time.perf_counter()
        span = now - self._window_start
        if span < 1.0 or not self._window_gaps:
            return
        logger.info("goals %.1f Hz over %.1fs  send-gap avg/max %.1f/%.1f ms",
                    self._window_steps / span, span,
                    sum(self._window_gaps) / len(self._window_gaps), max(self._window_gaps))
        self._window_start = now
        self._window_steps = 0
        self._window_gaps.clear()


def lead(goal_pos, goal_quat_xyzw, m: Measured) -> tuple[float, float]:
    """(position, orientation) divergence of an absolute goal from the arm.

    Recorded, never acted on: neither upstream bounds it, and a sped-up plan
    leads the arm on purpose.
    """
    pos_err = float(np.linalg.norm(
        np.asarray(goal_pos, dtype=np.float64) - m.pos.astype(np.float64)))
    rel = (Rotation.from_quat(np.asarray(goal_quat_xyzw, dtype=np.float64))
           * Rotation.from_quat(m.quat_xyzw.astype(np.float64)).inv())
    return pos_err, float(np.linalg.norm(rel.as_rotvec()))


def configure_process() -> None:
    """Hand the GIL over quickly, so camera and inference threads cannot hold the
    goal clock for CPython's default 5 ms switch interval."""
    sys.setswitchinterval(float(fc.policy("baselines.exec.gil_switch_interval_s")))


# ---------------------------------------------------------------------------
# Keyboard  (copied from residual_wrapper/run_residual.py, plus left-arrow)
# ---------------------------------------------------------------------------

def stdin_key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def read_key() -> str:
    """'right', 'left', 'up', 'ctrl_c', or ''. Caller must be in raw mode.

    os.read, never sys.stdin.read: the text buffer would swallow the CSI tail.
    """
    time.sleep(0.03)
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith(b"\x1b[C") or data.startswith(b"\x1bOC"):
        return "right"
    if data.startswith(b"\x1b[D") or data.startswith(b"\x1bOD"):
        return "left"
    if data.startswith(b"\x1b[A") or data.startswith(b"\x1bOA"):
        return "up"
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
    """`home_q` keyed by the EXPOSED prefix, as the training recordings were homed."""
    if getattr(args, "home_q", None) is not None:
        q = np.asarray(args.home_q, dtype=np.float64)
    else:
        q = fc.home_q(args.home_pose_name, key=ARM_KEY)
    # Every reset opens the gripper, whatever the pose file or --home-gripper says.
    return dict(home_q_left=None, home_q_right=q, gripper_norm=1.0,
                max_time_s=args.home_max_time_s, tol_rad=args.home_tol_rad)


def home(controller, kwargs: dict) -> bool:
    """Non-convergence warns and proceeds; the verdict goes into the episode row."""
    ok = bool(controller.home(**kwargs))
    if not ok:
        logger.warning("homing did not converge; proceeding anyway")
    return ok


def demo_start_q(repo_id: str) -> np.ndarray | None:
    """Mean joint configuration of the training recording's first frames.

    None when the recording is not on this machine, or its state does not name
    this rig's `{ARM_KEY}_joint_*` columns.
    """
    import json

    import pandas as pd
    from lerobot_robot_bimanual_franka.lerobot_source import resolve_root

    try:
        root = resolve_root(repo_id)
    except FileNotFoundError:
        return None
    names = json.loads((root / "meta" / "info.json").read_text())["features"][
        "observation.state"]["names"]
    if isinstance(names, dict):
        names = next(iter(names.values()))
    if names and isinstance(names[0], (list, tuple)):
        names = names[0]
    wanted = [f"{ARM_KEY}_joint_{i}" for i in range(1, NUM_JOINTS + 1)]
    if not set(wanted) <= set(names):
        return None
    frames = pd.concat([pd.read_parquet(p, columns=["observation.state", "frame_index"])
                        for p in sorted((root / "data").glob("*/*.parquet"))])
    first = np.stack(frames.loc[frames["frame_index"] == 0, "observation.state"].to_numpy())
    return first[:, [names.index(k) for k in wanted]].astype(np.float64).mean(axis=0)


def _home_gap(q, start_pos, start_quat) -> tuple[float, float]:
    """(m, rad) between the EE at joint configuration `q` and the demonstrations' start pose."""
    pos, quat = eef_poses_from_qpos(np.asarray(q, dtype=np.float64)[None])
    rot = Rotation.from_quat(quat[0]) * Rotation.from_quat(start_quat).inv()
    return float(np.linalg.norm(pos[0] - start_pos)), float(rot.magnitude())


def resolve_home(args, train_dataset: dict) -> dict:
    """Home where the training demonstrations started, and say so in the manifest.

    A policy rolled out from anywhere else starts outside its own distribution:
    its first plans drag the arm toward the demonstrations' start and disagree
    with each other, which on the arm is a lurch and then a shaking, bouncing
    rollout. Without --home-pose-name / --home-q, picks the saved pose
    (home_poses/*.json) whose EE is nearest that start; either way, refuses a
    home further than baselines.exec.home_tolerance_{m,rad} from it unless
    --allow-home-mismatch.
    """
    tol_m = float(fc.policy("baselines.exec.home_tolerance_m"))
    tol_rad = float(fc.policy("baselines.exec.home_tolerance_rad"))
    q0 = demo_start_q(train_dataset["repo_id"])
    if q0 is None:
        if args.home_q is None and args.home_pose_name is None:
            args.home_pose_name = fc.default_home_pose_name()
        logger.warning("recording %s is not on this machine; cannot check home %s against "
                       "where its demonstrations started", train_dataset["repo_id"],
                       args.home_pose_name or "--home-q")
        return {"checked": False, "home_pose": args.home_pose_name}
    pos0, quat0 = (a[0] for a in eef_poses_from_qpos(q0[None]))

    source = "flag"
    if args.home_q is None and args.home_pose_name is None:
        source = "matched to the demonstrations' start"
        scored = []
        for path in sorted(fc.home_poses_dir().glob("*.json")):
            try:
                q = fc.home_q(path.stem, key=ARM_KEY)
            except (KeyError, ValueError):
                continue
            scored.append((*_home_gap(q, pos0, quat0), path.stem))
        args.home_pose_name = min(scored)[2] if scored else fc.default_home_pose_name()
    q = (np.asarray(args.home_q, dtype=np.float64) if args.home_q is not None
         else fc.home_q(args.home_pose_name, key=ARM_KEY))
    gap_m, gap_rad = _home_gap(q, pos0, quat0)
    name = args.home_pose_name if args.home_q is None else "--home-q"
    if gap_m > tol_m or gap_rad > tol_rad:
        msg = (f"home {name!r} puts the EE {100 * gap_m:.1f} cm / {np.degrees(gap_rad):.1f} deg "
               f"from where {train_dataset['repo_id']}'s demonstrations started "
               f"(EE {np.round(pos0, 3).tolist()}, joints {np.round(q0, 3).tolist()}). "
               "The policy would start outside its training distribution.")
        if not getattr(args, "allow_home_mismatch", False):
            raise ValueError(
                f"{msg} Save that start as a home pose (scripts/home_pose.py save <name>, "
                "with the arm there) and pass --home-pose-name <name>, or pass "
                "--allow-home-mismatch to roll out from here anyway.")
        logger.warning("%s Proceeding: --allow-home-mismatch.", msg)
    logger.info("homing to %s (%s): %.1f cm / %.1f deg from the demonstrations' start",
                name, source, 100 * gap_m, np.degrees(gap_rad))
    return {"checked": True, "home_pose": name, "source": source,
            "demo_start_q": np.round(q0, 5).tolist(),
            "demo_start_ee_pos": np.round(pos0, 4).tolist(),
            "gap_m": round(gap_m, 4), "gap_rad": round(gap_rad, 4)}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def build_dataset(args, controller, root: Path):
    """Same `observation.state` / `action` features as run_residual.py's recordings."""
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
    # Frames arrive in one burst after the episode (Recorder.flush); a streaming
    # encoder drops what its queue cannot take, so encode at save_episode instead.
    common = dict(batch_encoding_size=1, vcodec="auto", streaming_encoding=False,
                  image_writer_processes=0, image_writer_threads=4 * n_cams)
    return LeRobotDataset.create(
        args.repo_id, record_fps(args), root=root,
        robot_type=controller.name, features=features, use_videos=True, **common,
    )


def frames_in_progress(dataset) -> int:
    """Frames added to the current, not-yet-saved episode (num_frames only moves on save)."""
    if dataset is None:
        return 0
    try:
        return int(dataset.writer.episode_buffer["size"])
    except Exception:
        buf = getattr(dataset, "episode_buffer", None)
        return int(buf["size"]) if buf else 0


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
    """One mp4 per camera with the frame index and time burned in."""
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


class Recorder:
    """Every 1/fps s, from its own thread: the newest observation and the goal last sent.

    Upstream B-Spline's `record_stride` writer. Frames are only buffered during
    the episode and written by `flush` after it: the dataset's video encoder
    holds the GIL for ~1 s when an episode's first frame arrives, which froze
    the goal stream at the start of every episode.
    """

    def __init__(self, pump: ObservationPump, dispatcher: Dispatcher, fps: int) -> None:
        self.pump = pump
        self.dispatcher = dispatcher
        self.fps = int(fps)
        self.buffer: list[tuple[dict, dict]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def frames(self) -> int:
        return len(self.buffer)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        period = 1.0 / self.fps
        t0 = time.perf_counter()
        k = 0
        while not self._stop.wait(max(0.0, t0 + k * period - time.perf_counter())):
            k += 1
            snap = self.pump.peek()
            action = self.dispatcher.last_action
            if snap is not None and action is not None:
                self.buffer.append((snap.obs, action))

    def flush(self, dataset, task: str, cameras, video_dir: Path | None, video_stem: str) -> int:
        """Write the buffered frames; returns how many went into `dataset`."""
        writers: dict = {}
        try:
            for i, (obs, action) in enumerate(self.buffer):
                if dataset is not None:
                    add_frame(dataset, obs, action, task, cameras)
                if video_dir is not None:
                    for cam in cameras:
                        img = obs.get(cam)
                        if isinstance(img, np.ndarray) and img.ndim == 3:
                            write_video_frame(writers, video_dir, video_stem, self.fps, cam, img, i)
        finally:
            for w in writers.values():
                w.release()
        return len(self.buffer) if dataset is not None else 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One rollout attempt, as written to episodes.jsonl.

    `success` is operator-marked: right arrow ends the episode as a success,
    left arrow as a failure, and a timeout is a failure.
    """
    episode: int
    success: bool = False
    verdict: str | None = None          # success | failure | timeout | aborted
    wall_time_s: float = 0.0
    started_at: str | None = None
    ended_at: str | None = None

    steps: int = 0                      # goals dispatched
    frames_recorded: int = 0            # rows added to the LeRobotDataset
    dataset_episode_index: int | None = None
    inferences: int = 0
    slow_steps: int = 0                 # SAIL precision modulation
    guided_inferences: int = 0          # SAIL EAG fired

    exec_fps: float | None = None       # nominal goal rate
    achieved_fps: float | None = None   # steps / wall_time
    send_gap_ms_mean: float | None = None
    send_gap_ms_max: float | None = None

    max_lead_m: float = 0.0
    max_lead_rad: float = 0.0
    # |F| at the EE over the goals dispatched, from libfranka's estimated external wrench.
    ee_force_n: dict | None = None
    homed: bool | None = None
    aborted: str | None = None
    notes: dict = field(default_factory=dict)   # per-method extras

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


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
    p.add_argument("--record-fps", type=float, default=None,
                   help="dataset frame rate; default baselines.exec.record_fps "
                        "(null = the demonstrations' rate)")
    p.add_argument("--dry-run", action="store_true",
                   help="connect, infer and log, but never send an action")
    p.add_argument("--allow-missing-cameras", action="store_true",
                   help="proceed when the rig lacks a camera the checkpoint wants "
                        "(the policy then runs off-distribution)")

    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--task", default=f"{policy_name} rollout")
    add_output_args(p)

    p.add_argument("--home-pose-name", default=None,
                   help="home_poses/<name>.json; default: the saved pose nearest where "
                        "the training demonstrations started")
    p.add_argument("--home-q", nargs=NUM_JOINTS, type=float, default=None)
    p.add_argument("--allow-home-mismatch", action="store_true",
                   help="roll out from a home that is not where the demonstrations "
                        "started (the policy then starts off-distribution)")
    p.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    p.add_argument("--home-tol-rad", type=float, default=fc.control("homing.tol_rad"))


def environment(args, controller) -> dict:
    """The rig this run actually drove, PHYSICAL arm included (the key prefix is `r_` on both)."""
    profile = fc.profile(args.rig)
    arm_name = profile.arms[ARM_KEY]
    spec = fc.arm(arm_name)
    return {
        "rig_profile": args.rig,
        "key_prefix": ARM_KEY,
        "physical_arm": arm_name,
        "arm": {
            "robot_ip": spec.robot_ip,
            "server_ip": spec.server_ip,
            "rpyc_port": spec.rpyc_port,
            "gripper_rpyc_port": spec.gripper_rpyc_port,
            "gripper_kind": spec.gripper.kind,
            "gripper_ip": spec.gripper.ip,
            "nuc": spec.ssh_target,
            "ee_sphere": {"center_tool_m": list(spec.ee_sphere.center_tool_m),
                          "radius_m": spec.ee_sphere.radius_m},
        },
        "cameras": {name: [cam.height, cam.width]
                    for name, cam in controller.cameras.items()},
        "robot_type": controller.name,
        "control_fps": fc.control_fps(),
        "home_pose": args.home_pose_name,
        "home_q_override": list(args.home_q) if args.home_q else None,
        "torque": {
            "default_kp": fc.control("torque.osc.default_kp"),
            "default_damping_ratio": fc.control("torque.osc.default_damping_ratio"),
            "gain_exp_base": fc.control("torque.osc.gain_exp_base"),
            "uncouple_pos_ori": fc.control("torque.osc.uncouple_pos_ori"),
            "cross_coupling_compensation": fc.control(
                "torque.osc.cross_coupling_compensation", None),
            "rotor_inertia_kg_m2": list(fc.control("torque.rotor_inertia_kg_m2")),
            "delta_pos_max_m": fc.control("torque.delta.pos_max_m"),
            "delta_rot_max_rad": fc.control("torque.delta.rot_max_rad"),
        },
        "tuning": {
            "ee_translation_fudge": fc.control("tuning.ee_translation_fudge"),
            "ee_rotation_fudge": fc.control("tuning.ee_rotation_fudge"),
            "friction_kc": fc.control("tuning.friction_kc"),
            "kp_ori_scale": list(fc.control("tuning.kp_ori_scale")),
            "kp_pos_scale": list(fc.control("tuning.kp_pos_scale")),
            "kd_ori_scale": list(fc.control("tuning.kd_ori_scale")),
            "kd_pos_scale": list(fc.control("tuning.kd_pos_scale")),
        },
        "safety": {
            "worktable_height_m": fc.worktable_height_m(),
            "brake_distance_min_m": fc.control("worktable_brake.distance_min_m"),
        },
        "dry_run": bool(args.dry_run),
    }


def _shared_parameters(args) -> dict:
    return {
        "data_fps": data_fps(),
        "record_fps": record_fps(args),
        "num_episodes": args.num_episodes,
        "episode_time_s": args.episode_time_s,
        "task": args.task,
        "allow_missing_cameras": bool(args.allow_missing_cameras),
    }


def _osc(gains: dict) -> dict:
    base = float(fc.control("torque.osc.gain_exp_base"))
    return {"osc_kp": float(fc.control("torque.osc.default_kp")) * base ** gains["kp"],
            "osc_damping_ratio": float(fc.control("torque.osc.default_damping_ratio"))
            * base ** gains["kd"],
            "gain_action": [gains["kp"], gains["kd"]]}


def sail_parameters(args, meta: dict, settings: dict) -> dict:
    """Every knob SAIL's executor resolved."""
    return {
        **_shared_parameters(args),
        "speed_up_times": settings["speed"],
        "exec_fps": settings["goal_hz"],
        "goal_hz": settings["goal_hz"],
        "fast_fps": settings["fast_hz"],
        "slow_fps": settings["slow_hz"],
        "precision_modulation": settings["precision"],
        "precision_available": bool(meta.get("precision_column")),
        "eag": settings["eag"],
        "eag_available": bool(meta.get("fac_enabled")),
        "eag_horizon": meta.get("fac_horizon"),
        "inf_delay": settings["inf_delay"],
        "execute_n_actions": settings["execute_n"],
        "slowdown_window_size": settings["window"],
        "pos_teb": settings["pos_teb"],
        "ori_teb": settings["ori_teb"],
        "velocity_lead_s": np.round(settings["lead"], 5).tolist(),
        "spline_knot_rows": settings["knot_rows"],
        "frame_spacing_s": settings["frame_spacing_s"],
        "action_horizon": meta.get("action_horizon"),
        "prediction_horizon": meta.get("prediction_horizon"),
        **_osc(settings["gains"]),
    }


def bspline_parameters(args, meta: dict, planner_kwargs: dict, control_freq: float,
                       lead: np.ndarray) -> dict:
    """Every knob the spline planner resolved."""
    return {
        **_shared_parameters(args),
        "exec_fps": control_freq,
        "control_freq": control_freq,
        **{k: v for k, v in planner_kwargs.items() if not k.startswith("_")},
        "action_format": meta.get("action_format"),
        "act_dim": meta.get("act_dim"),
        "grasp_hold_s": float(fc.policy("baselines.bspline.grasp_hold_s")),
        "grasp_threshold": float(fc.policy("baselines.bspline.grasp_threshold")),
        "goal_lead_s": np.round(lead, 5).tolist(),
        **_osc(stock_gains()),
    }


def add_output_args(p) -> None:
    """Where a run is filed. Shared with residual_wrapper so all three methods
    land in the same tree under one set of flag names."""
    p.add_argument("--train-dataset", default=None,
                   help="repo id of the dataset this policy was TRAINED on; names "
                        "the output directory. Optional when the checkpoint carries "
                        "it (converter --source-repo-id)")
    p.add_argument("--outputs-root", default=None,
                   help=f"default {rr.DEFAULT_ROOT}")
    p.add_argument("--repo-id", default=None,
                   help="repo id for the LeRobotDataset recorded during the run; "
                        "default <train-dataset>-<method>-<timestamp>")
    p.add_argument("--no-record", action="store_true",
                   help="skip the LeRobotDataset (manifest and episodes are always written)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--save-videos", action="store_true",
                   help="one time-aligned mp4 per camera into <run-dir>/videos")


def open_run(args, method: str, train_dataset: dict) -> tuple[rr.RunDir, rr.RunRecord]:
    """Create the run directory and start its manifest."""
    run_dir = rr.RunDir(train_dataset["repo_id"], method,
                        root=args.outputs_root or rr.DEFAULT_ROOT)
    record = rr.RunRecord(run_dir, method, train_dataset)
    if args.repo_id is None:
        args.repo_id = f"{Path(train_dataset['repo_id']).name}-{run_dir.run_id}"
    logger.info("run directory: %s", run_dir.path)
    return run_dir, record


class Stopper:
    """Episode termination: operator verdict, or a timeout counted as failure."""

    def __init__(self, episode_time_s: float | None) -> None:
        self.limit = episode_time_s
        self.t0 = time.perf_counter()
        self.verdict: str | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def check(self) -> str | None:
        """-> 'success' | 'failure' | 'retry' | 'timeout' | None. Raises on Ctrl-C.

        Latched: a keypress is consumed by reading it.
        """
        if self.verdict is not None:
            return self.verdict
        if stdin_key_pressed():
            key = read_key()
            if key == "ctrl_c":
                raise KeyboardInterrupt
            if key in ("right", "left", "up"):
                self.verdict = {"right": "success", "left": "failure", "up": "retry"}[key]
        if self.verdict is None and self.limit is not None and self.elapsed() >= self.limit:
            self.verdict = "timeout"
        return self.verdict


def run_episodes(args, controller, run_dir, record, episode_fn, *, nominal_hz: float) -> None:
    """Outer harness: home, wait for the operator, run, record, repeat.

    `episode_fn(controller, dispatcher, pump, ep, stopper)` runs one episode and
    fills in `ep`; the pump and the recorder run only while it does.
    """
    kw = home_kwargs(args)
    homer = None
    dataset = None
    encoder = None
    forces = ForceLog(run_dir.force_profiles_path)
    chunks = ChunkLog(run_dir.path / CHUNKS_FILE)
    record.set("outputs", force_profiles=str(run_dir.force_profiles_path))
    try:
        if not args.no_record:
            from lerobot.datasets.video_utils import VideoEncodingManager
            dataset = build_dataset(args, controller, run_dir.dataset_dir)
            encoder = VideoEncodingManager(dataset)
            encoder.__enter__()
        record.set("outputs", dataset={
            "recorded": dataset is not None,
            "repo_id": args.repo_id if dataset is not None else None,
            "path": str(run_dir.dataset_dir) if dataset is not None else None,
            "fps": record_fps(args),
        }, videos_dir=str(run_dir.video_dir) if args.save_videos else None)

        homed = home(controller, kw)
        homer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="home")
        idx = 0
        while idx < args.num_episodes:
            print(f"\r\nepisode {idx + 1}/{args.num_episodes}: place the scene, "
                  f"then press RIGHT ARROW to start\r", flush=True)
            wait_for_right_arrow()
            print(f"\r\nrunning ({args.episode_time_s:.0f}s max). "
                  f"RIGHT = success, LEFT = failure, UP = retry, Ctrl-C = abort\r", flush=True)

            ep = Episode(episode=idx, homed=homed, started_at=rr.stamp(), exec_fps=nominal_hz)
            dispatcher = Dispatcher(controller, dry_run=args.dry_run)
            pump = ObservationPump(controller)
            recorder = Recorder(pump, dispatcher, record_fps(args))
            dispatcher.frame_index = lambda r=recorder: r.frames
            stopper = None
            interrupted = False
            try:
                pump.start()
                recorder.start()
                stopper = Stopper(args.episode_time_s)
                with raw_stdin():
                    episode_fn(controller, dispatcher, pump, ep, stopper)
            except PolicyTimeout as exc:
                # The episode is a failure with the reason attached; the run goes on.
                ep.aborted = f"{type(exc).__name__}: {exc}"
                ep.verdict = "aborted"
                logger.error("episode %d aborted: %s", idx, exc)
            except KeyboardInterrupt:
                interrupted = True
                raise
            finally:
                recorder.stop()
                pump.stop()
                # Home while this episode is written out, not after: the two take
                # seconds each and share nothing. Not on Ctrl-C or a crash.
                failed = interrupted or sys.exc_info()[1] is not None
                homing = None if failed else homer.submit(home, controller, kw)
                ep.steps = dispatcher.steps
                if stopper is not None and not ep.wall_time_s:
                    ep.wall_time_s = stopper.elapsed()
                ep.ended_at = rr.stamp()
                ep.verdict = ep.verdict or (stopper.verdict if stopper else None) or "incomplete"
                # A retry leaves no trace: no episode row, frames, video, force or chunks.
                retry = ep.verdict == "retry" and not interrupted
                if ep.wall_time_s > 0:
                    ep.achieved_fps = round(ep.steps / ep.wall_time_s, 2)
                ep.send_gap_ms_mean, ep.send_gap_ms_max = dispatcher.gap_stats()
                if not retry:
                    ep.ee_force_n = forces.add_trace(idx, dispatcher.wrench)
                    chunks.add(idx, dispatcher.chunks)
                if not interrupted and not retry:
                    ep.frames_recorded = recorder.flush(
                        dataset, args.task, controller.cameras,
                        run_dir.video_dir if args.save_videos else None,
                        f"{record.method}_ep{idx:03d}")
                if dataset is not None:
                    ep.dataset_episode_index = dataset.num_episodes if ep.frames_recorded else None
                if not retry:
                    record.add_episode(ep)

            if retry:
                print(f"\r\nepisode {idx}: RETRY after {ep.wall_time_s:.2f}s; discarded, "
                      f"running it again\r", flush=True)
                homed = homing.result()
                continue

            print(f"\r\nepisode {idx}: {ep.verdict.upper()} "
                  f"in {ep.wall_time_s:.2f}s, {ep.steps} steps{force_note(ep.ee_force_n)}\r",
                  flush=True)
            if dataset is not None:
                if ep.frames_recorded:
                    # In-process: NVENC cannot initialise CUDA in a forked encoder.
                    dataset.save_episode(parallel_encoding=False)
                else:
                    logger.warning("episode %d recorded no frames; not saved", idx)
            homed = homing.result()
            idx += 1
    finally:
        if homer is not None:
            homer.shutdown(wait=True)
        if dataset is not None:
            if encoder is not None:
                encoder.__exit__(None, None, None)
            dataset.finalize()
            record.set("outputs", dataset={
                "recorded": True,
                "repo_id": args.repo_id,
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
                       videos=sorted(p.name for p in run_dir.video_dir.glob("*.mp4")))
        record.write()
