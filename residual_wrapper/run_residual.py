"""Entry point for running and recording residual-policy episodes on the Franka.

Two residual families, told apart by the checkpoint --residual-policy names:

  best.pt (torch)        the point-cloud residual on a LeRobot base policy: the
                         loop in this file, recorded as a LeRobotDataset under a
                         run directory. The base emits ABSOLUTE poses (EE_POS; a
                         delta-trained checkpoint is refused). Each chunk is taken
                         relative to the pose the base planned from, the residual
                         is summed there, and each target is dispatched as the
                         one-step EE_DELTA from the pose read at that step
                         (env_wrapper, "Chunk-start-relative targets").
  ft_policy_*.zip (SB3)  multi-fast's FAST residual on the analytic reach base:
                         reach_residual.run, recorded as episodes.hdf5 plus the
                         same viz.py HTML per episode (--policy both runs the
                         base alone and then base + residual on one curve).

  python residual_wrapper/run_residual.py --base-policy <ckpt> --residual-policy best.pt
  python residual_wrapper/run_residual.py --residual-policy ft_policy_400000_steps.zip --num-episodes 5
"""

import argparse
import itertools
import json
import logging
import os
import select
import sys
import termios
import time
import tty
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import env_wrapper
from env_wrapper import (
    _ACTION_KEYS,
    _ARM_KEY,
    _CHUNK_EXEC,
    _RESIDUAL_HORIZON,
    _ROT_SCALE,
    _STATE_OBS_KEYS,
    chunk_to_relative,
    compose_chunk,
    current_ee_pose,
    delta_action,
    ee_pose_to_world,
    measured_ee_twist_world,
    relative_to_poses,
    residual_input,
    split_gripper,
    strip_depth,
    target_to_delta,
    to_sim_world_points,
    to_sim_world_pose,
    to_sim_world_twist,
)
from env_wrapper import default_home_q as _default_home_q
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from policy_wrapper import BasePolicy, ResidualPolicy, Trajectory
from viz import (EpisodeRecorder, save_episode_html, save_latency_html, save_policy_pcd_npz,
                 save_rollout_html)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.force_log import ForceLog, WrenchTrace, note as force_note  # noqa: E402
from baselines.rollout_common import Episode, environment, frames_in_progress  # noqa: E402

logger = logging.getLogger(__name__)

METHOD = "multifast"  # this method's name in the shared outputs tree, beside sail and bspline
_POSES_DIR = fc.home_poses_dir()
_PLANT_DIR = Path(__file__).resolve().parent.parent / "multi-fast" / "cfg" / "plant"


# ---------------------------------------------------------------------------
# Operator keys
# ---------------------------------------------------------------------------

def _stdin_key_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _read_key() -> str:
    """'right', 'left', 'ctrl_c' or ''. The caller holds stdin in raw mode."""
    time.sleep(0.03)  # let the whole escape sequence arrive
    # os.read, not sys.stdin.read: the text buffer would swallow the CSI tail bytes.
    data = os.read(sys.stdin.fileno(), 16)
    if b"\x03" in data:
        return "ctrl_c"
    if data.startswith((b"\x1b[C", b"\x1bOC")):
        return "right"
    if data.startswith((b"\x1b[D", b"\x1bOD")):
        return "left"
    return ""


@contextmanager
def _raw_terminal():
    old = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    try:
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def _wait_for_right_arrow() -> None:
    """Block until RIGHT is pressed; Ctrl-C raises KeyboardInterrupt."""
    with _raw_terminal():
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = _read_key()
                if key == "right":
                    return
                if key == "ctrl_c":
                    raise KeyboardInterrupt


def _verdict(elapsed_s: float, episode_time_s: float | None) -> str | None:
    """'timeout', the operator's 'success' (RIGHT) or 'failure' (LEFT), else None."""
    if episode_time_s is not None and elapsed_s >= episode_time_s:
        return "timeout"
    if not _stdin_key_pressed():
        return None
    key = _read_key()
    if key == "ctrl_c":
        raise KeyboardInterrupt
    return {"right": "success", "left": "failure"}.get(key)


# ---------------------------------------------------------------------------
# Checkpoint guards
# ---------------------------------------------------------------------------

def _plant_mismatch(name: str) -> list[str]:
    """How config/control.yaml's controller law differs from a multi-fast plant's."""
    import yaml
    plant = yaml.safe_load((_PLANT_DIR / f"{name}.yaml").read_text())
    law = {
        "law_armature": (plant.get("law_armature"), fc.control("torque.rotor_inertia_kg_m2")),
        "kp": (plant.get("kp"), fc.control("torque.osc.default_kp")),
        "damping": (plant.get("damping"), fc.control("torque.osc.default_damping_ratio")),
        "uncouple_pos_ori": (plant.get("uncouple_pos_ori"), fc.control("torque.osc.uncouple_pos_ori")),
    }
    return [f"{k}: plant {a} vs control.yaml {b}" for k, (a, b) in law.items()
            if a is not None and not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float))]


def _check_teacher(residual: ResidualPolicy, allow_plant_mismatch: bool) -> None:
    """eval_distill's guards: one base-action contract, and the plant the student trained on."""
    t = residual.teacher
    if not t:
        return
    want = {"goal_mode": "target", "chunk_relative": True,
            "target_orientation": True, "impedance_mode": "variable"}
    bad = {k: t.get(k) for k, v in want.items() if t.get(k) != v}
    if bad:
        raise SystemExit(f"residual trained on teacher settings {bad}; this executor runs {want}")
    plants = t["plant"] if isinstance(t.get("plant"), list) else [t.get("plant")]
    report = {p: _plant_mismatch(p) for p in plants if p and (_PLANT_DIR / f"{p}.yaml").is_file()}
    if not report or any(not v for v in report.values()):
        return
    lines = "\n  ".join(f"{p}: {m}" for p, ms in report.items() for m in ms)
    msg = (f"the residual trained on plant(s) {plants}, whose controller law differs from "
           f"config/control.yaml:\n  {lines}\nSet torque: to the plant's values and re-run "
           "scripts/deploy_nuc_server.sh, or pass --allow-plant-mismatch.")
    if not allow_plant_mismatch:
        raise SystemExit(msg)
    logger.warning(msg)


def _check_base(base_policy, controller, args) -> str:
    """The base's training action space, which must be EE_POS, on a rig with its cameras."""
    try:
        space = base_policy.action_space()
    except (FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"cannot verify the base policy's action space: {exc}") from exc
    if space != "EE_POS":
        raise SystemExit(
            f"{args.base_policy or args.replay_dataset}: trained on {space} actions, and this "
            "runner executes absolute EE_POS targets. Train the base on an EE_POS recording "
            "(or relabel one with scripts/replay_dataset.py --mode ee_pose).")
    print(f"base policy action space: {space}")
    cfg = getattr(getattr(base_policy, "policy", None), "config", None)
    missing = [k for k in (getattr(cfg, "image_features", None) or {})
               if k.removeprefix("observation.images.") not in controller.cameras]
    if missing:
        raise SystemExit(
            f"{args.base_policy}: needs {missing}, and rig {args.rig} has cameras "
            f"{sorted(controller.cameras)}. Pass the --rig it was recorded on.")
    return space


# ---------------------------------------------------------------------------
# Planning a chunk
# ---------------------------------------------------------------------------

def _pose(kin: dict) -> tuple[np.ndarray, np.ndarray]:
    """The driven arm's base-frame EE position and quaternion (xyzw) in a kinematic snapshot."""
    snap = kin[_ARM_KEY]
    return np.array(snap[3], dtype=np.float64), np.array(snap[4], dtype=np.float64)


@dataclass
class Snapshot:
    """One control step's reads. Inference may hold them on another thread."""
    obs: dict             # depth stripped
    ee_pose: np.ndarray   # the residual's proprio pose
    kin: dict             # get_observation's state read: the chunk anchor
    cloud: np.ndarray     # a fresh array per get_observation
    t: float

    @classmethod
    def read(cls, controller, sim_convention: bool, t: float) -> "Snapshot":
        obs = controller.get_observation()
        # controller.kin now: send_action consumes it.
        return cls(strip_depth(obs), current_ee_pose(obs, sim_convention=sim_convention),
                   controller.kin, controller.last_full_point_cloud, t)


@dataclass
class Chunk:
    """A planned chunk: normalised relative to its anchor, and as base-frame poses (anchor first)."""
    base_rel: np.ndarray
    res_chunk: np.ndarray
    total_rel: np.ndarray
    anchor_pos: np.ndarray
    base_poses: np.ndarray
    total_poses: np.ndarray
    network_pcd: np.ndarray | None
    latency: dict


class ChunkPlanner:
    """Base (+ residual) inference over a Snapshot, on the loop thread or the prefetch worker."""

    def __init__(self, controller, base_policy, residual: ResidualPolicy | None,
                 proprio_frame: str = "world", sim_convention: bool = True,
                 dump_dir: Path | None = None, latency_debug: bool = False) -> None:
        self.controller = controller
        self.base = base_policy
        self.residual = residual
        self.world = proprio_frame == "world"
        self.sim_convention = sim_convention
        self.dump_dir = dump_dir
        self.latency_debug = latency_debug

    def plan(self, window: list[dict], snap: Snapshot, gains: tuple[float, float],
             index: int) -> Chunk:
        """`gains` is the (kp, kd) in effect when snap was read; `index` names the dump file."""
        if snap.kin is None:
            raise RuntimeError("no kinematic snapshot to anchor the chunk on; "
                               "get_observation() must precede inference")
        t0 = time.perf_counter()
        start_pos = self._pos_now()
        base_chunk = self.base.infer(window)
        t_base = time.perf_counter()

        anchor_pos, anchor_quat = _pose(snap.kin)
        base_rel = chunk_to_relative(base_chunk, anchor_pos, anchor_quat)
        res_chunk, network_pcd, bound = np.empty((0, 9)), None, float(len(base_rel))
        if self.residual is not None:
            residual_obs = self._residual_obs(snap, base_rel, gains)
            res_chunk = self.residual.infer(residual_obs)
            network_pcd = self.residual.last_network_pcd
            bound = self.residual.composed_action_bound
            self._dump(index, base_chunk, anchor_pos, anchor_quat, residual_obs, res_chunk)
        t_res = time.perf_counter()
        # Composed once per chunk, as StudentPredictor composes it.
        total_rel = compose_chunk(base_rel, res_chunk, bound=bound)

        latency = {
            "base_ms": (t_base - t0) * 1e3,
            "residual_ms": (t_res - t_base) * 1e3,
            "total_ms": (time.perf_counter() - t0) * 1e3,
            "start_pos": start_pos,
            "end_pos": self._pos_now(),
            "t_obs": snap.t,
        }
        return Chunk(base_rel, res_chunk, total_rel, anchor_pos,
                     relative_to_poses(base_rel, anchor_pos, anchor_quat),
                     relative_to_poses(total_rel, anchor_pos, anchor_quat),
                     network_pcd, latency)

    def warmup(self) -> None:
        """Pay torch.compile, cuDNN selection and the CUDA context before the episode.

        Once on a worker too: the prefetch calls the same compiled modules from there.
        """
        t0 = time.perf_counter()
        snap = Snapshot.read(self.controller, self.sim_convention, t0)
        self.base.observe(snap.obs)
        args = (self.base.window(), snap, (0.0, 0.0), 0)
        for _ in range(3):
            self.plan(*args)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(self.plan, *args).result()
        self.base.reset()
        print(f"policy warmup done in {time.perf_counter() - t0:.1f}s")

    def _residual_obs(self, snap: Snapshot, base_rel: np.ndarray,
                      gains: tuple[float, float]) -> dict:
        """The student's observation, in sim's world convention unless --raw-proprio."""
        kp, kd = gains
        pose, cloud = snap.ee_pose, snap.cloud
        # Measured twist (J @ dq), in the proprio pose's frame.
        vel = measured_ee_twist_world(snap.kin[_ARM_KEY],
                                      self.controller._r_robot_in_world if self.world else np.eye(3))
        if self.world:
            # The cloud is world-frame; the pose must be too, for center_on_eef.
            pose = ee_pose_to_world(pose, self.controller._r_robot_in_world,
                                    self.controller._t_robot_in_world)
            if self.sim_convention:
                pose, vel, cloud = to_sim_world_pose(pose), to_sim_world_twist(vel), to_sim_world_points(cloud)
        return {
            "action_chunk": residual_input(base_rel[:_RESIDUAL_HORIZON]),
            "proprio": np.concatenate([
                split_gripper(pose).astype(np.float32),
                np.array([kd, kp], dtype=np.float32),  # sim's controller_state: damping first
                np.asarray(vel, dtype=np.float32),
            ]),
            "point_cloud": cloud,
        }

    def _dump(self, index: int, base_chunk: np.ndarray, anchor_pos: np.ndarray,
              anchor_quat: np.ndarray, residual_obs: dict, res_chunk: np.ndarray) -> None:
        if self.dump_dir is None:
            return
        np.savez_compressed(
            self.dump_dir / f"obs_{index:05d}.npz",
            base_chunk_raw=base_chunk.astype(np.float32),
            anchor_pos=anchor_pos.astype(np.float32),
            anchor_quat=anchor_quat.astype(np.float32),
            action_chunk=residual_obs["action_chunk"],
            proprio=residual_obs["proprio"],
            point_cloud=residual_obs["point_cloud"],
            network_pcd=self.residual.last_network_pcd,
            res_chunk=res_chunk.astype(np.float32),
        )

    def _pos_now(self) -> np.ndarray | None:
        return _pose(self.controller.read_kinematic_state())[0] if self.latency_debug else None


# ---------------------------------------------------------------------------
# Executing a chunk
# ---------------------------------------------------------------------------

class ChunkCursor:
    """The chunk being executed and the step within it."""

    def __init__(self, chunk: Chunk, replaying: bool) -> None:
        self.chunk = chunk
        # A replayed recording's own targets and gains drive the arm; its residual is shown only.
        self.rel = chunk.base_rel if replaying else chunk.total_rel
        self.poses = chunk.base_poses if replaying else chunk.total_poses
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= _CHUNK_EXEC

    @property
    def target(self) -> np.ndarray:
        return self.poses[self.i + 1]

    def action(self, kin_now: dict) -> dict:
        """This step's target as the one-step EE_DELTA from the pose in kin_now."""
        pos, quat = _pose(kin_now)
        row = self.rel[self.i]
        return delta_action(target_to_delta(self.target, pos, quat),
                            gripper=float(row[6]), kp=float(row[7]), kd=float(row[8]))


def _record_chunk(recorder: EpisodeRecorder, chunk: Chunk) -> None:
    if chunk.network_pcd is not None:
        recorder.record_policy_pcd(len(recorder), chunk.network_pcd)
    recorder.record_chunk(
        step=len(recorder),
        ee_pos=chunk.anchor_pos.astype(np.float32),
        base_traj=chunk.base_poses[:, :3].astype(np.float32),
        total_traj=chunk.total_poses[:, :3].astype(np.float32),
        base_traj_pose=chunk.base_poses.astype(np.float32),
        total_traj_pose=chunk.total_poses.astype(np.float32),
    )


def _record_step(recorder: EpisodeRecorder, snap: Snapshot, cursor: ChunkCursor,
                 action: dict) -> None:
    chunk, i = cursor.chunk, cursor.i
    res = chunk.res_chunk[i] if i < len(chunk.res_chunk) else None
    recorder.record(
        q=np.array([snap.obs[f"r_joint_{j}"] for j in range(1, 8)]),
        # The observation's pose, which is what FK(q) shows; base frame, as the targets.
        actual_ee_pos=_pose(snap.kin)[0].astype(np.float32),
        base_desired_pos=chunk.base_poses[i + 1, :3].astype(np.float32),
        total_desired_pos=cursor.target[:3].astype(np.float32),
        kp=action["kp"],
        kd=action["kd"],
        gripper=action[f"{_ARM_KEY}_gripper"],
        res_gripper=float(res[8]) if res is not None else 0.0,
        point_cloud=snap.cloud,
        res_rotvec=((cursor.rel[i, 3:6] - chunk.base_rel[i, 3:6]) * _ROT_SCALE).astype(np.float32),
    )


def _dataset_frame(obs: dict, action: dict, task: str, cameras) -> dict:
    frame = {
        "observation.state": np.array([obs[k] for k in _STATE_OBS_KEYS], dtype=np.float32),
        "action": np.array([action[k] for k in _ACTION_KEYS], dtype=np.float32),
        "task": task,
    }
    for cam in cameras:
        img = obs.get(cam)
        if isinstance(img, np.ndarray) and img.ndim == 3:
            frame[f"observation.images.{cam}"] = img
    return frame


class StepVideo:
    """One mp4 per camera. Frame index == control step, so runs at one fps stitch side by side."""

    def __init__(self, video_dir: Path | None, stem: str, fps: float, cams: list[str]) -> None:
        self.dir, self.stem, self.fps, self.cams = video_dir, stem, fps, cams
        self.writers: dict[str, cv2.VideoWriter] = {}

    def write(self, obs: dict, step: int) -> None:
        if self.dir is None:
            return
        for cam in self.cams:
            img = obs.get(cam)
            if isinstance(img, np.ndarray) and img.ndim == 3:
                self._writer(cam, img).write(self._frame(img, step))

    def close(self) -> None:
        for w in self.writers.values():
            w.release()
        if self.writers:
            logger.info("saved %d camera video(s) to %s", len(self.writers), self.dir)

    def _writer(self, cam: str, img: np.ndarray) -> cv2.VideoWriter:
        if cam not in self.writers:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.writers[cam] = cv2.VideoWriter(str(self.dir / f"{self.stem}_{cam}.mp4"),
                                                cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                                (img.shape[1], img.shape[0]))
        return self.writers[cam]

    def _frame(self, img: np.ndarray, step: int) -> np.ndarray:
        frame = np.ascontiguousarray(img[:, :, ::-1])  # RGB->BGR copy; the obs image stays pristine
        label, org = f"{step:05d} {step / self.fps:6.2f}s", (4, frame.shape[0] - 6)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


class Pacer:
    """Absolute deadlines, so idle steps absorb the few ms inference steps overrun."""

    def __init__(self, fps: float) -> None:
        self.dt = 1.0 / fps
        self.deadline = time.perf_counter() + self.dt

    def wait(self) -> None:
        sleep_s = self.deadline - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        self.deadline += self.dt
        if self.deadline < time.perf_counter():  # after a stall, resync rather than repay it
            self.deadline = time.perf_counter() + self.dt


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _peak(v: list[float]) -> float:
    return max(v) if v else 0.0


class LoopStats:
    """Logged once a second: loop rate, busy time, send gaps and prefetch waits (ms)."""

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.t_prev_send = 0.0
        self._restart(time.perf_counter())

    def _restart(self, now: float) -> None:
        self.start, self.frames = now, 0
        self.busy, self.chunk_busy, self.wait, self.send_gap = [], [], [], []

    def sent(self, t_send: float) -> None:
        if self.t_prev_send:
            self.send_gap.append((t_send - self.t_prev_send) * 1e3)
        self.t_prev_send = t_send

    def step_done(self, busy_ms: float, chunk_step: bool) -> None:
        self.busy.append(busy_ms)
        if chunk_step:
            self.chunk_busy.append(busy_ms)
        self.frames += 1
        now = time.perf_counter()
        if now - self.start >= 1.0:
            self._log(now)
            self._restart(now)

    def _log(self, now: float) -> None:
        # send-gap max is how long the OSC loop sat on one goal: the visible hitch.
        logger.info(
            "loop fps: %.2f target: %.2f busy avg/max: %.1f/%.1f ms "
            "(chunk-step avg: %.1f ms) send-gap avg/max: %.1f/%.1f ms "
            "prefetch-wait avg/max: %.1f/%.1f ms",
            self.frames / (now - self.start), self.fps, _mean(self.busy), _peak(self.busy),
            _mean(self.chunk_busy), _mean(self.send_gap), _peak(self.send_gap),
            _mean(self.wait), _peak(self.wait),
        )


# ---------------------------------------------------------------------------
# Latency debug (--viz-latency-debug)
# ---------------------------------------------------------------------------

def _note_first_send(lat: dict, controller, target: np.ndarray, t_send: float,
                     events: list[dict]) -> None:
    """Where the arm was when a chunk's first goal went out, and where that goal landed.

    drift_mm: how far the arm moved between the chunk's observation and that send.
    backtrack_mm: how far the goal sits BEHIND the arm along its motion (> 0 pulls it back).
    """
    goal = (getattr(controller, "_last_osc_goal", None) or {}).get(_ARM_KEY)
    sent_on = (getattr(controller, "_last_osc_anchor", None) or {}).get(_ARM_KEY)
    lat["obs_to_send_ms"] = (t_send - lat["t_obs"]) * 1e3
    lat["target_pos"] = np.asarray(target[:3], dtype=np.float64)
    lat["send_pos"] = None if sent_on is None else np.asarray(sent_on[0], dtype=np.float64)
    lat["goal_pos"] = None if goal is None else np.asarray(goal[0], dtype=np.float64)
    if lat["send_pos"] is not None:
        moved = lat["send_pos"] - lat["anchor_pos"]
        lat["drift_mm"] = float(np.linalg.norm(moved)) * 1e3
        if lat["goal_pos"] is not None and lat["drift_mm"] > 0.5:
            u = moved / np.linalg.norm(moved)
            lat["backtrack_mm"] = float(-(lat["goal_pos"] - lat["send_pos"]) @ u) * 1e3
    events.append(lat)


def _latency_summary(events: list[dict]) -> dict:
    """Per-episode numbers for the episode row; mm and ms."""
    values = {k: np.array([e[k] for e in events if e.get(k) is not None], dtype=np.float64)
              for k in ("base_ms", "residual_ms", "total_ms", "obs_to_send_ms", "drift_mm",
                        "backtrack_mm")}
    out = {}
    for k, v in values.items():
        if len(v):
            out[f"{k}_median"] = round(float(np.median(v)), 2)
            out[f"{k}_max"] = round(float(np.max(v)), 2)
    if len(values["backtrack_mm"]):
        out["chunks_commanded_backwards"] = int(np.sum(values["backtrack_mm"] > 1.0))
        out["chunks"] = len(events)
    return out


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def _drain(pending: Future | None, pool: ThreadPoolExecutor) -> None:
    """The in-flight pass holds the policies and may still be writing a dump npz."""
    if pending is not None:
        try:
            pending.result(timeout=5.0)
        except Exception:
            logger.exception("in-flight inference failed during teardown")
    pool.shutdown(wait=True)


def _run_episode(
    controller,
    base_policy,
    residual: ResidualPolicy | None,
    dataset: LeRobotDataset | None,
    episode_time_s: float | None,
    fps: float = 20.0,
    task: str = "",
    recorder: EpisodeRecorder | None = None,
    replaying: bool = False,
    proprio_frame: str = "world",
    sim_proprio_convention: bool = True,
    dump_dir: Path | None = None,
    video_dir: Path | None = None,
    video_cams: list[str] | None = None,
    video_stem: str = "episode",
    infer_lead: int = 1,
    ep: Episode | None = None,
    wrench: WrenchTrace | None = None,
    latency_debug: bool = False,
) -> None:
    """One episode: a chunk every _CHUNK_EXEC steps, one EE_DELTA goal per step.

    infer_lead > 1 starts a chunk's inference that many steps before its first
    action, on a worker, overlapping the chunk still executing; 1 infers on the
    step that executes it. episode_time_s None runs until the operator stops it.
    """
    planner = ChunkPlanner(controller, base_policy, residual, proprio_frame,
                           sim_proprio_convention, dump_dir, latency_debug)
    infer_lead = int(np.clip(infer_lead, 1, _CHUNK_EXEC))
    submit_at = _CHUNK_EXEC - infer_lead  # step within a chunk that submits the next one
    video = StepVideo(video_dir, video_stem, fps, video_cams or sorted(controller.cameras))
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")  # policies are stateful
    pending: Future | None = None
    cursor: ChunkCursor | None = None
    first_send: dict | None = None
    latency_events: list[dict] = []
    gains = (0.0, 0.0)  # (kp, kd) in effect when a step's observation is read
    chunks = 0
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    base_policy.reset()
    controller.cache_delta(np.zeros(3), np.zeros(3))  # the residual rides in the targets

    t_start = time.perf_counter()
    stats, pacer = LoopStats(fps), Pacer(fps)
    with _raw_terminal():
        try:
            for step in itertools.count():
                t_step = time.perf_counter()
                verdict = _verdict(t_step - t_start, episode_time_s)
                if verdict:
                    if verdict != "timeout":
                        print(f"\r\n{verdict}\r", flush=True)
                    if ep is not None:
                        ep.verdict, ep.success = verdict, verdict == "success"
                        ep.wall_time_s = t_step - t_start
                    break

                snap = Snapshot.read(controller, sim_proprio_convention, t_step)
                base_policy.observe(snap.obs)  # every step, as select_action queues it
                video.write(snap.obs, step)

                if cursor is None or cursor.done:
                    if pending is None:
                        chunk = planner.plan(base_policy.window(), snap, gains, chunks)
                    else:
                        t_wait = time.perf_counter()
                        chunk, pending = pending.result(), None
                        stats.wait.append((time.perf_counter() - t_wait) * 1e3)
                    chunks += 1
                    cursor = ChunkCursor(chunk, replaying)
                    if latency_debug:
                        first_send = dict(chunk.latency, step=step, anchor_pos=chunk.anchor_pos)
                    if recorder is not None:
                        _record_chunk(recorder, chunk)
                    if ep is not None:
                        ep.inferences = chunks

                # The delta is computed against this read, and send_action composes on it.
                kin_now = controller.read_kinematic_state()
                action = cursor.action(kin_now)
                t_send = time.perf_counter()
                controller.send_action(action, anchor=kin_now)
                stats.sent(t_send)
                if first_send is not None:
                    _note_first_send(first_send, controller, cursor.target, t_send, latency_events)
                    first_send = None
                if wrench is not None:
                    wrench.sample(controller)

                # Only after the goal is on the wire: inference stays off the read -> write path.
                if infer_lead > 1 and cursor.i == submit_at and pending is None:
                    pending = pool.submit(planner.plan, base_policy.window(), snap, gains, chunks)

                if recorder is not None:
                    _record_step(recorder, snap, cursor, action)
                if dataset is not None:
                    dataset.add_frame(_dataset_frame(snap.obs, action, task, controller.cameras))

                gains = (action["kp"], action["kd"])
                cursor.i += 1
                if ep is not None:
                    ep.steps = step + 1
                stats.step_done((time.perf_counter() - t_step) * 1e3, chunk_step=cursor.i == 1)
                pacer.wait()
        finally:
            _drain(pending, pool)
            video.close()
            if latency_debug:
                summary = _latency_summary(latency_events)
                logger.info("latency debug: %s", summary)
                if ep is not None:
                    ep.notes["latency"] = summary
                if recorder is not None:
                    recorder.latency_events = latency_events


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------

def _build_dataset(args, controller) -> LeRobotDataset:
    """A fresh LeRobotDataset at args.output_dir; each run has its own directory."""
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
    return LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=args.output_dir,
        robot_type=controller.name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(controller.cameras),
        batch_encoding_size=1,
        vcodec="auto",
        streaming_encoding=True,
        encoder_queue_maxsize=8,
        encoder_threads=2,
    )


def _save_viz(recorder: EpisodeRecorder, path: str, residual: ResidualPolicy | None,
              title: str, frame_stride: int, fps: float) -> None:
    """save_rollout_html (base only) or save_episode_html (residual), plus the extras."""
    if residual is None:
        save_rollout_html(recorder, path, title=f"base policy — {title}",
                          frame_stride=frame_stride, fps=fps)
    else:
        save_episode_html(recorder, path, title=f"residual — {title}",
                          frame_stride=frame_stride, fps=fps)
    stem = path.removesuffix(".html")
    if recorder.latency_events:
        save_latency_html(recorder.latency_events, stem + "_latency.html",
                          title=f"inference latency — {title}")
        print(f"saved latency timeline to {stem}_latency.html")
    if recorder.policy_pcd_events:
        centered = residual is not None and residual.center_on_eef
        save_policy_pcd_npz(recorder.policy_pcd_events, stem + "_policy_pcd.npz",
                            center_on_eef=centered, fps=fps)
        print(f"saved policy-input clouds to {stem}_policy_pcd.npz (plot with plot_policy_pcd.py)")


def _base_policy_dataset(base_policy: str | None) -> str | None:
    """`dataset.repo_id` from the train_config.json LeRobot writes beside pretrained_model/."""
    if not base_policy:
        return None
    here = Path(base_policy).expanduser()
    for candidate in (here, here.parent, here.parent.parent):
        try:
            repo_id = (json.loads((candidate / "train_config.json").read_text())
                       .get("dataset") or {}).get("repo_id")
        except Exception:
            continue
        if repo_id:
            return repo_id
    return None


def _policy_record(args, base_action_space: str) -> dict:
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
    """Every knob this runner resolved, including the normalisation contract the checkpoints share."""
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
        "residual": {k: fc.policy(f"residual.{k}") for k in (
            "chunk_exec", "horizon", "pos_scale_m", "rot_scale_rad", "gains_mag", "residual_mag",
            "residual_trans_mag", "residual_rot_mag", "res_pos_gain", "res_rot_gain")},
    }


def _environment_record(args, controller) -> dict:
    """rollout_common.environment, so the record matches the baselines' shape."""
    shim = argparse.Namespace(rig=controller.config.rig_profile, home_pose_name=args.home_pose_name,
                              home_q=args.home_q, dry_run=False)
    return environment(shim, controller)


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """What every episode of a run drives and writes to."""
    controller: object
    base_policy: object
    residual: ResidualPolicy | None
    run_dir: rr.RunDir
    record: rr.RunRecord
    forces: ForceLog
    dump_root: Path | None
    dataset: LeRobotDataset | None = None


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--zero-residual", action="store_true",
                        help="ablation: keep the residual's gains, zero its position, rotation "
                             "and gripper corrections (eval_distill's zero_residual)")
    parser.add_argument("--zero-residual-gains", action="store_true",
                        help="ablation: keep the residual's corrections, run the stock gains")
    parser.add_argument("--allow-plant-mismatch", action="store_true",
                        help="run a residual whose training plant's controller law differs "
                             "from config/control.yaml's (eval_distill refuses this too)")
    parser.add_argument("--proprio-frame", choices=("robot", "world"), default="world",
                        help="Frame for the residual proprio pose: 'robot' = raw franka_fk "
                             "(current behavior), 'world' = transformed to the world frame "
                             "the point cloud lives in")
    parser.add_argument("--raw-proprio", action="store_true",
                        help="A/B control: skip the sim-convention proprio correction "
                             "(45° flange-vs-body quat + 6.9 mm TCP-vs-site pos; see "
                             "env_wrapper.current_ee_pose) and feed the legacy raw "
                             "franka_fk pose to the residual policy")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda/cpu)")
    parser.add_argument("--viz-latency-debug", action="store_true",
                        help="per chunk: time base and residual inference, read the arm's "
                             "pose as inference starts, returns and its first goal goes "
                             "out, and draw them on the episode page")
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
    parser.add_argument("--home-pose-name", default=fc.default_home_pose_name(),
                        help=f"Name of a saved pose JSON in {_POSES_DIR} (overrides --home-q)")
    parser.add_argument("--home-q", nargs=7, type=float, default=None,
                        help="7 joint angles (rad) overriding the saved home pose")
    parser.add_argument("--home-gripper", type=float, default=fc.control("homing.gripper_norm"))
    parser.add_argument("--home-max-time-s", type=float, default=fc.control("homing.max_time_s"))
    parser.add_argument("--home-tol-rad", type=float, default=fc.control("homing.tol_rad"))

    # The run directory owns every output path, so there is no --output-dir.
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
    parser.add_argument("--replay-dataset", default=None,
                        help="HuggingFace id for the dataset to replay from")
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
    args.output_dir = None  # _build_dataset's root, set once the run directory exists
    return args


def _home_kwargs(args) -> dict:
    """controller.home()'s arguments: --home-q, else the saved pose and its gripper."""
    if args.home_q is not None:
        home_q, gripper = np.asarray(args.home_q, dtype=np.float64), args.home_gripper
    else:
        pose = fc.load_home_pose(args.home_pose_name)
        home_q = _default_home_q(args.home_pose_name)
        gripper = float(pose.get("gripper", args.home_gripper))
    return dict(home_q_left=None, home_q_right=home_q, gripper_norm=gripper,
                max_time_s=args.home_max_time_s, tol_rad=args.home_tol_rad)


def _home(controller, home_kwargs: dict) -> bool:
    homed = bool(controller.home(**home_kwargs))
    if not homed:
        logger.warning("homing did not converge; proceeding anyway")
    return homed


def _load_base(args, controller):
    """The base policy (or a recording replayed as one) and its verified action space."""
    if args.replay_dataset is None:
        print(f"attempting to start base policy: {args.base_policy}")
        base = BasePolicy(args.base_policy, device=args.device,
                          amp=args.base_amp, compile_mode=args.base_compile)
        print("base policy started!")
    else:
        print(f"attempting to fetch replay dataset: {args.replay_dataset}")
        base = Trajectory(args.replay_dataset, device=args.device)
        print("replay dataset found!")
    return base, _check_base(base, controller, args)


def _load_residual(args) -> ResidualPolicy | None:
    if args.no_residual:
        print("residual policy disabled (--no-residual)")
        return None
    print(f"attempting to start residual policy: {args.residual_policy}")
    residual = ResidualPolicy(args.residual_policy, device=args.device)
    residual.zero_residual = args.zero_residual
    residual.zero_gains = args.zero_residual_gains
    print("residual policy started"
          + (" (position/rotation/gripper corrections zeroed)" if args.zero_residual else "")
          + (" (gains zeroed: stock controller)" if args.zero_residual_gains else ""))
    _check_teacher(residual, args.allow_plant_mismatch)
    return residual


def _open_dump_dir(args, residual: ResidualPolicy | None) -> Path | None:
    if not args.dump_obs_dir:
        return None
    if residual is None:
        print("--dump-obs-dir ignored: residual_obs only exists with a residual policy")
        return None
    root = Path(args.dump_obs_dir).expanduser() / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps({
        "residual_policy": str(Path(args.residual_policy).resolve()),
        "residual_policy_sha256": rr.sha256(args.residual_policy),
        "base_policy": args.base_policy or args.replay_dataset,
        "proprio_frame": args.proprio_frame,
        "raw_proprio": bool(args.raw_proprio),
        "fps": args.fps,
        "argv": sys.argv,
    }, indent=2))
    print(f"dumping residual obs bundles to {root}")
    return root


def _open_run(args, controller, base_action_space: str) -> tuple[rr.RunDir, rr.RunRecord]:
    """The run directory and its manifest; fills in the output defaults that depend on it."""
    train_dataset = rr.resolve_train_dataset(
        args.train_dataset, _base_policy_dataset(args.base_policy), None)
    run_dir = rr.RunDir(train_dataset["repo_id"], METHOD, root=args.outputs_root or rr.DEFAULT_ROOT)
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
    record.set("outputs", force_profiles=str(run_dir.force_profiles_path))
    return run_dir, record


def _episode(args, run: Run, ep_idx: int, homed: bool) -> None:
    print(f"\r\nepisode {ep_idx + 1}/{args.num_episodes}: place the scene, "
          f"then press RIGHT ARROW to start\r", flush=True)
    _wait_for_right_arrow()
    print(f"\r\nrunning ({args.episode_time_s:.0f}s max). "
          f"RIGHT = success, LEFT = failure, Ctrl-C = abort\r", flush=True)
    ep = Episode(episode=ep_idx, homed=homed, started_at=rr.stamp(), exec_fps=float(args.fps))
    recorder = EpisodeRecorder() if args.viz_dir else None
    wrench = WrenchTrace(_ARM_KEY)
    t0 = time.perf_counter()
    try:
        _run_episode(
            run.controller, run.base_policy, run.residual,
            dataset=run.dataset,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            task=args.task,
            recorder=recorder,
            replaying=args.replay_dataset is not None,
            proprio_frame=args.proprio_frame,
            sim_proprio_convention=not args.raw_proprio,
            dump_dir=run.dump_root / f"ep{ep_idx:03d}" if run.dump_root else None,
            video_dir=run.run_dir.video_dir if args.save_videos else None,
            video_cams=args.video_cams,
            video_stem=f"episode_{ep_idx:03d}",
            infer_lead=args.infer_lead,
            ep=ep,
            wrench=wrench,
            latency_debug=args.viz_latency_debug,
        )
    finally:
        _file_episode(args, run, ep, recorder, wrench, time.perf_counter() - t0)
    print(f"\r\nepisode {ep_idx}: {ep.verdict.upper()} in {ep.wall_time_s:.2f}s, "
          f"{ep.steps} steps{force_note(ep.ee_force_n)}\r", flush=True)
    if run.dataset is not None:
        run.dataset.save_episode()


def _file_episode(args, run: Run, ep: Episode, recorder: EpisodeRecorder | None,
                  wrench: WrenchTrace, elapsed_s: float) -> None:
    """Complete and record the episode row, however the episode ended."""
    ep.wall_time_s = ep.wall_time_s or elapsed_s
    ep.ended_at = rr.stamp()
    ep.verdict = ep.verdict or "incomplete"  # ended by neither the operator nor the clock
    if ep.wall_time_s > 0:
        ep.achieved_fps = round(ep.steps / ep.wall_time_s, 2)
    ep.ee_force_n = run.forces.add_trace(ep.episode, wrench)
    if run.dataset is not None:
        ep.frames_recorded = frames_in_progress(run.dataset)
        ep.dataset_episode_index = run.dataset.num_episodes
    if recorder is not None and len(recorder) > 0:
        viz_path = os.path.join(args.viz_dir, f"episode_{ep.episode:03d}.html")
        print(f"saving visualization to {viz_path}...")
        _save_viz(recorder, viz_path, run.residual, f"episode {ep.episode} — {args.task}",
                  args.viz_stride, args.fps)
        ep.notes["viz"] = os.path.basename(viz_path)
    run.record.add_episode(ep)


def _close_dataset(args, run: Run, encoder: VideoEncodingManager | None) -> None:
    if encoder is not None:
        encoder.__exit__(None, None, None)
    run.dataset.finalize()
    run.record.set("outputs", dataset={
        "recorded": True, "repo_id": args.repo_id,
        **rr.describe_lerobot_dataset(run.run_dir.dataset_dir),
    })
    if args.push_to_hub:
        try:
            run.dataset.push_to_hub()
            run.record.set("outputs", pushed_to_hub=True)
        except Exception:
            logger.exception("push_to_hub failed; dataset is on disk at %s", run.run_dir.dataset_dir)
            run.record.set("outputs", pushed_to_hub=False)


def _run_episodes(args, run: Run, home_kwargs: dict) -> None:
    encoder = None
    status, reason = "completed", None
    try:
        if not args.no_record:
            run.dataset = _build_dataset(args, run.controller)
            encoder = VideoEncodingManager(run.dataset)
            encoder.__enter__()
        recorded = run.dataset is not None
        run.record.set("outputs", dataset={
            "recorded": recorded,
            "repo_id": args.repo_id if recorded else None,
            "path": str(run.run_dir.dataset_dir) if recorded else None,
            "fps": args.fps,
        })
        print("homing...")
        homed = _home(run.controller, home_kwargs)
        for ep_idx in range(args.num_episodes):
            _episode(args, run, ep_idx, homed)
            if ep_idx < args.num_episodes - 1:
                print("resetting environment — homing arm before next episode...")
                homed = _home(run.controller, home_kwargs)
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt at the robot"
        print("\r\ninterrupted\r", flush=True)
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if run.dataset is not None:
            _close_dataset(args, run, encoder)
        if args.save_videos and run.run_dir.video_dir.is_dir():
            run.record.set("outputs", videos=sorted(v.name for v in run.run_dir.video_dir.glob("*.mp4")))
        run.record.finish(status, reason)
        print(f"\r\nrun written to {run.run_dir.path}\r", flush=True)


def main() -> None:
    args = _parse_args()
    if Path(args.residual_policy).suffix == ".zip":
        # A different base, loop and record entirely; nothing below applies.
        import reach_residual
        reach_residual.run(args.residual_policy, args.num_episodes, args.seed, args.arm,
                           args.policy, args.device, args.out, args.viz_stride, args.no_viz)
        return
    logging.basicConfig(level=logging.INFO, force=True)
    home_kwargs = _home_kwargs(args)

    print("attempting connection to robot...")
    controller = env_wrapper.start_controller(rig=args.rig)
    print(f"robot initialized: {controller.config.rig_profile} -> physical arm "
          f"{controller.config.arm_name(_ARM_KEY)!r} at {controller.config.r_robot_ip}, "
          f"cameras {sorted(controller.cameras)}")
    try:
        base_policy, base_action_space = _load_base(args, controller)
        residual = _load_residual(args)
        ChunkPlanner(controller, base_policy, residual, args.proprio_frame,
                     not args.raw_proprio).warmup()
        dump_root = _open_dump_dir(args, residual)
        run_dir, record = _open_run(args, controller, base_action_space)
        run = Run(controller, base_policy, residual, run_dir, record,
                  ForceLog(run_dir.force_profiles_path), dump_root)
        _run_episodes(args, run, home_kwargs)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
