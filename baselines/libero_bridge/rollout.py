#!/usr/bin/env python3
"""Roll a trained SAIL or B-Spline policy out on LIBERO, in simulation.

    ./scripts/libero_rollout.sh --backend sail --start-server --ckpt <CKPT.pth> --task 9

or, by hand, the policy in its own venv and this in multi-fast/.venv:

    multi-fast/.venv/bin/python -m baselines.libero_bridge.rollout \
        --backend sail --task <TASK> --num-episodes 20

The sim counterpart of `sail_bridge/rollout.py` and `bspline_bridge/rollout.py`.
It is a separate loop rather than a `--sim` flag on those because the clock
differs in kind: on the arm a plan runs against the wall clock and inference
latency is whatever it is, while here the only clock is the env's own step
counter. Holding the simulated latency fixed -- SAIL's `inf_delay`, which is
upstream's own model precisely because upstream evaluates in sim -- is what
makes a success rate reproducible rather than a measurement of this GPU.

Both backends drive the env through absolute pose targets, the space their
converted datasets are in. `sim_env.SimTask.action` inverts each target into the
normalised OSC delta LIBERO takes, through multi-fast's own inverse of the
relabeler that wrote the training targets, so the executor and the converter
cannot drift.

Episode time is reported in SIMULATED seconds (the sum of step periods), which is
what makes a time-to-success comparable between methods and between machines.
It is written to `wall_time_s` so `scripts/rollout_summary.py` reads it unchanged;
the real elapsed time is recorded beside it as `clock_time_s`.

The wrist force/torque sensor is read after every env step, as multi-fast's
`eval_fast.py` reads it. The per-step vectors go to `force_profiles.npz` in the
run directory and each episode's |F| statistics to its `ee_force_n` entry.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "franka_config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import franka_config as fc  # noqa: E402

from baselines import run_record as rr  # noqa: E402
from baselines.force_log import ForceLog  # noqa: E402
from baselines.bspline_bridge.spline_plan import (  # noqa: E402
    GRIPPER_INDEX, POSE_DIM as BSPLINE_POSE_DIM, SplinePlanner, decode_action,
)
from baselines.libero_bridge import sim_env  # noqa: E402
from baselines.policy_math import lead_goal, slowdown_mode, tracking_error_low  # noqa: E402
from baselines.zmq_client import PolicyClient  # noqa: E402

logger = logging.getLogger("baselines.libero.rollout")

BACKENDS = ("sail", "bspline", "pi05")
# The two that live behind a ZMQ policy server in their own venv. pi05 is
# multi-fast's base policy and loads in THIS process -- openpi is already in
# multi-fast/.venv, which is the interpreter this loop runs in.
SERVED = ("sail", "bspline")


@dataclass
class SimEpisode:
    """One rollout attempt, as written to episodes.jsonl.

    `success` is the env's own `done`, not an operator verdict: LIBERO scores the
    task itself, which is the whole reason the comparison is worth running here.
    """
    episode: int
    init_state: int = 0
    success: bool = False
    verdict: str | None = None
    wall_time_s: float = 0.0        # SIMULATED seconds; see the module docstring
    sim_time_s: float = 0.0
    clock_time_s: float = 0.0
    started_at: str | None = None
    ended_at: str | None = None
    steps: int = 0
    inferences: int = 0
    slow_steps: int = 0
    guided_inferences: int = 0
    # |F| at the wrist over the episode's steps: mean, median, p95, max.
    ee_force_n: dict | None = None
    notes: dict = field(default_factory=dict)


class Stepper:
    """Drives the env, owns the episode's clock, and keeps the frame stack.

    One observation per env STEP, which is what upstream's FrameStackWrapper
    gives the policy: consecutive control steps, not consecutive inferences.
    """

    def __init__(self, task: sim_env.SimTask, shapes: dict, frame_stack: int,
                 max_steps: int, video_stride: int = 0) -> None:
        self.task = task
        self.shapes = shapes
        self.control_freq = float(task.env.env.control_freq)
        self.max_time = int(max_steps) / self.control_freq
        self.history: deque = deque(maxlen=max(1, int(frame_stack)))
        self.video_stride = int(video_stride)
        self.video: list[np.ndarray] = []
        # One (3,) force and torque per env step, read after the step.
        self.ee_force: list[np.ndarray] = []
        self.ee_torque: list[np.ndarray] = []
        self.step_times: list[float] = []
        self.raw: dict = {}
        self.steps = 0
        self.sim_time = 0.0
        self.done = False
        self._next_frame = 0.0

    def begin(self, episode: int, state=None) -> None:
        self.raw = self.task.start(episode, state=state)
        self.steps = 0
        self.sim_time = 0.0
        self.done = False
        self._next_frame = 0.0
        self.video.clear()
        self.step_times.clear()
        self.ee_force.clear()
        self.ee_torque.clear()
        self.history.clear()
        frame = self.task.observe(self.raw, self.shapes)
        # Seeded with copies, as the wrapper seeds its stack on reset.
        for _ in range(self.history.maxlen):
            self.history.append(frame)
        self._capture()

    @property
    def finished(self) -> bool:
        # Rounded so a sum of 1/hz steps lands exactly on the budget.
        return self.done or round(self.sim_time, 6) >= self.max_time

    @property
    def measured(self) -> tuple[np.ndarray, np.ndarray]:
        """(grip-site position, wrist-body quaternion xyzw) -- the frames the
        converted targets are expressed in."""
        return (np.asarray(self.raw["robot0_eef_pos"], dtype=np.float64),
                np.asarray(self.raw["robot0_eef_quat"], dtype=np.float64))

    def _capture(self) -> None:
        if self.video_stride and round(self.sim_time - self._next_frame, 6) >= 0:
            # Vertical flip only: undoes the mirrored render and leaves the
            # frame the right way round for a human, unlike the policy's 180.
            self.video.append(np.ascontiguousarray(
                self.raw["agentview_image"][::-1]).astype(np.uint8))
            self._next_frame += self.video_stride / self.control_freq

    def send(self, pos, rotvec, gripper: float, hz: float) -> None:
        """Command one absolute pose target for 1/hz simulated seconds.

        One env step whose control period is 1/hz, which is how SAIL's own sim
        (its patched robosuite) plays action rows faster than they were recorded.
        The delta is recomputed against the pose measured that step, the same
        semantics as EE_POS on the arm.
        """
        if self.finished:
            return
        self._advance(self.task.action(pos, rotvec, gripper), 1.0 / hz)

    def send_raw(self, action) -> None:
        """One env step with a LIBERO action as given.

        pi0.5 emits LIBERO's own action space -- seven normalised OSC deltas --
        so unlike the two baselines there is no absolute pose to invert.
        """
        if self.finished:
            return
        self._advance(action)

    def _advance(self, action, dt: float | None = None) -> bool:
        """One env step (of `dt` s, else the env's own period) and its bookkeeping;
        True when the task succeeded."""
        self.raw, _, done, _ = self.task.step(action, dt)
        self.steps += 1
        self.sim_time += dt or 1.0 / self.control_freq
        self.step_times.append(self.sim_time)
        force, torque = self.task.wrench()
        self.ee_force.append(force)
        self.ee_torque.append(torque)
        self.history.append(self.task.observe(self.raw, self.shapes))
        self._capture()
        if done:
            self.done = True
        return bool(done)


def sim_osc(backend: str, args) -> dict:
    """The backend's execution controller, as `sim_env.osc_config` arguments.

    Empty is the stock controller the demonstrations were recorded under. pi05
    always runs that: its normalised deltas were learned under it.
    """
    flags = {"kp": args.osc_kp, "damping_ratio": args.osc_damping_ratio}
    if backend not in SERVED:
        if any(v is not None for v in flags.values()):
            raise SystemExit("pi05's deltas were learned under the stock controller, "
                             "so it runs that; drop the --osc-* flags")
        return {}
    out = dict(fc.policy(f"baselines.{backend}.sim_osc") or {})
    out.update({k: v for k, v in flags.items() if v is not None})
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# SAIL
# ---------------------------------------------------------------------------

def sail_settings(meta: dict, args, fast_fps: float) -> dict:
    horizon = int(meta["action_horizon"])
    s = {
        "precision": bool(meta["precision_column"]) and not args.no_precision,
        "eag": bool(meta["guided"]) and bool(meta["fac_enabled"]) and not args.no_eag,
        "fac_horizon": int(meta["fac_horizon"]),
        "action_horizon": horizon,
        "inf_delay": int(fc.policy("baselines.sail.inf_delay")),
        "execute_n": int(fc.policy("baselines.sail.execute_n_actions")),
        "window": int(fc.policy("baselines.sail.slowdown_window_size")),
        "pos_teb": float(fc.policy("baselines.sail.pos_teb")),
        "ori_teb": float(fc.policy("baselines.sail.ori_teb")),
        "fast_fps": fast_fps,
        "slow_fps": float(args.slow_fps or fc.policy("baselines.exec.slow_fps")),
    }
    if horizon < s["inf_delay"] + s["execute_n"]:
        raise SystemExit(
            f"checkpoint's action_horizon is {horizon} but the receding horizon "
            f"consumes inf_delay + execute_n_actions = {s['inf_delay'] + s['execute_n']} "
            "rows per inference. Lower them in config/policy.yaml (baselines.sail).")
    return s


def sail_episode(stepper: Stepper, client: PolicyClient, s: dict, ep: SimEpisode) -> None:
    client.reset()
    prev_chunk: np.ndarray | None = None
    prev_index = 0
    ref: np.ndarray | None = None
    executed: list[np.ndarray] = []

    def dispatch(src: np.ndarray, idx: int) -> None:
        row = src[idx]
        # Appended BEFORE the window check, as upstream does: the current row
        # sits in its own left window.
        executed.append(row)
        slow = s["precision"] and slowdown_mode(executed, row, src[idx:], s["window"])
        a = row[:-1] if s["precision"] else row
        if slow:
            ep.slow_steps += 1
        stepper.send(a[:3], a[3:6], float(a[6]), s["slow_fps"] if slow else s["fast_fps"])

    while not stepper.finished:
        frames = list(stepper.history)

        guide = None
        if s["eag"] and ref is not None and len(ref):
            pos, quat = stepper.measured
            if tracking_error_low(pos, quat, ref[0][:3], ref[0][3:6],
                                  s["pos_teb"], s["ori_teb"]):
                guide = ref
                ep.guided_inferences += 1

        # The old plan carries on for the rows inference is modelled to cost.
        # Upstream's fixed count IS the latency model in sim, and its row 0 is
        # the pose the observation was taken at, so a new chunk is entered at
        # the row matching what went out since.
        entry = 0
        if prev_chunk is not None:
            while (prev_index < len(prev_chunk) and entry < s["inf_delay"]
                   and not stepper.finished):
                dispatch(prev_chunk, prev_index)
                prev_index += 1
                entry += 1
        if stepper.finished:
            break

        t0 = time.perf_counter()
        rep = client.request({"obs": frames, "guide_actions": guide})
        if "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        chunk = np.asarray(rep["chunk"], dtype=np.float64)
        ep.inferences += 1
        ep.notes["inference_s"] = round(
            ep.notes.get("inference_s", 0.0) + (time.perf_counter() - t0), 3)

        index = entry
        for _ in range(s["execute_n"]):
            if index >= len(chunk) or stepper.finished:
                break
            dispatch(chunk, index)
            index += 1
        if index == entry:
            logger.warning("chunk of %d rows exhausted at entry %d; nothing executed",
                           len(chunk), entry)
            break
        prev_chunk, prev_index = chunk, index
        ref = chunk[index:index + s["fac_horizon"]] if s["eag"] else None


# ---------------------------------------------------------------------------
# B-Spline
# ---------------------------------------------------------------------------

def bspline_planner_kwargs(meta: dict, args, fast_fps: float) -> dict:
    return dict(
        degree=int(meta.get("degree") or args.degree),
        n_obs_steps=int(meta.get("n_obs_steps")
                        or fc.policy("baselines.bspline.n_obs_steps")),
        # One observation per 1/fast_fps step; the policy's frames stay one demo frame apart.
        obs_stride=int(round(fast_fps / fc.control_fps())),
        # Knot index units per SIMULATED second. The knots count demo frames and
        # the demos are one frame per env step, so this is the env's control
        # rate -- t then advances one knot per step at speed 1.
        origin_time_scale=float(args.origin_time_scale
                                if args.origin_time_scale is not None
                                else fc.control_fps()),
        speed_up_times=float(args.speed_up_times
                             if args.speed_up_times is not None
                             else fc.policy("baselines.bspline.speed_up_times")),
        predict_before_end=float(fc.policy("baselines.bspline.predict_before_end")),
        time_align_error_threshold=float(
            fc.policy("baselines.bspline.time_align_error_threshold")),
        time_align_larger_t=fc.policy("baselines.bspline.time_align_larger_t"),
        # No sim time passes during a synchronous inference, so a new plan
        # begins exactly at the observation it was predicted from and there is
        # nothing to stitch onto. Upstream's search would run on an empty window
        # and return its `inf` sentinel every replan -- the right start (min_t)
        # reached through a meaningless alignment. --time-align restores it for
        # whoever models inference latency in sim steps.
        disable_time_align=not bool(args.time_align),
        restart_on_time_align_error=False,
        consider_gripper_during_align=False,
        gripper_slowdown_enabled=bool(fc.policy("baselines.bspline.gripper_slowdown_enabled")),
        gripper_slowdown_threshold=float(fc.policy("baselines.bspline.gripper_slowdown_threshold")),
        gripper_slowdown_steps=int(fc.policy("baselines.bspline.gripper_slowdown_steps")),
        gripper_index=GRIPPER_INDEX,
        compare_dim=BSPLINE_POSE_DIM,
    )


def bspline_episode(stepper: Stepper, client: PolicyClient, planner_kwargs: dict,
                    fast_fps: float, ep: SimEpisode) -> None:
    client.reset()
    dt = 1.0 / fast_fps
    # Holds the arm's lag behind a sped-up plan at the demos' 1x lag, as upstream's servo scaling does.
    c = stepper.task.controller
    lead = np.asarray(c.kd) / np.asarray(c.kp) * (1.0 - 1.0 / planner_kwargs["speed_up_times"])
    if not fc.policy("baselines.bspline.goal_lead"):
        lead = np.zeros(6)
    # Synchronous, on the sim clock: an inference that overlapped stepping would
    # make the plan's phase depend on how fast this GPU is.
    planner = SplinePlanner(client, clock=lambda: stepper.sim_time,
                            synchronous=True, **planner_kwargs)
    try:
        pos, quat = stepper.measured
        # Hold where the arm already is until the first plan lands; a zero goal
        # would command the base-frame origin.
        goal = (pos, Rotation.from_quat(quat).as_rotvec(), -1.0)
        while not stepper.finished:
            sample = planner.step(stepper.history[-1])
            if sample is not None:
                p, q, grip = decode_action(sample)
                p, q = lead_goal(p, q, *decode_action(planner.peek(dt))[:2], dt, lead)
                goal = (p, Rotation.from_quat(q).as_rotvec(), grip)
            stepper.send(*goal, fast_fps)
    finally:
        ep.inferences = planner.plans
        finite = [e for e in planner.align_errors if np.isfinite(e)]
        if finite:
            ep.notes["time_align_error_max"] = round(float(max(finite)), 6)
        if len(finite) != len(planner.align_errors):
            # Not JSON, and not a real error -- an empty alignment window.
            ep.notes["time_align_empty"] = len(planner.align_errors) - len(finite)
        planner.close()


# ---------------------------------------------------------------------------
# pi0.5 (multi-fast's base policy)
# ---------------------------------------------------------------------------

def pi05_episode(stepper: Stepper, policy, chunk_size: int, ep: SimEpisode) -> None:
    """Replan every `chunk_size` steps and execute the chunk, which is what
    cfg/libero/fast_libero_90.yaml's chunk_size means and what eval_pi05.py does.

    No receding horizon and no latency model: unlike SAIL this policy is not
    predicting its own future poses, and upstream's own LIBERO evaluation
    replans on a fixed stride.
    """
    from baselines.libero_bridge import pi05 as pi05_mod

    while not stepper.finished:
        chunk = np.asarray(policy(pi05_mod.observation(stepper.raw), return_numpy=True))
        ep.inferences += 1
        rows = chunk[0] if chunk.ndim == 3 else chunk
        for row in rows[:chunk_size]:
            if stepper.finished:
                break
            stepper.send_raw(np.asarray(row, dtype=np.float64))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--suite", default="libero_90",
                   help="the suite --task indexes into; evaluate.py passes the prep dir's")
    p.add_argument("--task", required=True,
                   help="the task's index in --suite (9), its tag (task_9), or its LIBERO name")
    p.add_argument("--num-episodes", type=int, default=20,
                   help="init states are taken in order from the suite's own set")
    p.add_argument("--first-episode", type=int, default=0, help="init-state offset")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=None, help="default: the backend's in policy.yaml")
    p.add_argument("--ckpt", default=None, help="recorded for provenance; the server holds it")
    p.add_argument("--train-dataset", default=None,
                   help="default: the id the checkpoint's HDF5 carries")
    p.add_argument("--sweep-id", default=None,
                   help="tags this run as part of a sweep; evaluate.py sets it so "
                        "every rollout of one invocation can be summarised together")
    p.add_argument("--output-root", type=Path, default=None,
                   help=f"default {rr.DEFAULT_ROOT}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="default: the suite's own budget")
    p.add_argument("--render-resolution", type=int, default=None,
                   help="what the env renders at; default is the backend's own -- the "
                        "converter's for the baselines, multi-fast's for pi05")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-stride", type=int, default=2)
    p.add_argument("--no-precision", action="store_true", help="sail: ignore the precision label")
    p.add_argument("--no-eag", action="store_true", help="sail: disable error-adaptive guidance")
    p.add_argument("--fast-fps", type=float, default=None,
                   help="rate goals are sent at: SAIL's ordinary rows, every B-Spline "
                        "sample; default baselines.exec.fast_fps. Equal to the env's "
                        "control rate is the demonstrations' own speed")
    p.add_argument("--slow-fps", type=float, default=None,
                   help="sail: rate precision-labelled rows are played at; default "
                        "baselines.exec.slow_fps")
    p.add_argument("--osc-kp", type=float, default=None,
                   help="sail/bspline: OSC stiffness; default the backend's sim_osc in "
                        "config/policy.yaml, else robosuite's 150")
    p.add_argument("--osc-damping-ratio", type=float, default=None,
                   help="sail/bspline: OSC damping ratio; default as --osc-kp, else 1.0")
    p.add_argument("--speed-up-times", type=float, default=None, help="bspline")
    p.add_argument("--origin-time-scale", type=float, default=None, help="bspline")
    p.add_argument("--degree", type=int, default=3, help="bspline fallback")
    p.add_argument("--time-align", action="store_true",
                   help="bspline: stitch a new plan onto the old one. Off in sim; see "
                        "bspline_planner_kwargs")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="pi05: steps executed per replan; default the value in "
                        "multi-fast/cfg/libero/fast_libero_90.yaml")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    osc = sim_osc(args.backend, args)

    client = None
    policy = None
    port = None
    chunk_size = 0
    if args.backend in SERVED:
        port = args.port or int(fc.policy(f"baselines.zmq.{args.backend}_port"))
        client = PolicyClient(port, int(fc.policy("baselines.zmq.recv_timeout_ms")), args.host)
        meta = client.meta()
        if meta.get("backend") != args.backend:
            p.error(f"server on port {port} is a {meta.get('backend')!r} server, "
                    f"not {args.backend}")
        shapes = dict(meta["obs_key_shapes"])
        resolution = int(args.render_resolution or sim_env.RENDER_RESOLUTION)
        # One policy per task here, so a server reporting another task's dataset
        # is a stale one still holding the port. libero_90's files predate index
        # tags and carry the task's name instead.
        i = sim_env.resolve_task(args.suite, args.task)
        name = sim_env.task_names(args.suite)[i]
        said = str(meta.get("train_dataset") or "")
        if said and said not in (f"{args.suite}/{sim_env.task_tag(i)}", f"{args.suite}/{name}"):
            p.error(f"the server on port {port} serves a policy trained on {said!r}, "
                    f"but this rollout is {args.suite} task {i} ({name}). Check --suite "
                    f"and --task against --ckpt; if they agree, another run's server is "
                    f"still holding the port -- kill it and retry "
                    f"(baselines/LIBERO_SIM.md, 'When a run fails').")
    else:
        from baselines.libero_bridge import pi05 as pi05_mod
        chunk_size = int(args.chunk_size or pi05_mod.CHUNK_SIZE)
        task_index = sim_env.resolve_task(args.suite, args.task)
        policy = pi05_mod.load(args.suite, task_index, chunk_size=chunk_size)
        meta = pi05_mod.meta(
            policy, chunk_size,
            args.train_dataset or f"{args.suite}/{sim_env.task_tag(task_index)}")
        # pi0.5 reads the raw observation through multi-fast's own translation,
        # so the Stepper builds no policy observation of its own.
        shapes = {}
        # Its own render resolution too -- see pi05.render_resolution.
        resolution = int(args.render_resolution or pi05_mod.render_resolution())
    logger.info("checkpoint: %s", json.dumps(meta, default=str)[:400])

    try:
        train_dataset = rr.resolve_train_dataset(
            args.train_dataset, meta.get("train_dataset"), meta.get("training_hdf5"))
    except ValueError as exc:
        p.error(str(exc))

    task = sim_env.SimTask(args.suite, args.task, seed=args.seed,
                           resolution=resolution, controller=osc)
    budget = int(args.max_steps or task.max_steps)
    report = sim_env.osc_report(task.controller)

    run_dir = rr.RunDir(train_dataset["repo_id"], args.backend,
                        root=args.output_root or rr.DEFAULT_ROOT)
    record = rr.RunRecord(run_dir, args.backend, train_dataset, sweep=args.sweep_id)
    served = args.backend in SERVED
    record.set("policy",
               checkpoint=rr.file_provenance(args.ckpt),
               server=({"host": args.host, "port": port, **{k: v for k, v in meta.items()}}
                       if served else {k: v for k, v in meta.items()}),
               control_mode="absolute_pose_target" if served else "osc_delta",
               control_mode_reason=(
                   "both converted datasets train on absolute poses; sim_env.action "
                   "inverts them into LIBERO's OSC delta" if served else
                   "pi0.5 emits LIBERO's own normalised OSC deltas, stepped as given"))
    record.set("environment",
               simulator="libero/robosuite",
               plant="stock",
               plant_reason="no plant or gripper overrides; "
                            "regenerate_libero_dataset.py recorded the demos under this model",
               controller=("stock" if not osc else "sim_osc"),
               controller_overrides=osc,
               osc=report,
               suite=args.suite, task=task.name, task_index=task.task_index,
               language=task.language, seed=args.seed,
               render_resolution=resolution,
               policy_image_shapes={k: list(v) for k, v in shapes.items() if k.endswith("_image")},
               control_freq=float(task.env.env.control_freq),
               max_steps=budget, settle_steps=sim_env.SETTLE_STEPS,
               init_states=task.n_init_states)

    fast_fps = float(args.fast_fps or fc.policy("baselines.exec.fast_fps"))
    settings = sail_settings(meta, args, fast_fps) if args.backend == "sail" else {}
    planner_kwargs = (bspline_planner_kwargs(meta, args, fast_fps)
                      if args.backend == "bspline" else {})
    params = {"sail": settings,
              "bspline": {**planner_kwargs, "fast_fps": fast_fps,
                          "goal_lead": bool(fc.policy("baselines.bspline.goal_lead"))},
              "pi05": {"chunk_size": chunk_size}}
    record.set("parameters", **params[args.backend])

    frame_stack = int(meta.get("frame_stack") or meta.get("n_obs_steps") or 1)
    stepper = Stepper(task, shapes, frame_stack, budget,
                      args.video_stride if args.save_video else 0)
    forces = ForceLog(run_dir.force_profiles_path, control_freq=stepper.control_freq)
    record.set("outputs", force_profiles=str(run_dir.force_profiles_path))

    logger.info("%s on %s/%s: %d episodes, budget %d steps (%.1f s of sim), "
                "osc kp %g damping %g step limit %g cm",
                args.backend, args.suite, task.name, args.num_episodes,
                budget, budget / stepper.control_freq,
                report["kp"][0], report["damping_ratio"][0], 100 * report["output_max"][0])

    status, reason = "completed", None
    successes = 0
    try:
        for i in range(args.num_episodes):
            init = (args.first_episode + i) % task.n_init_states
            ep = SimEpisode(episode=i, init_state=init, started_at=rr.stamp())
            t0 = time.perf_counter()
            stepper.begin(init)
            if args.backend == "sail":
                sail_episode(stepper, client, settings, ep)
            elif args.backend == "bspline":
                bspline_episode(stepper, client, planner_kwargs, fast_fps, ep)
            else:
                pi05_episode(stepper, policy, chunk_size, ep)
            ep.steps = stepper.steps
            ep.success = stepper.done
            ep.verdict = "success" if stepper.done else "timeout"
            ep.sim_time_s = round(stepper.sim_time, 3)
            ep.wall_time_s = ep.sim_time_s
            ep.clock_time_s = round(time.perf_counter() - t0, 2)
            ep.ended_at = rr.stamp()
            ep.ee_force_n = forces.add(i, stepper.ee_force, stepper.ee_torque, t=stepper.step_times)
            successes += int(ep.success)
            record.add_episode(ep)
            if args.save_video:
                write_video(run_dir.video_dir / f"{args.backend}_ep{i:03d}_{ep.verdict}.mp4",
                            stepper.video,
                            stepper.control_freq / max(1, args.video_stride))
            f = ep.ee_force_n or {}
            logger.info("ep %2d (init %2d): %-7s %3d steps  %5.2f s sim  %5.1f s clock  "
                        "|F| mean %5.1f p95 %6.1f max %6.1f N  [%d/%d = %.0f%%]",
                        i, init, ep.verdict, ep.steps, ep.sim_time_s, ep.clock_time_s,
                        f.get("mean", np.nan), f.get("p95", np.nan), f.get("max", np.nan),
                        successes, i + 1, 100 * successes / (i + 1))
    except KeyboardInterrupt:
        status, reason = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.finish(status, reason)
        task.close()
        if client is not None:
            client.close()
        summary = record.summary()
        force = summary["ee_force_n"]
        logger.info("success rate %s/%s = %s | median time-to-success %s s | %s",
                    summary["successes"], summary["episodes"],
                    "n/a" if summary["success_rate"] is None
                    else f"{100 * summary['success_rate']:.0f}%",
                    summary["median_time_to_success_s"],
                    "EE force not recorded" if force is None else
                    f"EE force avg / median / p95 / max: {force['mean']:.2f} / "
                    f"{force['median']:.2f} / {force['p95']:.2f} / {force['max']:.2f} N")
        logger.info("run written to %s", run_dir.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
