#!/usr/bin/env python3
"""Exercise both baseline rollout loops with no hardware and no conda env.

    python scripts/check_baseline_rollout_offline.py
    python scripts/check_baseline_rollout_offline.py --only sail

Two fakes stand in for the two halves that are not available here:

  * `FakeArm` duck-types SingleArmFranka. It holds a real joint configuration and
    responds to a goal with a damped-Jacobian step toward it, using this repo's
    own `zero_jacobian` -- so the (q -> FK pose) relationship the rollout reads
    back is self-consistent, which is what makes the lead monitor and the
    tracking-error check testable at all. It clips an EE_DELTA command to
    torque.delta.pos_max_m so the harness sees the truncation hardware would.
  * A real ZMQ REP server on a thread, answering the same protocol the two policy
    servers do with synthetic chunks and splines. The client path, the pickling,
    and the meta handshake are therefore the real ones.

What it cannot tell you: whether the arm tracks, and whether a real checkpoint's
actions are sane. Everything between those is covered.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import franka_config as fc  # noqa: E402

from baselines import rollout_common as rc  # noqa: E402
from baselines import run_record as rr  # noqa: E402
from baselines.bspline_bridge import rollout as bsp_rollout  # noqa: E402
from baselines.policy_math import apply_lead  # noqa: E402
from baselines.sail_bridge import rollout as sail_rollout  # noqa: E402
from baselines.sail_bridge.schedule import ScheduledPlan  # noqa: E402
from lerobot_robot_bimanual_franka import ControlMode  # noqa: E402
from lerobot_robot_bimanual_franka.ee_kinematics import eef_poses_from_qpos  # noqa: E402
from lerobot_robot_bimanual_franka.franka_jacobian import zero_jacobian  # noqa: E402

ALPHA = 0.3          # first-order tracking, matching reach's chunk_alpha
# The rig's real frame size (observation.image_height/width), deliberately NOT
# the policy's input size so the resize in rollout_common._images is exercised.
# Must stay realistic: the video encoder rejects frames below its minimum
# dimension, which is a property of the encoder, not of the rollout.
_IMG = (int(fc.control("observation.image_height")),
        int(fc.control("observation.image_width")))


class _Cam:
    def __init__(self, h, w):
        self.height, self.width = h, w


class FakeArm:
    """Duck-types SingleArmFranka for the parts a rollout touches."""

    name = "fake_single_arm"

    def __init__(self, control_mode: ControlMode, cams=("cam_2", "cam_5")) -> None:
        self.control_mode = control_mode
        self.k = rc.ARM_KEY
        self.active_arms = (self.k,)
        self.cameras = {c: _Cam(*_IMG) for c in cams}
        self.robot_manager = self
        self.q = fc.home_q(key=rc.ARM_KEY).astype(np.float64)
        self.gripper = 1.0
        self.sent: list[dict] = []
        self.send_times: list[float] = []
        self.anchored = 0        # goals sent with get_observation's state read as their anchor
        self.clipped = 0
        self.frames = 0
        self.freeze = False      # stop tracking: a plan that outruns the arm must not end the episode
        self.gripper_rate = None  # normalised range/s the fingers move at; None: at once
        self.gripper_target = self.gripper
        self.gripper_log: list[tuple[float, float]] = []   # (time, measured gripper)

    # -- lifecycle
    def connect(self):
        pass

    def disconnect(self):
        pass

    def home(self, home_q_left=None, home_q_right=None, **kw) -> bool:
        # Mirror the real signature: a bare home() must fail here too.
        if {"l": home_q_left, "r": home_q_right}[self.k] is None:
            raise TypeError(f"home() got no q for key {self.k!r}")
        self.q = np.asarray(home_q_right, dtype=np.float64).copy()
        self.gripper = float(kw.get("gripper_norm", 1.0))
        return True

    def current_kinematic_state_batch(self, arms):
        # Registered under the key PREFIX, not the physical arm; echoing back any
        # key would hide a wrong-key lookup.
        bad = [a for a in arms if a != self.k]
        if bad:
            raise KeyError(f"no driver for {bad}; registered: [{self.k!r}]")
        pos, quat = self._pose()
        return {self.k: (tuple(self.q), tuple(np.zeros(7)), None, tuple(pos),
                         tuple(quat), tuple(np.zeros(6)))}

    # -- observation / action
    def _pose(self):
        pos, quat = eef_poses_from_qpos(self.q[None])
        return pos[0], quat[0]

    @property
    def last_ee_wrench(self) -> dict:
        # |F| equals the number of goals sent so far, so a run's max is its step count.
        return {self.k: np.array([0.0, 0.0, float(len(self.sent)), 0.0, 0.0, 0.0])}

    def read_kinematic_state(self):
        return self.current_kinematic_state_batch([self.k])

    def get_observation(self) -> dict:
        # Paced like a 30 fps camera read; every pixel carries the frame counter.
        time.sleep(1.0 / 30.0)
        if self.gripper_rate is not None:
            step = self.gripper_rate / 30.0
            self.gripper += float(np.clip(self.gripper_target - self.gripper, -step, step))
        self.gripper_log.append((time.perf_counter(), self.gripper))
        self.frames += 1
        obs = {f"{self.k}_joint_{i + 1}": float(v) for i, v in enumerate(self.q)}
        obs[f"{self.k}_gripper"] = self.gripper
        for cam in self.cameras:
            obs[cam] = np.full((*_IMG, 3), self.frames % 256, dtype=np.uint8)
        return obs

    def send_action(self, action: dict, anchor=None) -> None:
        self.anchored += int(anchor is not None)
        self.sent.append(dict(action))
        self.send_times.append(time.perf_counter())
        pos, quat = self._pose()
        a = np.array([action[k] for k in rc.EE_ACTION_KEYS[:7]], dtype=np.float64)
        if self.control_mode is ControlMode.EE_DELTA:
            lim = float(fc.control("torque.delta.pos_max_m"))
            d = a[:3]
            if np.any(np.abs(d) > lim + 1e-12):
                self.clipped += 1
            goal_pos = pos + np.clip(d, -lim, lim)
            goal_rot = Rotation.from_quat(a[3:7]) * Rotation.from_quat(quat)
        else:
            goal_pos, goal_rot = a[:3], Rotation.from_quat(a[3:7])
        self.gripper_target = float(np.clip(action[f"{self.k}_gripper"], 0.0, 1.0))
        if self.gripper_rate is None:
            self.gripper = self.gripper_target
        if self.freeze:
            return
        err = np.concatenate([goal_pos - pos,
                              (goal_rot * Rotation.from_quat(quat).inv()).as_rotvec()])
        J = zero_jacobian(self.q, ee_pos_base=pos)
        self.q = self.q + ALPHA * (np.linalg.pinv(J) @ err)


# ---------------------------------------------------------------------------
# Fake policy servers (real ZMQ, synthetic payloads)
# ---------------------------------------------------------------------------

class _Server(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self._ctx = zmq.Context()
        self.sock = self._ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://*:{port}")
        self.sock.setsockopt(zmq.RCVTIMEO, 500)
        self.calls = 0
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                req = self.sock.recv_pyobj()
            except zmq.error.Again:
                continue
            try:
                self.sock.send_pyobj(self.handle(req))
            except Exception as exc:      # keep the socket in step
                self.sock.send_pyobj({"error": repr(exc)})

    def stop(self):
        self._stop = True


class FakeSail(_Server):
    """Chunks tagged so the receding-horizon bookkeeping is reconstructable.

    Row j of chunk k carries gripper = k + j/100, which passes through
    `ee_pos_action` untouched. Reading it back off the dispatched
    action dicts says exactly which (chunk, index) each goal came from.
    """

    def __init__(self, port, *, action_keys, act_dim, precision,
                 action_horizon=16, fac_horizon=4, guided=False, delay_s=0.0):
        super().__init__(port)
        self.delay_s = delay_s
        self.action_keys, self.act_dim = action_keys, act_dim
        self.precision, self.action_horizon = precision, action_horizon
        self.fac_horizon, self.guided = fac_horizon, guided
        self.chunk_id = -1
        self.guide_seen: list = []
        self.frames_seen: list = []
        self.frame_ids: list[list[int]] = []

    def handle(self, req):
        if "meta" in req:
            return {"backend": "sail", "action_keys": self.action_keys,
                    "act_dim": self.act_dim, "action_horizon": self.action_horizon,
                    "prediction_horizon": 2 * self.action_horizon,
                    "obs_key_shapes": {"cam_2_image": [3, 84, 84],
                                       "robot0_eef_pos": [3], "robot0_eef_quat": [4],
                                       "robot0_joint_pos": [7], "robot0_gripper_qpos": [1]},
                    "n_obs_steps": 2, "precision_column": self.precision,
                    "fac_enabled": self.guided, "fac_horizon": self.fac_horizon,
                    "guided": self.guided}
        if "reset" in req:
            self.chunk_id = -1
            return {}
        self.calls += 1
        self.chunk_id += 1
        self.guide_seen.append(req.get("guide_actions"))
        time.sleep(self.delay_s)
        absolute = not self.action_keys[0] == "actions"
        chunk = np.zeros((self.action_horizon, self.act_dim))
        # A list is consecutive frames, oldest first; the newest is the anchor.
        obs = req["obs"][-1] if isinstance(req["obs"], (list, tuple)) else req["obs"]
        self.frames_seen.append(len(req["obs"]) if isinstance(req["obs"], (list, tuple)) else 1)
        self.frame_ids.append([int(o["cam_2_image"][0, 0, 0]) for o in req["obs"]])
        base = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        for j in range(self.action_horizon):
            if absolute:
                chunk[j, :3] = base + np.array([0.002 * (j + 1), 0.0, 0.0])
                chunk[j, 3:6] = Rotation.from_quat(
                    np.asarray(obs["robot0_eef_quat"], dtype=np.float64)).as_rotvec()
            else:
                chunk[j, :3] = [0.002, 0.0, 0.0]
                chunk[j, 3:6] = 0.0
            chunk[j, 6] = self.chunk_id + j / 100.0        # the tag
            if self.precision:
                # Deterministic: label set on odd chunk ids, rows 2..4.
                chunk[j, 7] = 1.0 if (self.chunk_id % 2 == 1 and 2 <= j <= 4) else 0.0
        return {"chunk": chunk.astype(np.float32)}


class FakeBSpline(_Server):
    DEG, ACT, SPAN = 3, 10, 20      # 20 knot-index units == 1.0 s at 20 Hz

    def __init__(self, port, span=None, travel=0.03, first_knot=0.0, grip=(0.0, 1.0),
                 grip_dip=False):
        super().__init__(port)
        self.span = self.SPAN if span is None else span
        self.travel = float(travel)   # metres of x the plan sweeps
        self.first_knot = float(first_knot)   # where each plan's support starts, knot units
        self.grip = grip              # the gripper command's first and last control points
        self.grip_dip = grip_dip      # opening command crosses 0.5 up, back down, then up
        self.frames_seen: list[list[int]] = []

    def handle(self, req):
        if "meta" in req:
            return {"backend": "bspline", "act_dim": self.ACT, "degree": self.DEG,
                    "n_obs_steps": 2, "precision_column": False,
                    "action_format": "real_bimanual_base_rot6d",
                    "obs_key_shapes": {"cam_2_image": [3, 84, 84], "arm_pos": [3],
                                       "arm_quat": [4], "gripper_pos": [1]}}
        if "reset" in req:
            return {}
        self.calls += 1
        self.frames_seen.append([int(o["cam_2_image"][0, 0, 0]) for o in req["obs"]
                                 if "cam_2_image" in o])
        obs = req["obs"][-1] if isinstance(req["obs"], (list, tuple)) else req["obs"]
        return {"bspline": self.make(np.asarray(obs["arm_pos"], dtype=np.float64),
                                     np.asarray(obs["arm_quat"], dtype=np.float64))}

    def make(self, pos, quat):
        n_ctrl = 12
        interior = np.linspace(0, self.span, n_ctrl - self.DEG + 1)
        t = np.concatenate([[0] * self.DEG, interior, [self.span] * self.DEG]) + self.first_knot
        raw = np.zeros((n_ctrl + self.DEG + 1, 1 + self.ACT))
        raw[:, 0] = t
        M = Rotation.from_quat(quat).as_matrix()
        ctrl = np.zeros((n_ctrl, self.ACT))
        for i in range(n_ctrl):
            ctrl[i, :3] = pos + np.array([self.travel * i / (n_ctrl - 1), 0.0, 0.0])
            ctrl[i, 3:9] = np.concatenate([M[0], M[1]])
            ctrl[i, 9] = self.grip[0] + (self.grip[1] - self.grip[0]) * i / (n_ctrl - 1)
            if self.grip_dip:
                ctrl[i, 9] = [0.0, 0.0, 0.7, 0.2, 0.7, 1.0, 1.0][min(i * 7 // n_ctrl, 6)]
        raw[:n_ctrl, 1:] = ctrl
        return raw


# ---------------------------------------------------------------------------
# Harness plumbing
# ---------------------------------------------------------------------------

class Results:
    def __init__(self):
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")
        return bool(ok)

    @property
    def failed(self):
        return [r for r in self.rows if not r[0]]


@contextlib.contextmanager
def operator(verdict_after: int | None = None, verdict: str = "success",
             first_attempt: str | None = None):
    """Stand in for the person at the robot.

    There is no tty here, so raw mode and the arrow keys have to be replaced;
    `verdict_after` makes Stopper.check return a verdict after N polls so the
    success path is exercised rather than only the timeout path.
    `first_attempt` is a verdict pressed 30 polls into the first attempt only.
    """
    saved = (rc.wait_for_right_arrow, rc.raw_stdin, rc.Stopper.check)
    calls = {"n": 0, "stoppers": []}

    def fake_check(self):
        # Latched, like the real Stopper: a verdict must survive being polled
        # again, which is what the B-Spline inner dispatch loop does.
        if getattr(self, "verdict", None) is not None:
            return self.verdict
        calls["n"] += 1
        if self not in calls["stoppers"]:
            calls["stoppers"].append(self)
            self._polls = 0
        self._polls += 1
        if first_attempt is not None and self is calls["stoppers"][0] and self._polls > 30:
            self.verdict = first_attempt
        elif verdict_after is not None and calls["n"] > verdict_after:
            self.verdict = verdict
        elif self.limit is not None and self.elapsed() >= self.limit:
            self.verdict = "timeout"
        return getattr(self, "verdict", None)

    rc.wait_for_right_arrow = lambda: None
    rc.raw_stdin = contextlib.nullcontext
    rc.Stopper.check = fake_check
    try:
        yield calls
    finally:
        rc.wait_for_right_arrow, rc.raw_stdin, rc.Stopper.check = saved


def _args(**kw):
    """Namespace matching the entrypoints' argparse, defaults filled in."""
    ns = argparse.Namespace(
        rig="single_arm_franka", ckpt=None, port=None, host="localhost",
        record_fps=None, dry_run=False,
        allow_missing_cameras=True,   # the fake rig's cameras are its own
        num_episodes=1, episode_time_s=1.0, task="offline check",
        train_dataset="Offline/check", outputs_root=None, repo_id=None,
        no_record=True, push_to_hub=False, save_videos=False,
        home_pose_name=fc.default_home_pose_name(), home_q=None,
        home_max_time_s=fc.control("homing.max_time_s"),
        home_tol_rad=fc.control("homing.tol_rad"),
        allow_home_mismatch=False,
        speed=1.0, no_precision=False, no_eag=True, no_velocity_lead=False,
        control_freq=None, origin_time_scale=None, predict_before_end=None,
        degree=3, gripper_slowdown=False, restart_on_time_align_error=False,
        consider_gripper_during_align=False, disable_time_align=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


_RUN_ROOT: Path | None = None


def _open_run(args, method):
    """Run directory for one check, under the harness's temp root."""
    args.outputs_root = str(_RUN_ROOT)
    train = {"repo_id": args.train_dataset, "resolved_from": "flag",
             "flag": args.train_dataset, "checkpoint_says": None, "agrees": None}
    return rc.open_run(args, method, train)


def _client(port):
    from baselines.zmq_client import PolicyClient
    return PolicyClient(port, 5000)


def _tags(arm: FakeArm) -> list[tuple[int, int]]:
    """Dispatched (chunk_id, row_index) pairs, decoded from the gripper tag.

    Integer arithmetic on g*100: the server stores chunks as float32, so
    comparing fractional parts directly is off by ~1e-7.
    """
    out = []
    for a in arm.sent:
        n = int(round(a[f"{rc.ARM_KEY}_gripper"] * 100))
        out.append((n // 100, n % 100))
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_pure(res: Results) -> None:
    print("\n[pure] ported helpers and unit conversion")

    # slowdown_mode against upstream's own source, exec'd standalone.
    src = (_ROOT / "baselines/sail/robomimic/SAIL/utils/dev_utils.py").read_text()
    ns = {"np": np}
    exec(src[src.index("def get_slowdown_mode_from_model"):
             src.index("def get_slowdown_mode_from_gripper")], ns)
    upstream = ns["get_slowdown_mode_from_model"]
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(2000):
        prev = [rng.random(8) for _ in range(int(rng.integers(0, 7)))]
        cur, fut = rng.random(8), rng.random((int(rng.integers(1, 10)), 8))
        for a in prev:
            a[-1] = rng.choice([0.0, 1.0])
        cur[-1] = rng.choice([0.0, 1.0])
        fut[:, -1] = rng.choice([0.0, 1.0], size=fut.shape[0])
        w = int(rng.choice([1, 2, 3, 6, 7]))
        bad += bool(upstream(prev, cur.copy(), fut, w)) != rc.slowdown_mode(prev, cur, fut, w)
    res.check(bad == 0, "slowdown_mode matches upstream", f"{bad}/2000 mismatches")

    # rot6d round-trip
    worst = 0.0
    for _ in range(2000):
        R = Rotation.random(random_state=int(rng.integers(1 << 30)))
        M = R.as_matrix()
        q = rc.rot6d_to_quat_xyzw(np.concatenate([M[0], M[1]]))
        worst = max(worst, (Rotation.from_quat(q) * R.inv()).magnitude())
    res.check(worst < 1e-9, "rot6d_to_quat_xyzw round-trips", f"worst {worst:.2e} rad")

    # the absolute action is a unit quaternion in metres, with the gain channels it was given
    d = rc.ee_pos_action([0.4, -0.02, 0.3], [0.0, 0.0, 0.0, 2.0], 0.4)
    q = np.array([d[f"{rc.ARM_KEY}_q{c}"] for c in "xyzw"])
    res.check(np.allclose(q, [0, 0, 0, 1]) and d[f"{rc.ARM_KEY}_x"] == 0.4
              and d["kp"] == 0.0 and d["kd"] == 0.0,
              "ee_pos_action normalises the quaternion; stock gains are normalised 0")

    # SAIL runs osc_kp_scale x the stock kp at osc_damping_ratio
    default_kp = float(fc.control("torque.osc.default_kp"))
    base = float(fc.control("torque.osc.gain_exp_base"))
    g = rc.sail_gains()
    kp = default_kp * base ** g["kp"]
    ratio = float(fc.control("torque.osc.default_damping_ratio")) * base ** g["kd"]
    want = float(fc.policy("baselines.sail.osc_damping_ratio"))
    res.check(abs(kp - 2.0 * default_kp) < 1e-9 and abs(ratio - want) < 1e-9,
              "SAIL's gain channels resolve to 2x the stock kp at osc_damping_ratio",
              f"kp {kp:.1f}, damping ratio {ratio:.2f}")
    res.check(np.allclose(rc.osc_lag(g), 2.0 * want / np.sqrt(kp)),
              "SAIL's velocity lead is the OSC's own kd/kp", f"{rc.osc_lag(g)[0]:.4f} s")

    # B-Spline: the stock controller at 1x, sped up with the plan above it
    res.check(not rc.bspline_lead(1.0).any(),
              "B-Spline at 1x sends the plan as sampled, to the stock controller")
    if fc.policy("baselines.bspline.goal_lead"):
        for speed in (2.0, 3.0, 5.0):
            lead = rc.bspline_lead(speed)
            res.check(np.allclose(rc.osc_lag(rc.stock_gains()) * speed - lead * speed,
                                  rc.osc_lag(rc.stock_gains())),
                      f"at {speed:g}x the goal lead holds the stock OSC's 1x lag",
                      f"lead {lead[0] * 1e3:.0f} ms")
    # A goal led by kd/kp along the target velocity is the PD law with a velocity term.
    rng = np.random.default_rng(1)
    x, v, xd, vd = (rng.normal(size=3) for _ in range(4))
    kpp, kdd = 300.0, 2.0 * np.sqrt(300.0)
    lead_pos, _ = apply_lead(xd, [0, 0, 0, 1.0], vd, np.zeros(3), np.full(6, kdd / kpp))
    res.check(np.allclose(kpp * (lead_pos - x) - kdd * v, kpp * (xd - x) + kdd * (vd - v)),
              "kp*(led goal - x) - kd*v == kp*e_p + kd*e_v")

    # SAIL executes absolute poses only, as upstream's evaluation does
    ok = all(sail_rollout.resolve_control_mode({"action_keys": [k]}) is ControlMode.EE_POS
             for k in ("absolute_actions", "absolute_actions_with_precision",
                       "commanded_absolute_actions_with_precision"))
    refused = 0
    for keys in (["actions"], ["mystery"], ["absolute_actions", "actions"]):
        try:
            sail_rollout.resolve_control_mode({"action_keys": keys})
        except ValueError:
            refused += 1
    res.check(ok and refused == 3,
              "SAIL takes an absolute action key and refuses deltas, unknown and multiple keys")

    # Stopper latches its verdict: reading a keypress consumes it, so a second
    # poll must not come back empty.
    st = rc.Stopper(None)
    st.verdict = "success"
    res.check(st.check() == "success" and st.check() == "success",
              "Stopper latches the operator verdict across repeated polls")
    st2 = rc.Stopper(0.0)
    res.check(st2.check() == "timeout" and st2.check() == "timeout",
              "Stopper latches a timeout too")

    # A run directory is created once and never collides, and the same training
    # dataset groups every method's runs together.
    a1, a2 = _args(), _args()
    d1, _ = _open_run(a1, "sail")
    d2, _ = _open_run(a2, "bspline")
    res.check(d1.path.parent == d2.path.parent and d1.path != d2.path,
              "methods trained on one dataset share its output directory",
              str(d1.path.parent.relative_to(_RUN_ROOT)))
    res.check(d1.run_id.endswith("-sail") and d2.run_id.endswith("-bspline")
              and d1.path.is_dir(),
              "run id is <timestamp>-<method>", f"{d1.run_id}")
    res.check(a1.repo_id != a2.repo_id and a1.repo_id.startswith("check"),
              "recorded dataset repo id is derived per run", a1.repo_id)

    # Finding 7: the slowdown window must include the current row on the left,
    # as upstream's append-then-call ordering does.
    lab = lambda v: np.array([0, 0, 0, 0, 0, 0, 0, float(v)])
    #   window 6 -> segment 3. Only the row two back carries a label.
    prev_with_cur = [lab(0), lab(1), lab(0), lab(0)]     # ..., cur appended last
    res.check(rc.slowdown_mode(prev_with_cur, lab(0), np.array([lab(0)]), 6) is True,
              "slowdown window reaches 3 rows back including the current one")
    res.check(rc.slowdown_mode([lab(1), lab(0), lab(0), lab(0)], lab(0),
                               np.array([lab(0)]), 6) is False,
              "and no further back than that")

    # Finding 5: a checkpoint naming a camera the rig lacks is refused up front.
    class _R:
        cameras = {"cam_2": None, "cam_5": None}
    meta = {"obs_key_shapes": {"cam_1_image": [3, 84, 84], "arm_pos": [3]}}
    refused = False
    try:
        rc.check_camera_coverage(meta, _R())
    except ValueError as exc:
        refused = "cam_1" in str(exc)
    res.check(refused, "a camera the rig lacks is refused before homing")
    kept = rc.check_camera_coverage(meta, _R(), allow_missing=True)
    res.check("cam_1_image" not in kept and "arm_pos" in kept,
              "--allow-missing-cameras drops the key instead")

    # the dataset schema is run_residual.py's
    res.check(rc.ACTION_KEYS == (*(f"{rc.ARM_KEY}_{a}" for a in
                                   ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")),
                                 "kp", "kd")
              and rc.STATE_OBS_KEYS[-1] == f"{rc.ARM_KEY}_gripper"
              and len(rc.STATE_OBS_KEYS) == fc.num_joints() + 1,
              "dataset features match a run_residual.py recording")


def check_outputs(res: Results) -> None:
    print("\n[outputs] one directory per task, one run per method")

    # The headline requirement: SAIL, B-Spline and multi-fast trained on the
    # same demonstrations file themselves under that task, not under themselves.
    task = "HuskyMango/pickup-bowl"
    dirs = {}
    for method in rr.METHODS:
        a = _args(train_dataset=task)
        dirs[method], _ = _open_run(a, method)
    parents = {d.path.parent for d in dirs.values()}
    res.check(len(parents) == 1
              and parents.pop() == _RUN_ROOT / "HuskyMango" / "pickup-bowl",
              "all three methods land under outputs/<org>/<task>",
              ", ".join(sorted(d.run_id for d in dirs.values())))
    res.check(all(d.run_id.endswith(f"-{m}") for m, d in dirs.items()),
              "each run is timestamped and names its method")

    # Two runs of one method at the same instant must be refused, not merged:
    # a shared directory would interleave two rollouts' episodes.jsonl.
    fixed = time.time()
    rr.RunDir("Collision/check", "sail", root=_RUN_ROOT, when=fixed)
    refused = False
    try:
        rr.RunDir("Collision/check", "sail", root=_RUN_ROOT, when=fixed)
    except FileExistsError as exc:
        refused = "same second" in str(exc)
    res.check(refused, "a duplicate run id is refused rather than overwritten")

    # A task id with no org still produces a single level.
    plain, _ = _open_run(_args(train_dataset="pickup-bowl"), "sail")
    res.check(plain.path.parent == _RUN_ROOT / "pickup-bowl",
              "an un-namespaced dataset id gives one level")

    # Path separators in a dataset id cannot escape the outputs root.
    evil, _ = _open_run(_args(train_dataset="../../etc/passwd"), "sail")
    res.check(_RUN_ROOT.resolve() in evil.path.resolve().parents,
              "a dataset id cannot escape the outputs root",
              str(evil.path.relative_to(_RUN_ROOT)))

    # The manifest carries every section the comparison needs.
    a = _args(train_dataset="Manifest/check")
    run_dir, record = _open_run(a, "sail")
    record.set("policy", checkpoint=rr.file_provenance(None))
    record.set("parameters", exec_fps=100.0)
    record.set("environment", rig_profile="single_arm_right")
    record.add_episode(rc.Episode(episode=0, success=True, verdict="success",
                                  wall_time_s=8.4, steps=840, inferences=53))
    record.finish("completed")
    doc = json.loads(run_dir.manifest_path.read_text())
    want = {"schema_version", "run", "train_dataset", "environment", "policy",
            "parameters", "outputs", "summary"}
    res.check(want <= set(doc), "manifest carries every section",
              ", ".join(sorted(want - set(doc))) or "all present")
    res.check(doc["run"]["status"] == "completed"
              and doc["run"]["method"] == "sail"
              and doc["environment"]["git"]["commit"],
              "manifest records status, method and the code that produced it")
    res.check(doc["summary"]["successes"] == 1
              and doc["summary"]["success_rate"] == 1.0
              and doc["summary"]["mean_time_to_success_s"] == 8.4,
              "summary aggregates the episode log")

    # Episode rows carry the fields the comparison is scored on.
    row = json.loads(run_dir.episodes_path.read_text().strip())
    for key in ("episode", "success", "verdict", "wall_time_s", "steps",
                "frames_recorded", "inferences", "exec_fps", "max_lead_m"):
        if key not in row:
            res.check(False, f"episode row is missing {key!r}")
            break
    else:
        res.check(True, "episode rows carry the scored fields",
                  f"{len(row)} fields")


def check_force(res: Results, run_dir, ep: dict) -> None:
    """One wrench sample per dispatched goal, summarised as eval_fast does."""
    f = ep.get("ee_force_n") or {}
    res.check(f.get("max") == ep["steps"] and f.get("mean") == (ep["steps"] + 1) / 2,
              "ee_force_n summarises one |F| sample per dispatched goal", str(f))
    with np.load(run_dir.force_profiles_path) as z:
        force, t = z["ee_force_000"], z["time_000"]
        torque = z["ee_torque_000"]
    res.check(force.shape == (ep["steps"], 3) and torque.shape == force.shape
              and len(t) == len(force) and bool(np.all(np.diff(t) > 0)),
              "force_profiles.npz holds per-step force, torque and time",
              f"force {force.shape}, time {t[0]:.3f}..{t[-1]:.3f} s")
    doc = json.loads(run_dir.manifest_path.read_text())
    res.check((doc["summary"].get("ee_force_n") or {}).get("max") == f.get("max"),
              "the manifest summary carries the run's force")


def check_dispatch(res: Results) -> None:
    print("\n[dispatch] goals go out with the observation's state read as their anchor")

    class Sink:
        last_ee_wrench = {rc.ARM_KEY: np.zeros(6)}

        def __init__(self):
            self.calls: list[dict] = []

        def send_action(self, a, **kw):
            self.calls.append(kw)

    sink = Sink()
    d = rc.Dispatcher(sink)
    d.send({})
    d.send({}, anchor={rc.ARM_KEY: "snapshot"})
    res.check(sink.calls == [{}, {"anchor": {rc.ARM_KEY: "snapshot"}}],
              "an anchor is passed to send_action only when there is one")
    res.check(d.steps == 2 and len(d.wrench.force) == 2,
              "every goal is counted and its wrench sampled")


def check_schedule(res: Results) -> None:
    print("\n[schedule] SAIL's hardware executor: rows due in time, a fitted spline, its velocity")
    window = int(fc.policy("baselines.sail.slowdown_window_size"))
    knot_rows = int(fc.policy("baselines.sail.spline_knot_rows"))
    n, fast, slow = 16, 40.0, 20.0
    rows = np.zeros((n, 8))
    rows[:, 0] = 0.4 + 0.002 * np.arange(n)                  # +x, 2 mm a row
    rows[:, 2] = 0.3
    yaw = Rotation.from_rotvec(np.outer(0.02 * np.arange(n), [0.0, 0.0, 1.0]))  # 0.02 rad a row about base z
    rows[:, 3:6] = (yaw * Rotation.from_rotvec([2.5, 0.0, 0.0])).as_rotvec()
    rows[:, 6] = np.arange(n) / 100.0                         # gripper tags its row
    rows[3:6, 7] = 1.0                                        # precision label on rows 3..5

    p = ScheduledPlan(rows, 5.0, fast, slow, precision=True, window=window, knot_rows=knot_rows)
    want = np.array([rc.slowdown_mode(list(rows[:k + 1]), rows[k], rows[k:], window)
                     for k in range(n)])
    gaps = np.diff(p.times)
    res.check(np.array_equal(p.slow, want) and p.slow[3:6].all() and not p.slow[-4:].any(),
              "a row is slow exactly where upstream's window says", f"slow rows {np.flatnonzero(p.slow)}")
    res.check(p.times[0] == 5.0 and np.allclose(gaps, np.where(p.slow[1:], 1 / slow, 1 / fast)),
              "row k is due at t_obs plus each step's own 1/hz",
              f"{sorted({round(g * 1e3, 1) for g in gaps})} ms")

    q = ScheduledPlan(rows, 5.0, fast, slow, precision=False, window=window, knot_rows=knot_rows)
    fit = max(np.linalg.norm(q.pose(t)[0] - r[:3]) for t, r in zip(q.times, rows))
    v, w = q.velocity(q.times[7] + 0.004)
    res.check(fit < 1e-9 and np.allclose(v, [0.002 * fast, 0, 0], atol=1e-9),
              "a straight plan is fitted exactly and its velocity is the rows' rate",
              f"v {v.round(4)} m/s")
    res.check(np.allclose(w, [0, 0, 0.02 * fast], atol=1e-6),
              "angular velocity is in the base frame", f"w {w.round(4)} rad/s")
    res.check(not np.any(q.velocity(q.end)[0]) and np.allclose(q.pose(q.end + 1.0)[0], rows[-1, :3]),
              "past its last row a plan holds that row, at rest")
    res.check(q.entry(q.times[3]) == 3 and q.entry(q.times[3] + 1e-6) == 4
              and abs(q.index_at(0.5 * (q.times[2] + q.times[3])) - 2.5) < 1e-12,
              "a plan arriving at t takes over at the first row not yet due")
    res.check(q.gripper(0.5 * (q.times[2] + q.times[3])) == rows[3, 6],
              "the gripper is the row being moved toward, as upstream commands it")
    g = p.rows_at(2.5, 4)
    res.check(g.shape == (4, 8) and np.allclose(g[:, :7], 0.5 * (rows[2:6, :7] + rows[3:7, :7]))
              and set(g[:, 7]) <= {0.0, 1.0},
              "the guide is the plan's own rows at the observation's fraction, labels not blended")


def check_home(res: Results) -> None:
    print("\n[home] a rollout starts where the demonstrations did")
    import tempfile

    import pandas as pd

    q_home = fc.home_q(fc.default_home_pose_name(), key=rc.ARM_KEY)

    def recording(root: Path, q0: np.ndarray) -> str:
        (root / "meta").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        names = [*(f"{rc.ARM_KEY}_joint_{i}" for i in range(1, fc.num_joints() + 1)),
                 f"{rc.ARM_KEY}_gripper"]
        (root / "meta" / "info.json").write_text(
            json.dumps({"features": {"observation.state": {"names": [names]}}}))
        frames = [{"observation.state": np.r_[q0 + 0.01 * i, 1.0].astype(np.float32),
                   "frame_index": i, "episode_index": e} for e in range(3) for i in range(4)]
        pd.DataFrame(frames).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
        return str(root)

    def ns(**kw):
        base = dict(home_pose_name=None, home_q=None, allow_home_mismatch=False)
        return argparse.Namespace(**{**base, **kw})

    with tempfile.TemporaryDirectory() as tmp:
        same = recording(Path(tmp) / "same", q_home)
        far_q = q_home.copy()
        far_q[3] += 0.4                                       # elbow: the EE moves ~15 cm
        far = recording(Path(tmp) / "far", far_q)

        a = ns()
        got = rc.resolve_home(a, {"repo_id": same})
        res.check(got["checked"] and got["gap_m"] < 1e-3 and a.home_pose_name is not None,
                  "without a flag the saved pose nearest the demonstrations' start is homed to",
                  f"{a.home_pose_name}, {got['gap_m'] * 100:.2f} cm")
        refused = ""
        try:
            rc.resolve_home(ns(), {"repo_id": far})
        except ValueError as exc:
            refused = str(exc)
        res.check("demonstrations started" in refused,
                  "no saved pose near the start is refused, not guessed", refused[:60])
        refused = ""
        try:
            rc.resolve_home(ns(home_pose_name=fc.default_home_pose_name()), {"repo_id": far})
        except ValueError as exc:
            refused = str(exc)
        res.check("--allow-home-mismatch" in refused,
                  "an explicit home far from the start is refused too")
        got = rc.resolve_home(ns(home_pose_name=fc.default_home_pose_name(),
                                 allow_home_mismatch=True), {"repo_id": far})
        res.check(got["gap_m"] > float(fc.policy("baselines.exec.home_tolerance_m")),
                  "--allow-home-mismatch rolls out from it, recording the gap",
                  f"{got['gap_m'] * 100:.1f} cm")
        a = ns()
        got = rc.resolve_home(a, {"repo_id": str(Path(tmp) / "absent")})
        res.check(not got["checked"] and a.home_pose_name == fc.default_home_pose_name(),
                  "a recording not on this machine falls back to the default home, unchecked")


def check_timing(res: Results) -> None:
    print("\n[timing] spline wall clock and origin_time_scale")
    from baselines.bspline_bridge.spline_plan import SplinePlanner

    class Direct:
        def __init__(self, srv): self.srv = srv
        def request(self, payload):
            return self.srv.handle(payload)

    srv = FakeBSpline(0)
    for speed, ots in ((1.0, 20.0), (2.0, 20.0), (1.0, 10.0)):
        p = SplinePlanner(Direct(srv), degree=3, n_obs_steps=1, obs_stride=1,
                          origin_time_scale=ots, speed_up_times=speed,
                          predict_before_end=1e9, gripper_index=9, compare_dim=9)
        p.step({"arm_pos": np.zeros(3), "arm_quat": np.array([0.0, 0, 0, 1.0])})
        for _ in range(400):
            if not p.waiting_for_first_plan():
                break
            time.sleep(0.005)
        t0 = time.perf_counter()
        while p.poll_action() is not None:
            time.sleep(0.002)
        dur, want = time.perf_counter() - t0, srv.span / ots / speed
        res.check(abs(dur - want) < 0.08,
                  f"plan plays in wall clock at speed={speed} ots={ots}",
                  f"{dur:.3f}s vs {want:.3f}s")
        p.close()
    res.check(True, "ots=10 against 20 Hz data plays at 0.5x (the misconfiguration "
                    "config/policy.yaml warns about)")

    # A plan whose first knot lies ahead of the elapsed time: upstream jumps to it,
    # defer_stitch holds the old plan until it is due and then aligns. The request
    # goes out 0.5 s before the old plan ends, so the old plan covers the wait.
    srv = FakeBSpline(0, first_knot=4.0)
    for defer in (False, True):
        clock = {"t": 0.0}
        p = SplinePlanner(Direct(srv), degree=3, n_obs_steps=1, obs_stride=1,
                          origin_time_scale=20.0, speed_up_times=1.0, predict_before_end=0.5,
                          gripper_index=9, compare_dim=9, defer_stitch=defer,
                          clock=lambda: clock["t"], synchronous=True)
        obs = {"arm_pos": np.zeros(3), "arm_quat": np.array([0.0, 0, 0, 1.0])}
        events = []
        while clock["t"] < 1.5:
            before = p.plans
            p.step(obs)
            if p.plans != before:
                events.append((round(clock["t"], 3), p.plans))
            clock["t"] += 0.005
        second = next(t for t, n in events if n == 2)
        if not defer:
            res.check(not np.isfinite(p.align_errors[0]),
                      "upstream: a plan that lands before its first knot is entered at that knot, unaligned")
        else:
            res.check(p.deferred >= 1 and np.isfinite(p.align_errors[0]),
                      "defer_stitch: the plan waits for its first knot, then the stitch is searched",
                      f"second plan installed at {second:.3f} s, {p.deferred} deferral(s)")
        p.close()


def _run_sail(port, args, lead: bool = True, **server_kw):
    srv = FakeSail(port, action_keys=["absolute_actions_with_precision"], act_dim=8,
                   precision=True, **server_kw)
    srv.start()
    try:
        client = _client(port)
        meta = client.meta()
        s = sail_rollout.settings(meta, args)
        arm = FakeArm(sail_rollout.resolve_control_mode(meta))
        if lead:
            s["lead"] = rc.osc_lag(s["gains"], arm)
        run_dir, record = _open_run(args, "sail")
        record.set("environment", **rc.environment(args, arm))
        record.set("policy", control_mode=ControlMode.EE_POS.value)
        with operator():
            rc.run_episodes(args, arm, run_dir, record,
                            sail_rollout.make_episode_fn(
                                client, s, rc.check_camera_coverage(meta, arm, True)),
                            nominal_hz=s["goal_hz"])
        return srv, arm, s, run_dir, record.episodes[0]
    finally:
        srv.stop()


def _segments(tags: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    """Dispatched goals grouped by the plan they came from, in dispatch order."""
    out: list[tuple[int, list[int]]] = []
    for k, j in tags:
        if not out or out[-1][0] != k:
            out.append((k, []))
        out[-1][1].append(j)
    return out


def check_sail(res: Results, port: int) -> None:
    print("\n[sail] the paper's hardware executor: rows due in time, plans switched on arrival")
    execute_n = int(fc.policy("baselines.sail.execute_n_actions"))
    inf_delay = int(fc.policy("baselines.sail.inf_delay"))
    goal_hz = float(fc.policy("baselines.sail.goal_hz"))

    # --- instant inference, precision on, 2x
    args = _args(speed=2.0, episode_time_s=2.0)
    srv, arm, s, run_dir, ep = _run_sail(port, args)
    res.check(s["fast_hz"] == 2.0 * rc.data_fps() and s["slow_hz"] == float(fc.policy("baselines.exec.slow_fps")),
              "--speed 2 makes fast rows due at 2x the demonstrations' rate; slow rows stay 1x",
              f"{s['fast_hz']:.0f}/{s['slow_hz']:.0f} Hz")
    gaps = np.diff(arm.send_times)
    res.check(abs(float(np.median(gaps)) - 1.0 / goal_hz) < 0.002,
              "the plan streams at baselines.sail.goal_hz",
              f"median gap {1e3 * np.median(gaps):.1f} ms, max {1e3 * gaps.max():.1f} ms")
    res.check(arm.anchored == len(arm.sent) > 0,
              "every goal goes out on the observation's state read (one RPyC call, not two)",
              f"{arm.anchored}/{len(arm.sent)}")
    segs = _segments(_tags(arm))
    ids = [k for k, _ in segs]
    res.check(ids == sorted(set(ids)) and ids[0] == 0,
              "plans take over in order and an old plan never resumes", f"{len(ids)} plans")
    res.check(all(rows == sorted(rows) for _, rows in segs),
              "within a plan, rows go out in schedule order")
    res.check(all(rows[0] >= 1 for _, rows in segs[1:]),
              "a new plan takes over at the row due when it lands, never its row 0",
              f"entry rows {sorted({rows[0] for _, rows in segs[1:]})}")
    span = [rows[-1] - rows[0] + 1 for _, rows in segs[1:-1]]
    res.check(bool(span) and all(execute_n - 1 <= n <= execute_n + 3 for n in span),
              "each plan runs about execute_n_actions rows before the next takes over",
              f"rows per plan {sorted(set(span))}")
    res.check(ep["slow_steps"] > 0 and ep["notes"]["plan_exhausted_s"] == 0.0,
              "precision rows are counted, and an instant inference never runs a plan dry",
              f"{ep['slow_steps']} slow rows")
    kp = {round(a["kp"], 6) for a in arm.sent}
    kd = {round(a["kd"], 6) for a in arm.sent}
    res.check(kp == {round(rc.sail_gains()["kp"], 6)} and kd == {round(rc.sail_gains()["kd"], 6)},
              "every goal carries SAIL's gain channels")
    spacing = [b - a for a, b in srv.frame_ids[1:]]
    res.check(all(n == 2 for n in srv.frames_seen) and bool(spacing) and all(d >= 1 for d in spacing),
              "frame stack is two distinct camera frames", f"camera-frame spacing {sorted(set(spacing))}")
    check_force(res, run_dir, ep)
    grips = [a[f"{rc.ARM_KEY}_gripper"] for a in arm.sent]
    res.check(not all(g in (0.0, 1.0) for g in grips),
              "precision label stripped; gripper is the chunk's own column")

    # --- EAG, and an inference slower than inf_delay rows
    t_f = 4
    args = _args(speed=2.0, episode_time_s=2.5, no_eag=False, no_precision=True)
    srv, arm, s, run_dir, ep = _run_sail(port + 1, args, fac_horizon=t_f, guided=True, delay_s=0.15)
    sent = [(i, np.asarray(g)) for i, g in enumerate(srv.guide_seen) if g is not None]
    ok = bool(sent)
    starts = []
    for i, g in sent:
        tags = np.round(g[:, 6] * 100).astype(int)       # chunk*100 + row, interpolated
        ok &= g.shape == (t_f, 8) and bool(np.all(tags // 100 == i - 1))
        ok &= bool(np.allclose(np.diff(g[:, 6]), 0.01, atol=1e-6))
        starts.append(float(g[0, 6] * 100 - (i - 1) * 100))
    res.check(ok and ep["guided_inferences"] == len(sent),
              "guidance is the previous plan's rows from the observation on, full width",
              f"{len(sent)} of {len(srv.guide_seen)} inferences guided, from rows "
              f"{sorted({round(x, 1) for x in starts})}")
    res.check(bool(starts) and max(starts) <= s["horizon"] - inf_delay + 1e-6,
              "a request goes out by row H - inf_delay at the latest")
    segs = _segments(_tags(arm))
    ids = [k for k, _ in segs]
    res.check(ids == sorted(set(ids)) and ep["inferences"] >= 3,
              "a slow inference still hands over in order", f"{ep['inferences']} plans")
    res.check(ep["notes"]["plan_exhausted_s"] < 0.25 * ep["wall_time_s"],
              "with a 150 ms inference the next request goes out early enough to keep the plan fed",
              f"held the last row {ep['notes']['plan_exhausted_s']:.3f} s of {ep['wall_time_s']:.2f} s")

    # --- the recorded run renders, each chunk drawn from the row it took over at
    from baselines import rollout_viz
    import pandas as pd
    args = _args(speed=2.0, episode_time_s=2.0, no_record=False, train_dataset="Offline/sail-viz")
    _, _, s, run_dir, ep = _run_sail(port + 2, args, lead=False)
    pages = rollout_viz.render_run(run_dir.path)
    res.check([q.name for q in pages] == ["episode_000.html"], "the run renders its episode page")
    df = pd.read_parquet(sorted((run_dir.dataset_dir / "data").glob("*/*.parquet"))[0])
    goal = np.stack(df["action"].values)[:, :3]
    with np.load(run_dir.path / "chunks.npz") as z:
        steps, poses = z["chunk_step_000"], z["chunk_pose_000"]
    reach = int(np.ceil(s["fast_hz"] / rc.data_fps()))
    rows = [p[~np.isnan(p[:, 0]), :3] for p in poses]     # chunks are NaN-padded to one length
    # chunk_step is the next frame recorded, 0-1 recorder periods after the take-over.
    lag = [int(np.argmin(np.linalg.norm(rows[k] - goal[st], axis=1)))
           for k, st in enumerate(steps) if st < len(goal) and len(rows[k])]
    res.check(bool(lag) and max(lag) <= reach + 1,
              "each chunk is logged from the row it takes over at",
              f"goal at the chunk's frame is row {sorted(set(lag))} of the logged chunk")


def _run_bspline(port, args, arm=None, first_attempt=None, **server_kw):
    srv = FakeBSpline(port, **server_kw)
    srv.start()
    try:
        client = _client(port)
        meta = client.meta()
        control_freq = float(args.control_freq or fc.policy("baselines.bspline.control_freq"))
        kwargs = bsp_rollout.planner_settings(meta, args, control_freq)
        arm = arm or FakeArm(ControlMode.EE_POS)
        run_dir, record = _open_run(args, "bspline")
        record.set("environment", **rc.environment(args, arm))
        record.set("policy", control_mode=ControlMode.EE_POS.value)
        with operator(first_attempt=first_attempt):
            rc.run_episodes(args, arm, run_dir, record,
                            bsp_rollout.make_episode_fn(
                                client, kwargs, control_freq,
                                rc.check_camera_coverage(meta, arm, True),
                                rc.stock_gains(), rc.bspline_lead(args.speed, arm)),
                            nominal_hz=control_freq)
        return srv, meta, arm, kwargs, run_dir, record
    finally:
        srv.stop()


def check_bspline(res: Results, port: int) -> None:
    print("\n[bspline] upstream's control loop: fixed rate, stride, no goal before a plan")
    rejected = False
    try:
        bsp_rollout.decode(np.zeros(7))
    except ValueError:
        rejected = True
    res.check(rejected, "decode refuses an action that is not 10-dim")

    control_freq = 100.0
    args = _args(control_freq=control_freq, episode_time_s=1.5, speed=1.0)
    srv, meta, arm, kwargs, run_dir, record = _run_bspline(port, args)
    ep = record.episodes[0]
    res.check(meta["act_dim"] == 10, "meta reports the 10-dim rot6d action our converter produces")
    res.check(kwargs["origin_time_scale"] == rc.data_fps(),
              "origin_time_scale is the demonstrations' rate", f"{kwargs['origin_time_scale']}")
    res.check(ep["steps"] > 0 and ep["inferences"] >= 2,
              "goals dispatched and the plan replanned", f"{ep['steps']} goals, {ep['inferences']} plans")

    home_pos, _ = eef_poses_from_qpos(fc.home_q(key=rc.ARM_KEY)[None])
    first = np.array([arm.sent[0][f"{rc.ARM_KEY}_{a}"] for a in "xyz"])
    res.check(np.linalg.norm(first - home_pos[0]) < 0.005,
              "the first goal is the first plan's start, sent only once a plan exists",
              f"{1e3 * np.linalg.norm(first - home_pos[0]):.2f} mm from the homed pose")

    gaps = np.diff(arm.send_times)
    med = float(np.median(gaps)) if len(gaps) else 0.0
    res.check(abs(med - 1.0 / control_freq) < 0.002, "goals go out at --control-freq",
              f"median gap {med * 1e3:.2f} ms, target {1e3 / control_freq:.2f} ms, "
              f"max {1e3 * float(np.max(gaps)):.1f} ms")
    res.check({a["kp"] for a in arm.sent} == {0.0} and {a["kd"] for a in arm.sent} == {0.0},
              "at 1x every goal carries the stock gain channels")
    res.check(arm.anchored == len(arm.sent),
              "every goal goes out on the observation's state read (one RPyC call, not two)")
    spacing = [b - a for fr in srv.frames_seen if len(fr) == 2 for a, b in [fr]]
    res.check(bool(spacing) and all(1 <= d <= 3 for d in spacing),
              "the policy's two observations are distinct frames ~one demonstration step apart",
              f"camera-frame spacing {sorted(set(spacing))}")
    res.check(ep["max_lead_m"] > 0.0, "the goal's lead over the arm is recorded",
              f"max lead {ep['max_lead_m']:.4f} m")

    # --- sped up, the controller stays stock and the goals lead the plan
    args = _args(control_freq=control_freq, episode_time_s=1.0, speed=2.0)
    _, _, arm, _, _, record = _run_bspline(port + 2, args)
    res.check({a["kp"] for a in arm.sent} == {0.0} and {a["kd"] for a in arm.sent} == {0.0},
              "at 2x every goal still carries the stock gain channels")

    # --- a grasp holds the arm for grasp_hold_s, then the plan resumes
    hold_s = float(fc.policy("baselines.bspline.grasp_hold_s"))
    if hold_s > 0:
        arm = FakeArm(ControlMode.EE_POS)
        arm.gripper_rate = 1.5            # ~0.65 s to close, as the WSG
        args = _args(control_freq=control_freq, episode_time_s=2.5)
        _, _, arm, _, _, record = _run_bspline(port + 3, args, arm=arm, span=60, travel=0.45,
                                               grip=(1.0, 0.0))
        ep = record.episodes[0]
        g = np.array([[a[f"{rc.ARM_KEY}_{c}"] for c in "xyz"] for a in arm.sent])
        grip = np.array([a[f"{rc.ARM_KEY}_gripper"] for a in arm.sent])
        t = np.array(arm.send_times)
        first = int(np.argmax(grip == 0.0))
        same = np.all(np.isclose(g[first:], g[first]), axis=1)
        last = first + int(np.argmin(same)) - 1          # last goal of the hold
        held = t[last + 1] - t[first]
        res.check(ep["notes"]["grasp_holds"] == 1 and abs(held - hold_s) < 0.03,
                  "pause, grasp, unpause: the arm holds for grasp_hold_s with the gripper shut",
                  f"{held * 1e3:.0f} ms hold")
        res.check(np.all(grip[first:] == 0.0) and grip[first - 1] > 0.0,
                  "the gripper is commanded shut through the hold and after it")
        step_in = np.linalg.norm(g[first - 1] - g[first - 2])
        jump = np.linalg.norm(g[last + 1] - g[first - 1])
        res.check(jump < 3 * step_in + 1e-3,
                  "the plan resumes where it stopped, not the hold's length further on",
                  f"{jump * 1e3:.1f} mm across the hold")

    # --- a release never pauses, even when its command dips back under the threshold
    if hold_s > 0:
        args = _args(control_freq=control_freq, episode_time_s=1.5)
        _, _, arm, _, _, record = _run_bspline(port + 4, args, span=40, travel=0.3,
                                               grip=(0.0, 1.0), grip_dip=True)
        t = np.array(arm.send_times)
        res.check(record.episodes[0]["notes"]["grasp_holds"] == 0 and np.diff(t).max() < 0.05,
                  "opening the gripper never pauses the arm, even when the command wobbles",
                  f"{record.episodes[0]['notes']['grasp_holds']} holds")

    # --- nothing bounds the lead: a plan that outruns a stalled arm keeps going
    arm = FakeArm(ControlMode.EE_POS)
    arm.freeze = True          # the arm stops tracking; the plan runs on
    args = _args(control_freq=control_freq, episode_time_s=2.0)
    _, _, arm, _, _, record = _run_bspline(port + 1, args, arm=arm, travel=0.4)
    ep = record.episodes[0]
    last = np.array([arm.sent[-1][f"{rc.ARM_KEY}_{a}"] for a in "xyz"])
    pos, _ = arm._pose()
    res.check(ep["aborted"] is None and ep["verdict"] == "timeout",
              "a stalled arm does not end the episode", ep["verdict"])
    res.check(np.linalg.norm(last - pos) > 0.15 and ep["max_lead_m"] > 0.15,
              "the goal was never pulled back toward the arm",
              f"final divergence {np.linalg.norm(last - pos):.3f} m")


def check_dataset(res: Results, port: int) -> None:
    print("\n[dataset] recorded rollout is readable and correctly labelled")
    fps = rc.record_fps(_args())
    args = _args(control_freq=100.0, episode_time_s=0.8, num_episodes=2, no_record=False,
                 train_dataset="Offline/dataset-check")
    _, _, arm, _, run_dir, record = _run_bspline(port, args)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    d = LeRobotDataset(repo_id=args.repo_id, root=str(run_dir.dataset_dir))
    info = d.meta.info["features"]
    res.check(float(d.fps) == fps == rc.data_fps(),
              "dataset is written at the demonstrations' rate, off the control loop",
              f"{d.fps} fps")
    per_ep = [e["frames_recorded"] for e in record.episodes]
    expect = [e["wall_time_s"] * fps for e in record.episodes]
    res.check(d.num_episodes == 2 and d.num_frames == sum(per_ep)
              and all(0.6 * x - 2 <= n <= x + 2 for n, x in zip(per_ep, expect)),
              "one frame per 1/record_fps of the episode",
              f"{per_ep} frames for {[round(x, 1) for x in expect]} expected")
    res.check(list(info["action"]["names"][0]) == list(rc.ACTION_KEYS)
              and list(info["observation.state"]["names"][0]) == list(rc.STATE_OBS_KEYS),
              "recorded feature names match run_residual.py's")
    res.check(sorted(d.meta.camera_keys)
              == sorted(f"observation.images.{c}" for c in arm.cameras),
              "both cameras encoded")
    act = d[0]["action"].numpy()
    res.check(abs(act[8]) < 1e-9 and abs(act[9]) < 1e-9 and 0.0 <= act[7] <= 1.0,
              "a recorded action carries the stock gains and an absolute gripper")
    res.check(run_dir.manifest_path.is_file() and run_dir.episodes_path.is_file(),
              "manifest and episodes written into the run directory",
              str(run_dir.path.relative_to(_RUN_ROOT)))
    doc = json.loads(run_dir.manifest_path.read_text())
    res.check(doc["outputs"]["dataset"]["repo_id"] == args.repo_id
              and doc["outputs"]["dataset"]["episodes"] == 2,
              "manifest records the dataset it produced",
              f'{doc["outputs"]["dataset"]["episodes"]} episodes')
    res.check(doc["summary"]["episodes"] == 2
              and len(run_dir.episodes_path.read_text().strip().splitlines()) == 2,
              "one episodes.jsonl line per episode")
    res.check([e["dataset_episode_index"] for e in record.episodes] == [0, 1],
              "dataset indices are recorded per episode")
    with np.load(run_dir.path / "chunks.npz") as z:
        steps = z["chunk_step_000"]
    res.check(len(steps) == record.episodes[0]["inferences"] and steps.max() <= per_ep[0],
              "chunks are indexed by recorded frame, as rollout_viz draws them",
              f"{list(steps)}")

    args2 = _args(control_freq=100.0, episode_time_s=0.8, num_episodes=1, save_videos=True,
                  train_dataset="Offline/video-check")
    _, _, arm2, _, run_dir2, record2 = _run_bspline(port + 1, args2)
    files = sorted(run_dir2.video_dir.glob("*.mp4")) if run_dir2.video_dir.is_dir() else []
    # --- UP retries: the attempt leaves no trace and the episode runs again
    args3 = _args(control_freq=100.0, episode_time_s=0.8, num_episodes=1, no_record=False,
                  save_videos=True, train_dataset="Offline/retry-check")
    _, _, arm3, _, run_dir3, record3 = _run_bspline(port + 2, args3, first_attempt="retry")
    d3 = LeRobotDataset(repo_id=args3.repo_id, root=str(run_dir3.dataset_dir))
    with np.load(run_dir3.force_profiles_path) as z:
        traces = sorted(k for k in z.files if k.startswith("time_"))
    vids = sorted(run_dir3.video_dir.glob("*.mp4"))
    res.check(len(record3.episodes) == 1 and record3.episodes[0]["verdict"] == "timeout"
              and d3.num_episodes == 1 and traces == ["time_000"]
              and len(vids) == len(arm3.cameras),
              "UP discards the attempt and reruns the episode: one row, one recorded episode",
              f"{len(record3.episodes)} row(s), {d3.num_episodes} dataset episode(s), traces {traces}")
    import cv2
    counts = [int(cv2.VideoCapture(str(f)).get(cv2.CAP_PROP_FRAME_COUNT)) for f in files]
    res.check(len(files) == len(arm2.cameras), "one mp4 per camera written",
              f"{[f.name for f in files]}")
    res.check(bool(counts) and all(abs(c - counts[0]) <= 1 for c in counts) and counts[0] > 0,
              "video frames are written on the recorder's clock",
              f"{counts} frames over {record2.episodes[0]['wall_time_s']:.2f} s at {fps} fps")


def check_servers(res: Results) -> None:
    """The policy servers' own logic, under a stubbed robomimic.

    The fake servers elsewhere in this file stand in for the REAL ones, so
    nothing else here exercises what sail_bridge/policy_server.py does between
    the wire and the model -- and that is exactly where the first hardware run
    would have failed: robomimic wants frame-stacked, processed observations
    and `return_action_sequence`, none of which the client sends.
    """
    import importlib
    import types

    print("\n[servers] policy-server request handling")

    # A robomimic whose process_obs_dict does what the real one does to images
    # (HWC uint8 -> CHW float in [0, 1]) and nothing else; a torch is present
    # in this venv already.
    def _process_obs_dict(d):
        out = {}
        for k, v in d.items():
            v = np.asarray(v)
            if k.endswith("_image"):
                v = np.transpose(v.astype(np.float32) / 255.0, (2, 0, 1))
            out[k] = v
        return out
    stubs = {}
    for name in ("robomimic", "robomimic.utils", "robomimic.utils.file_utils",
                 "robomimic.utils.obs_utils", "robomimic.utils.tensor_utils",
                 "robomimic.utils.torch_utils", "robomimic.config"):
        stubs[name] = types.ModuleType(name)
    stubs["robomimic.utils.obs_utils"].process_obs_dict = _process_obs_dict
    stubs["robomimic.config"].config_factory = lambda *a, **k: None
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        sys.modules.pop("baselines.sail_bridge.policy_server", None)
        srv_mod = importlib.import_module("baselines.sail_bridge.policy_server")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # train.data in both forms train.py can leave it in.
    ns = types.SimpleNamespace
    res.check(srv_mod.training_hdf5(ns(train=ns(data="~/x/sail.hdf5"))).endswith("/x/sail.hdf5"),
              "training_hdf5 reads the string form")
    res.check(srv_mod.training_hdf5(ns(train=ns(data=[{"path": "/a/b.hdf5"}]))) == "/a/b.hdf5",
              "training_hdf5 reads the --dataset list form")
    res.check(srv_mod.training_hdf5(ns(train=ns(data=None))) is None,
              "training_hdf5 tolerates an unset train.data")

    class FakePolicy:
        def __init__(self):
            self.calls = []
            self.episodes = 0
        def start_episode(self):
            self.episodes += 1
        def __call__(self, ob, **kwargs):
            self.calls.append((ob, kwargs))
            return np.zeros((16, 8), dtype=np.float32)

    server = srv_mod.SAILPolicyServer.__new__(srv_mod.SAILPolicyServer)
    server.policy = FakePolicy()
    server.guide_config = None
    server.frame_stack = 2
    server.obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "cam_2_image"]
    server.history = None

    def obs(seed):
        rng = np.random.default_rng(seed)
        return {"robot0_eef_pos": np.full(3, float(seed), np.float32),
                "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
                "robot0_gripper_qpos": np.zeros(1, np.float32),
                "robot0_joint_pos": np.zeros(7, np.float32),      # sent for AWE, not a policy key
                "cam_2_image": rng.integers(0, 255, (8, 6, 3), dtype=np.uint8)}

    server._reset()
    first = server._prepare(obs(1))
    res.check(sorted(first) == sorted(server.obs_keys),
              "keys the checkpoint does not use are dropped before processing")
    res.check(first["robot0_eef_pos"].shape == (2, 3) and first["cam_2_image"].shape == (2, 3, 8, 6),
              "every key is stacked to (frame_stack, ...) with images CHW",
              f"{first['robot0_eef_pos'].shape} {first['cam_2_image'].shape}")
    res.check(first["cam_2_image"].dtype == np.float32 and float(first["cam_2_image"].max()) <= 1.0,
              "images are processed to float [0, 1] before stacking")
    res.check(np.array_equal(first["robot0_eef_pos"][0], first["robot0_eef_pos"][1]),
              "the stack is seeded with copies of the first observation, as FrameStackWrapper does")
    second = server._prepare(obs(2))
    res.check(second["robot0_eef_pos"][0, 0] == 1.0 and second["robot0_eef_pos"][1, 0] == 2.0,
              "the stack rolls: [previous, current]")
    server._reset()
    third = server._prepare(obs(3))
    res.check(server.policy.episodes == 2 and third["robot0_eef_pos"][0, 0] == 3.0,
              "reset starts a new episode and clears the stack")

    missing = obs(4); missing.pop("cam_2_image")
    try:
        server._prepare(missing)
        res.check(False, "a missing checkpoint key is refused")
    except KeyError as exc:
        res.check("cam_2_image" in str(exc), "a missing checkpoint key is refused, by name")

    rep = server._infer({"obs": obs(5), "guide_actions": None})
    ob, kwargs = server.policy.calls[-1]
    res.check(kwargs.get("return_action_sequence") is True,
              "the policy is asked for the whole sequence (return_action_sequence=True)")
    res.check("guide_actions" not in kwargs, "no guide kwargs without --guide-config")
    res.check(rep["chunk"].shape == (16, 8) and rep["chunk"].dtype == np.float32,
              "reply carries the (action_horizon, act_dim) float32 chunk")
    server.frame_stack = 1
    server._reset()
    flat = server._prepare(obs(6))
    res.check(flat["robot0_eef_pos"].shape == (3,), "frame_stack 1 passes single frames through")


    # B-Spline server: the checkpoint's cfg is the authority for the dataset
    # path (relative to the checkpoint's ancestors) and for n_obs_steps.
    from baselines.bspline_bridge import policy_server as bsp_srv
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "diffusion_policy" / "data").mkdir(parents=True)
        (root / "diffusion_policy" / "data" / "x.hdf5").write_bytes(b"")
        ckpt = root / "bspline_policy" / "data" / "outputs" / "run" / "checkpoints" / "latest.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")
        cfg = ns(task=ns(dataset_path="../diffusion_policy/data/x.hdf5"), n_obs_steps=2, horizon=16)
        found = bsp_srv._training_hdf5(cfg, str(ckpt))
        res.check(found == str((root / "diffusion_policy" / "data" / "x.hdf5").resolve()),
                  "a relative dataset_path resolves against the checkpoint's ancestors", found)
        res.check(bsp_srv._cfg_lookup(cfg, lambda c: int(c.n_obs_steps)) == 2
                  and bsp_srv._cfg_lookup(cfg, lambda c: c.missing.key) is None,
                  "n_obs_steps comes off the checkpoint cfg; an absent key is None")
        res.check(bsp_srv._training_hdf5(ns(task=ns(dataset_path="/abs/y.hdf5")), str(ckpt)) == "/abs/y.hdf5",
                  "an absolute dataset_path is kept")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=("pure", "outputs", "dispatch", "schedule", "home", "timing",
                                      "servers", "sail", "bspline", "dataset"), default=None)
    p.add_argument("--base-port", type=int, default=5701)
    p.add_argument("--tmp", default=None,
                   help="scratch dir for the recorded dataset (default: a temp dir)")
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, force=True)

    global _RUN_ROOT
    import shutil
    import tempfile
    _RUN_ROOT = Path(args.tmp or tempfile.mkdtemp(prefix="baseline-runs-")) / "outputs"

    res = Results()
    if args.only in (None, "pure"):
        check_pure(res)
    if args.only in (None, "outputs"):
        check_outputs(res)
    if args.only in (None, "dispatch"):
        check_dispatch(res)
    if args.only in (None, "schedule"):
        check_schedule(res)
    if args.only in (None, "home"):
        check_home(res)
    if args.only in (None, "timing"):
        check_timing(res)
    if args.only in (None, "servers"):
        check_servers(res)
    if args.only in (None, "sail"):
        check_sail(res, args.base_port)
    if args.only in (None, "bspline"):
        check_bspline(res, args.base_port + 10)
    if args.only in (None, "dataset"):
        check_dataset(res, args.base_port + 20)

    if not args.tmp:
        shutil.rmtree(_RUN_ROOT.parent, ignore_errors=True)
    else:
        print(f"\nrun directories kept under {_RUN_ROOT}")

    print(f"\n{len(res.rows) - len(res.failed)}/{len(res.rows)} checks passed")
    for _, name, detail in res.failed:
        print(f"  FAILED: {name} {detail}")
    print("\nPASS" if not res.failed else "\nFAIL")
    return 0 if not res.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
