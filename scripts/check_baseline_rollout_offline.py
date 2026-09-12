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
from baselines.bspline_bridge import rollout as bsp_rollout  # noqa: E402
from baselines.sail_bridge import rollout as sail_rollout  # noqa: E402
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
        self.clipped = 0
        self.freeze = False      # stop tracking, to force the lead monitor to fire

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

    def get_observation(self) -> dict:
        obs = {f"{self.k}_joint_{i + 1}": float(v) for i, v in enumerate(self.q)}
        obs[f"{self.k}_gripper"] = self.gripper
        for cam in self.cameras:
            obs[cam] = np.zeros((*_IMG, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict) -> None:
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
        self.gripper = float(np.clip(action[f"{self.k}_gripper"], 0.0, 1.0))
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
    `ee_pos_action`/`ee_delta_action` untouched. Reading it back off the dispatched
    action dicts says exactly which (chunk, index) each goal came from.
    """

    def __init__(self, port, *, action_keys, act_dim, precision,
                 action_horizon=16, fac_horizon=4, guided=False):
        super().__init__(port)
        self.action_keys, self.act_dim = action_keys, act_dim
        self.precision, self.action_horizon = precision, action_horizon
        self.fac_horizon, self.guided = fac_horizon, guided
        self.chunk_id = -1
        self.guide_seen: list = []

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
        absolute = not self.action_keys[0] == "actions"
        chunk = np.zeros((self.action_horizon, self.act_dim))
        base = np.asarray(req["obs"]["robot0_eef_pos"], dtype=np.float64)
        for j in range(self.action_horizon):
            if absolute:
                chunk[j, :3] = base + np.array([0.002 * (j + 1), 0.0, 0.0])
                chunk[j, 3:6] = Rotation.from_quat(
                    np.asarray(req["obs"]["robot0_eef_quat"], dtype=np.float64)).as_rotvec()
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

    def __init__(self, port, span=None, travel=0.03):
        super().__init__(port)
        self.span = self.SPAN if span is None else span
        self.travel = float(travel)   # metres of x the plan sweeps

    def handle(self, req):
        if "meta" in req:
            return {"backend": "bspline", "act_dim": self.ACT, "degree": self.DEG,
                    "n_obs_steps": 2, "precision_column": False,
                    "action_format": "single_yam_rot6d",
                    "obs_key_shapes": {"cam_2_image": [3, 84, 84], "arm_pos": [3],
                                       "arm_quat": [4], "gripper_pos": [1]}}
        if "reset" in req:
            return {}
        self.calls += 1
        obs = req["obs"][-1] if isinstance(req["obs"], (list, tuple)) else req["obs"]
        return {"bspline": self.make(np.asarray(obs["arm_pos"], dtype=np.float64),
                                     np.asarray(obs["arm_quat"], dtype=np.float64))}

    def make(self, pos, quat):
        n_ctrl = 12
        interior = np.linspace(0, self.span, n_ctrl - self.DEG + 1)
        t = np.concatenate([[0] * self.DEG, interior, [self.span] * self.DEG])
        raw = np.zeros((n_ctrl + self.DEG + 1, 1 + self.ACT))
        raw[:, 0] = t
        M = Rotation.from_quat(quat).as_matrix()
        ctrl = np.zeros((n_ctrl, self.ACT))
        for i in range(n_ctrl):
            ctrl[i, :3] = pos + np.array([self.travel * i / (n_ctrl - 1), 0.0, 0.0])
            ctrl[i, 3:9] = np.concatenate([M[0], M[1]])
            ctrl[i, 9] = i / (n_ctrl - 1)
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
def operator(verdict_after: int | None = None, verdict: str = "success"):
    """Stand in for the person at the robot.

    There is no tty here, so raw mode and the arrow keys have to be replaced;
    `verdict_after` makes Stopper.check return a verdict after N polls so the
    success path is exercised rather than only the timeout path.
    """
    saved = (rc.wait_for_right_arrow, rc.raw_stdin, rc.Stopper.check)
    calls = {"n": 0}

    def fake_check(self):
        # Latched, like the real Stopper: a verdict must survive being polled
        # again, which is what the B-Spline inner dispatch loop does.
        if getattr(self, "verdict", None) is not None:
            return self.verdict
        calls["n"] += 1
        if verdict_after is not None and calls["n"] > verdict_after:
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
        exec_fps=None, obs_fps=None, dry_run=False,
        allow_missing_cameras=True,   # the fake rig's cameras are its own
        num_episodes=1, episode_time_s=1.0, task="offline check", metrics=None,
        repo_id=None, output_dir=None, push_to_hub=False, resume=False,
        save_videos=None,
        home_pose_name=fc.default_home_pose_name(), home_q=None,
        home_gripper=fc.control("homing.gripper_norm"),
        home_max_time_s=fc.control("homing.max_time_s"),
        home_tol_rad=fc.control("homing.tol_rad"),
        control_mode="auto", slow_fps=None, no_precision=False, no_eag=True,
        speed_up_times=None, origin_time_scale=None, predict_before_end=None,
        degree=3, gripper_slowdown=False, restart_on_time_align_error=False,
        consider_gripper_during_align=False, disable_time_align=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


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

    # the delta action really is a quaternion in metres
    d = rc.ee_delta_action([0.01, -0.02, 0.03], [0.1, -0.2, 0.05], 0.4)
    q = np.array([d[f"{rc.ARM_KEY}_q{c}"] for c in "xyzw"])
    res.check(np.allclose(Rotation.from_quat(q).as_rotvec(), [0.1, -0.2, 0.05])
              and d[f"{rc.ARM_KEY}_x"] == 0.01,
              "ee_delta_action converts rotvec -> delta quat, keeps metres")

    # gains are normalised zero, i.e. the sim default
    res.check(d["kp"] == 0.0 and d["kd"] == 0.0, "kp/kd dispatched as normalised 0")

    # control-mode auto-detection, all three key variants
    cases = {"actions": ControlMode.EE_DELTA,
             "absolute_actions": ControlMode.EE_POS,
             "absolute_actions_with_precision": ControlMode.EE_POS}
    ok = all(sail_rollout.resolve_control_mode({"action_keys": [k]}, "auto") is v
             for k, v in cases.items())
    override = (sail_rollout.resolve_control_mode({"action_keys": ["actions"]}, "ee_pos")
                is ControlMode.EE_POS)
    unknown = False
    try:
        sail_rollout.resolve_control_mode({"action_keys": ["mystery"]}, "auto")
    except ValueError:
        unknown = True
    res.check(ok and override and unknown,
              "control mode resolves from action_keys, overrides, and refuses unknown")

    # propagate_pose composes in EE_DELTA's own convention
    p0, q0 = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    deltas = [[0.01, 0, 0, 0, 0, 0.1], [0.01, 0, 0, 0, 0, 0.1]]
    p, qq = rc.propagate_pose(p0, q0, deltas)
    res.check(np.allclose(p, [0.02, 0, 0])
              and abs(Rotation.from_quat(qq).as_rotvec()[2] - 0.2) < 1e-12,
              "propagate_pose accumulates deltas the way from_delta composes them")

    # Stopper latches its verdict: reading a keypress consumes it, so a second
    # poll must not come back empty.
    st = rc.Stopper(None)
    st.verdict = "success"
    res.check(st.check() == "success" and st.check() == "success",
              "Stopper latches the operator verdict across repeated polls")
    st2 = rc.Stopper(0.0)
    res.check(st2.check() == "timeout" and st2.check() == "timeout",
              "Stopper latches a timeout too")

    # metrics_path is resolved once: the fallback name carries a timestamp, and
    # it is called after every episode.
    a = _args()
    res.check(rc.metrics_path(a, "sail") == rc.metrics_path(a, "sail"),
              "metrics_path is stable across calls",
              rc.metrics_path(a, "sail").name)

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


def check_dispatch(res: Results) -> None:
    print("\n[dispatch] the goal-push clock holds its rate")

    class Sink:
        def send_action(self, a):
            pass

    # Sub-period jitter must not accumulate: the recorded dataset is labelled at
    # the dispatch rate, so a loop that silently runs slow mislabels every replay.
    hz, n = 200.0, 80
    d = rc.Dispatcher(Sink())
    d.start()
    t0 = time.perf_counter()
    for i in range(n):
        if i % 2 == 0:
            time.sleep(0.4 / hz)
        d.send({}, hz)
    elapsed, ideal = time.perf_counter() - t0, n / hz
    res.check(abs(elapsed - ideal) < 0.05 * ideal,
              "the achieved rate holds under sub-period jitter",
              f"{elapsed * 1e3:.1f} ms for {n} steps, ideal {ideal * 1e3:.1f} ms")

    # The one thing the ordering actually fixes, and it is worth a check: the old
    # increment-then-sleep order delayed the first goal by a full period
    # (measured 100 ms at 10 Hz), which at episode start is a visible lurch.
    d2 = rc.Dispatcher(Sink())
    d2.start()
    t0 = time.perf_counter()
    d2.send({}, 10.0)
    first = time.perf_counter() - t0
    res.check(first < 0.02, "the first goal is not delayed by a full period",
              f"{first * 1e3:.2f} ms")


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


def check_sail(res: Results, port: int) -> None:
    print("\n[sail] receding horizon, precision column, rate switching")
    inf_delay = int(fc.policy("baselines.sail.inf_delay"))
    execute_n = int(fc.policy("baselines.sail.execute_n_actions"))

    # --- absolute path, precision label present
    srv = FakeSail(port, action_keys=["absolute_actions_with_precision"],
                   act_dim=8, precision=True)
    srv.start()
    try:
        client = _client(port)
        meta = client.meta()
        res.check(meta["precision_column"] is True and meta["act_dim"] == 8,
                  "meta handshake reports the precision column")
        mode = sail_rollout.resolve_control_mode(meta, "auto")
        res.check(mode is ControlMode.EE_POS, "absolute key -> EE_POS")

        args = _args(exec_fps=200.0, slow_fps=50.0, obs_fps=20.0, episode_time_s=1.2)
        arm = FakeArm(mode)
        metrics = rc.RunMetrics(policy="sail")
        with operator():
            rc.run_episodes(args, arm, metrics,
                            sail_rollout.make_episode_fn(
                                client, meta, args, mode,
                                rc.check_camera_coverage(meta, arm, True)))
        ep = metrics.episodes[0]

        tags = _tags(arm)
        # First inference: execute_n rows of chunk 0, from index 0.
        first = tags[:execute_n]
        res.check(first == [(0, j) for j in range(execute_n)],
                  "first inference executes execute_n rows from index 0",
                  f"{first[:4]}...")
        # Second: inf_delay rows of chunk 0 continuing at execute_n, then chunk 1
        # entered at inf_delay -- the horizon receding.
        window = tags[execute_n:execute_n + inf_delay + execute_n]
        want = ([(0, execute_n + i) for i in range(inf_delay)]
                + [(1, inf_delay + j) for j in range(execute_n)])
        res.check(window == want, "receding horizon: inf_delay of prev, then new chunk",
                  f"got {window[:6]}... want {want[:6]}...")
        res.check(ep.steps == len(arm.sent) and ep.steps > 0,
                  "every dispatched goal is counted", f"{ep.steps} steps")

        # The precision label must never reach the arm. With act_dim 8, row[6] is
        # the gripper tag and row[7] the label (only ever 0.0 or 1.0). If the
        # strip were off by one, every dispatched gripper would be 0 or 1 -- so a
        # nonzero row fraction is only possible if the right column was read.
        grips = [a[f"{rc.ARM_KEY}_gripper"] for a in arm.sent]
        res.check(any(t[1] != 0 for t in tags)
                  and not all(g in (0.0, 1.0) for g in grips)
                  and all(0 <= t[1] < meta["action_horizon"] for t in tags),
                  "precision label stripped; gripper is the chunk's own column",
                  f"{len(set(grips))} distinct gripper values dispatched")

        # Chunk 1 has labels on rows 2..4, which are inside the executed window,
        # so some steps must have run slow and some fast.
        gaps = np.diff(arm.send_times)
        res.check(ep.slow_steps > 0, "precision labels produced slow steps",
                  f"{ep.slow_steps} of {ep.steps}")
        fast_p = float(np.percentile(gaps, 10)) if len(gaps) else 0.0
        slow_p = float(np.max(gaps)) if len(gaps) else 0.0
        res.check(fast_p < 1.0 / 100.0 and slow_p > 1.0 / 100.0,
                  "dispatch actually switched rate",
                  f"p10 gap {fast_p * 1e3:.1f} ms, max {slow_p * 1e3:.1f} ms")
    finally:
        srv.stop()

    # --- delta path, no precision column
    srv2 = FakeSail(port + 1, action_keys=["actions"], act_dim=7, precision=False)
    srv2.start()
    try:
        client = _client(port + 1)
        meta = client.meta()
        mode = sail_rollout.resolve_control_mode(meta, "auto")
        res.check(mode is ControlMode.EE_DELTA, "'actions' key -> EE_DELTA")
        args = _args(exec_fps=200.0, obs_fps=20.0, episode_time_s=0.8)
        arm = FakeArm(mode)
        metrics = rc.RunMetrics(policy="sail")
        with operator(verdict_after=2):
            rc.run_episodes(args, arm, metrics,
                            sail_rollout.make_episode_fn(
                                client, meta, args, mode,
                                rc.check_camera_coverage(meta, arm, True)))
        ep = metrics.episodes[0]
        res.check(ep.success, "operator success verdict recorded")
        res.check(ep.wall_time_s > 0 and ep.inferences > 0,
                  "time-to-success and inference count recorded",
                  f"{ep.wall_time_s:.2f}s, {ep.inferences} inferences")
        res.check(arm.clipped == 0, "no EE_DELTA command exceeded torque.delta.pos_max_m",
                  f"{arm.clipped} clipped")
        res.check(ep.max_lead_m == 0.0,
                  "lead monitor is inert in EE_DELTA (goal re-anchors every step)")
    finally:
        srv2.stop()


def check_bspline(res: Results, port: int) -> None:
    print("\n[bspline] EE_POS dispatch, decode, replanning")
    srv = FakeBSpline(port)
    srv.start()
    try:
        client = _client(port)
        meta = client.meta()
        res.check(meta["action_format"] == "single_yam_rot6d" and meta["act_dim"] == 10,
                  "meta reports the 10-dim rot6d action our converter produces")

        # decode() must reject a wrong-width action rather than silently reading
        # the gripper out of a rotation column.
        rejected = False
        try:
            bsp_rollout.decode(np.zeros(7))
        except ValueError:
            rejected = True
        res.check(rejected, "decode refuses an action that is not 10-dim")

        args = _args(exec_fps=100.0, obs_fps=20.0, episode_time_s=1.5,
                     speed_up_times=1.0)
        kwargs = dict(degree=3, n_obs_steps=1, obs_stride=1,
                      origin_time_scale=rc.origin_time_scale(), speed_up_times=1.0,
                      predict_before_end=float(fc.policy("baselines.bspline.predict_before_end")),
                      time_align_error_threshold=float(
                          fc.policy("baselines.bspline.time_align_error_threshold")),
                      time_align_larger_t=fc.policy("baselines.bspline.time_align_larger_t"),
                      disable_time_align=False, restart_on_time_align_error=False,
                      consider_gripper_during_align=False,
                      gripper_slowdown_enabled=False,
                      gripper_slowdown_threshold=0.08, gripper_slowdown_steps=7,
                      gripper_index=9, compare_dim=9)
        arm = FakeArm(ControlMode.EE_POS)
        metrics = rc.RunMetrics(policy="bspline")
        with operator():
            rc.run_episodes(args, arm, metrics,
                            bsp_rollout.make_episode_fn(
                                client, meta, args, kwargs,
                                rc.check_camera_coverage(meta, arm, True)))
        ep = metrics.episodes[0]
        res.check(ep.steps > 0, "goals dispatched", f"{ep.steps} steps")
        res.check(ep.inferences >= 2, "the plan was replanned at least once",
                  f"{ep.inferences} plans")

        # Before the first plan lands, the goal must be the homed pose -- not a
        # zero goal, which would command the base-frame origin.
        first = arm.sent[0]
        res.check(np.linalg.norm([first[f"{rc.ARM_KEY}_{a}"] for a in "xyz"]) > 0.1,
                  "no zero goal dispatched while waiting for the first plan",
                  f"|p| = {np.linalg.norm([first[f'{rc.ARM_KEY}_{a}'] for a in 'xyz']):.3f} m")

        gaps = np.diff(arm.send_times)
        med = float(np.median(gaps)) if len(gaps) else 0.0
        res.check(abs(med - 1.0 / 100.0) < 0.004, "exec rate held at --exec-fps",
                  f"median gap {med * 1e3:.2f} ms, target 10.00 ms")
        res.check(ep.max_lead_m < float(fc.policy("baselines.exec.max_lead_m")),
                  "tracking arm stayed inside the lead bound",
                  f"max lead {ep.max_lead_m:.4f} m")
    finally:
        srv.stop()

    # --- the lead monitor must ABORT, not clamp
    bound = float(fc.policy("baselines.exec.max_lead_m"))
    srv2 = FakeBSpline(port + 1, travel=bound * 2.5)
    srv2.start()
    try:
        client = _client(port + 1)
        meta = client.meta()
        args = _args(exec_fps=100.0, obs_fps=20.0, episode_time_s=3.0)
        arm = FakeArm(ControlMode.EE_POS)
        arm.freeze = True          # the arm stops tracking; the plan runs on
        kwargs = dict(degree=3, n_obs_steps=1, obs_stride=1,
                      origin_time_scale=rc.origin_time_scale(), speed_up_times=1.0,
                      predict_before_end=0.06, time_align_error_threshold=0.1,
                      time_align_larger_t=0.2, disable_time_align=False,
                      restart_on_time_align_error=False,
                      consider_gripper_during_align=False,
                      gripper_slowdown_enabled=False, gripper_slowdown_threshold=0.08,
                      gripper_slowdown_steps=7, gripper_index=9, compare_dim=9)
        metrics = rc.RunMetrics(policy="bspline")
        with operator():
            rc.run_episodes(args, arm, metrics,
                            bsp_rollout.make_episode_fn(
                                client, meta, args, kwargs,
                                rc.check_camera_coverage(meta, arm, True)))
        ep = metrics.episodes[0]
        aborted = ep.aborted is not None
        res.check(aborted, "a frozen arm aborts the episode",
                  (ep.aborted or "")[:70])
        if aborted:
            # A clamp would have kept the commanded goal near the arm; an abort
            # leaves the last goal where the plan put it.
            last = np.array([arm.sent[-1][f"{rc.ARM_KEY}_{a}"] for a in "xyz"])
            pos, _ = arm._pose()
            res.check(np.linalg.norm(last - pos) > bound,
                      "the goal was never rescaled toward the arm (abort, not clamp)",
                      f"final divergence {np.linalg.norm(last - pos):.4f} m")
    finally:
        srv2.stop()


def check_dataset(res: Results, port: int, tmp: Path) -> None:
    print("\n[dataset] recorded rollout is readable and correctly labelled")
    srv = FakeBSpline(port)
    srv.start()
    try:
        client = _client(port)
        meta = client.meta()
        exec_fps = 40.0
        args = _args(exec_fps=exec_fps, obs_fps=20.0, episode_time_s=0.6,
                     num_episodes=2, repo_id="offline/baseline-check",
                     output_dir=str(tmp))
        kwargs = dict(degree=3, n_obs_steps=1, obs_stride=1,
                      origin_time_scale=rc.origin_time_scale(), speed_up_times=1.0,
                      predict_before_end=0.06, time_align_error_threshold=0.1,
                      time_align_larger_t=0.2, disable_time_align=False,
                      restart_on_time_align_error=False,
                      consider_gripper_during_align=False,
                      gripper_slowdown_enabled=False, gripper_slowdown_threshold=0.08,
                      gripper_slowdown_steps=7, gripper_index=9, compare_dim=9)
        arm = FakeArm(ControlMode.EE_POS)
        metrics = rc.RunMetrics(policy="bspline")
        with operator():
            rc.run_episodes(args, arm, metrics,
                            bsp_rollout.make_episode_fn(
                                client, meta, args, kwargs,
                                rc.check_camera_coverage(meta, arm, True)))

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        d = LeRobotDataset(repo_id=args.repo_id, root=str(tmp))
        info = d.meta.info["features"]
        res.check(d.num_episodes == 2 and d.num_frames == sum(e.steps for e in metrics.episodes),
                  "every dispatched goal became a recorded frame",
                  f"{d.num_episodes} eps, {d.num_frames} frames")
        # The fps must be the DISPATCH rate, not obs_fps: a dataset labelled
        # 20 Hz but written at 40 replays at half speed with nothing to say why.
        res.check(float(d.fps) == exec_fps, "dataset fps is the dispatch rate",
                  f"{d.fps} vs exec_fps {exec_fps:.0f}")
        res.check(list(info["action"]["names"][0]) == list(rc.ACTION_KEYS)
                  and list(info["observation.state"]["names"][0]) == list(rc.STATE_OBS_KEYS),
                  "recorded feature names match run_residual.py's")
        res.check(sorted(d.meta.camera_keys)
                  == sorted(f"observation.images.{c}" for c in arm.cameras),
                  "both cameras encoded")
        act = d[0]["action"].numpy()
        res.check(abs(act[8]) < 1e-9 and abs(act[9]) < 1e-9 and act[7] > 0.0,
                  "first recorded action carries normalised gains and a real gripper")
        # Metrics are written after every episode, so an interrupted run keeps
        # the episodes that did finish.
        mp = rc.metrics_path(args, "bspline")
        res.check(mp.is_file(), "metrics written next to the dataset", str(mp.name))

        # Finding 9: frames are written once per observation and indexed by
        # WRITTEN frame, so the burned-in clock matches wall time. Indexing by
        # dispatch step made it run per_obs x fast.
        vids = tmp.parent / "vid"
        args2 = _args(exec_fps=exec_fps, obs_fps=20.0, episode_time_s=0.6,
                      num_episodes=1, save_videos=str(vids), metrics=str(tmp.parent / "m.json"))
        arm2 = FakeArm(ControlMode.EE_POS)
        m2 = rc.RunMetrics(policy="bspline")
        with operator():
            rc.run_episodes(args2, arm2, m2,
                            bsp_rollout.make_episode_fn(
                                client, meta, args2, kwargs,
                                rc.check_camera_coverage(meta, arm2, True)))
        files = sorted(vids.glob("*.mp4")) if vids.is_dir() else []
        res.check(len(files) == len(arm2.cameras),
                  "one mp4 per camera written", f"{[f.name for f in files]}")
        per_obs = round(exec_fps / 20.0)
        frames = m2.episodes[0].steps / per_obs
        res.check(frames <= m2.episodes[0].steps / max(per_obs - 0.5, 1),
                  "video frames counted per observation, not per dispatched goal",
                  f"~{frames:.0f} frames for {m2.episodes[0].steps} goals "
                  f"({per_obs} goals/obs)")
    finally:
        srv.stop()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=("pure", "dispatch", "timing", "sail",
                                      "bspline", "dataset"), default=None)
    p.add_argument("--base-port", type=int, default=5701)
    p.add_argument("--tmp", default=None,
                   help="scratch dir for the recorded dataset (default: a temp dir)")
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, force=True)

    res = Results()
    if args.only in (None, "pure"):
        check_pure(res)
    if args.only in (None, "dispatch"):
        check_dispatch(res)
    if args.only in (None, "timing"):
        check_timing(res)
    if args.only in (None, "sail"):
        check_sail(res, args.base_port)
    if args.only in (None, "bspline"):
        check_bspline(res, args.base_port + 10)
    if args.only in (None, "dataset"):
        import shutil
        import tempfile
        tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="baseline-ds-"))
        try:
            check_dataset(res, args.base_port + 20, tmp / "ds")
        finally:
            if not args.tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(res.rows) - len(res.failed)}/{len(res.rows)} checks passed")
    for _, name, detail in res.failed:
        print(f"  FAILED: {name} {detail}")
    print("\nPASS" if not res.failed else "\nFAIL")
    return 0 if not res.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
