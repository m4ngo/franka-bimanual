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
        self.frames = 0
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

    @property
    def last_ee_wrench(self) -> dict:
        # |F| equals the number of goals sent so far, so a run's max is its step count.
        return {self.k: np.array([0.0, 0.0, float(len(self.sent)), 0.0, 0.0, 0.0])}

    def get_observation(self) -> dict:
        # Paced like a 30 fps camera read; every pixel carries the frame counter.
        time.sleep(1.0 / 30.0)
        self.frames += 1
        obs = {f"{self.k}_joint_{i + 1}": float(v) for i, v in enumerate(self.q)}
        obs[f"{self.k}_gripper"] = self.gripper
        for cam in self.cameras:
            obs[cam] = np.full((*_IMG, 3), self.frames % 256, dtype=np.uint8)
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

    def __init__(self, port, span=None, travel=0.03):
        super().__init__(port)
        self.span = self.SPAN if span is None else span
        self.travel = float(travel)   # metres of x the plan sweeps
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
        record_fps=None, dry_run=False,
        allow_missing_cameras=True,   # the fake rig's cameras are its own
        num_episodes=1, episode_time_s=1.0, task="offline check",
        train_dataset="Offline/check", outputs_root=None, repo_id=None,
        no_record=True, push_to_hub=False, save_videos=False,
        home_pose_name=fc.default_home_pose_name(), home_q=None,
        home_gripper=fc.control("homing.gripper_norm"),
        home_max_time_s=fc.control("homing.max_time_s"),
        home_tol_rad=fc.control("homing.tol_rad"),
        speed=1.0, no_precision=False, no_eag=True,
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

    # SAIL runs osc_kp_scale x the stock kp at the stock damping ratio; B-Spline runs stock
    g = rc.sail_gains()
    kp = float(fc.control("torque.osc.default_kp")) * float(fc.control("torque.osc.gain_exp_base")) ** g["kp"]
    res.check(abs(kp - 2.0 * float(fc.control("torque.osc.default_kp"))) < 1e-9 and g["kd"] == 0.0,
              "SAIL's gain channels resolve to 2x the stock kp at the stock damping ratio",
              f"kp {kp:.1f}, a_kp {g['kp']:.4f}, a_kd {g['kd']}")
    res.check(rc.stock_gains() == {"kp": 0.0, "kd": 0.0},
              "B-Spline's gain channels are the stock controller's")

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
    print("\n[dispatch] env.step(a, control_freq): each goal held for its own 1/hz")

    class Sink:
        last_ee_wrench: dict = {}

        def __init__(self):
            self.t: list[float] = []

        def send_action(self, a):
            self.t.append(time.perf_counter())

    # Each goal is held for ITS OWN step's period: SAIL's slow steps must not
    # land one step late.
    sink = Sink()
    d = rc.Dispatcher(sink)
    rates = [100.0, 20.0, 100.0, 100.0, 20.0, 20.0, 100.0, 50.0]
    t0 = time.perf_counter()
    for hz in rates:
        d.step({}, hz)
    d.wait_hold()
    gaps = np.diff(sink.t + [time.perf_counter()])
    want = 1.0 / np.array(rates)
    res.check(sink.t[0] - t0 < 0.005, "the first goal goes out at once",
              f"{(sink.t[0] - t0) * 1e3:.2f} ms")
    res.check(bool(np.all(np.abs(gaps - want) < 0.004)),
              "every goal is held for 1/hz of its own step",
              " ".join(f"{g * 1e3:.0f}/{w * 1e3:.0f}" for g, w in zip(gaps, want)))

    # Sub-period jitter does not accumulate.
    hz, n = 200.0, 80
    d = rc.Dispatcher(Sink())
    t0 = time.perf_counter()
    for i in range(n):
        if i % 2 == 0:
            time.sleep(0.4 / hz)
        d.step({}, hz)
    d.wait_hold()
    elapsed, ideal = time.perf_counter() - t0, n / hz
    res.check(abs(elapsed - ideal) < 0.05 * ideal,
              "the rate holds under sub-period jitter",
              f"{elapsed * 1e3:.1f} ms for {n} steps, ideal {ideal * 1e3:.1f} ms")

    # After a wait longer than a period (an inference) the next goal still gets its full hold.
    sink = Sink()
    d = rc.Dispatcher(sink)
    d.step({}, 100.0)
    time.sleep(0.05)
    d.step({}, 100.0)
    d.step({}, 100.0)
    res.check(abs((sink.t[2] - sink.t[1]) - 0.01) < 0.003,
              "a goal sent after a long wait is held for a full period, not rushed",
              f"{(sink.t[2] - sink.t[1]) * 1e3:.1f} ms")


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


def _sail_expected_tags(n: int, inf_delay: int, execute_n: int) -> list[tuple[int, int]]:
    """Upstream's (chunk, row) order: chunk 0 from row 0, then per inference
    inf_delay rows of the previous chunk and execute_n of the new one from row inf_delay."""
    out = [(0, j) for j in range(execute_n)]
    prev, k = execute_n, 1
    while len(out) < n:
        out += [(k - 1, prev + i) for i in range(inf_delay)]
        out += [(k, inf_delay + j) for j in range(execute_n)]
        prev, k = inf_delay + execute_n, k + 1
    return out[:n]


def _run_sail(port, args, **server_kw):
    srv = FakeSail(port, action_keys=["absolute_actions_with_precision"], act_dim=8,
                   precision=True, **server_kw)
    srv.start()
    try:
        client = _client(port)
        meta = client.meta()
        s = sail_rollout.settings(meta, args)
        arm = FakeArm(sail_rollout.resolve_control_mode(meta))
        run_dir, record = _open_run(args, "sail")
        record.set("environment", **rc.environment(args, arm))
        record.set("policy", control_mode=ControlMode.EE_POS.value)
        with operator():
            rc.run_episodes(args, arm, run_dir, record,
                            sail_rollout.make_episode_fn(
                                client, s, rc.check_camera_coverage(meta, arm, True)),
                            nominal_hz=s["fast_hz"])
        return srv, arm, s, run_dir, record.episodes[0]
    finally:
        srv.stop()


def check_sail(res: Results, port: int) -> None:
    print("\n[sail] upstream's receding horizon, precision rates, EAG, gains")
    inf_delay = int(fc.policy("baselines.sail.inf_delay"))
    execute_n = int(fc.policy("baselines.sail.execute_n_actions"))
    window = int(fc.policy("baselines.sail.slowdown_window_size"))

    # --- instant inference, precision on, 2x
    args = _args(speed=2.0, episode_time_s=2.0)
    srv, arm, s, run_dir, ep = _run_sail(port, args)
    res.check(s["fast_hz"] == 2.0 * rc.data_fps() and s["slow_hz"] == float(fc.policy("baselines.exec.slow_fps")),
              "--speed 2 sets fast_control_freq to 2x the demonstrations' rate; slow stays 1x",
              f"{s['fast_hz']:.0f}/{s['slow_hz']:.0f} Hz")
    tags = _tags(arm)
    want = _sail_expected_tags(len(tags), inf_delay, execute_n)
    first_bad = next((i for i, (a, b) in enumerate(zip(tags, want)) if a != b), None)
    res.check(len(tags) > 3 * (inf_delay + execute_n) and first_bad is None,
              "rows execute in upstream's order: inf_delay old rows, then the new chunk from row inf_delay",
              f"{len(tags)} steps" if first_bad is None else
              f"step {first_bad}: got {tags[first_bad]}, want {want[first_bad]}")
    kp = {round(a["kp"], 6) for a in arm.sent}
    res.check(kp == {round(rc.sail_gains()["kp"], 6)} and {a["kd"] for a in arm.sent} == {0.0},
              "every goal carries SAIL's 2x-kp gain channels")
    spacing = [b - a for a, b in srv.frame_ids[1:]]
    res.check(all(n == 2 for n in srv.frames_seen) and bool(spacing)
              and all(1 <= d <= 3 for d in spacing),
              "frame stack is two distinct camera frames ~one demonstration step apart",
              f"camera-frame spacing {sorted(set(spacing))}")

    # Each dispatched row's own rate, recomputed with upstream's slowdown rule.
    executed, rates = [], []
    for k, j in tags:
        row = np.zeros(8)
        row[7] = 1.0 if (k % 2 == 1 and 2 <= j <= 4) else 0.0
        fut = np.zeros((16 - j, 8))
        fut[:, 7] = [1.0 if (k % 2 == 1 and 2 <= jj <= 4) else 0.0 for jj in range(j, 16)]
        executed.append(row)
        slow = rc.slowdown_mode(executed, row, fut, window)
        rates.append(s["slow_hz"] if slow else s["fast_hz"])
    gaps = np.diff(arm.send_times)
    want_gaps = 1.0 / np.array(rates[:-1])
    close = np.abs(gaps - want_gaps) < 0.006
    res.check(ep["slow_steps"] == sum(r == s["slow_hz"] for r in rates) and ep["slow_steps"] > 0,
              "slow steps are exactly the ones upstream's window marks",
              f"{ep['slow_steps']} of {ep['steps']}")
    res.check(close.mean() > 0.9,
              "each goal is held 1/hz of its own step's rate",
              f"{close.mean():.0%} within 6 ms; median gap fast "
              f"{1e3 * np.median(gaps[np.array(rates[:-1]) == s['fast_hz']]):.1f} ms, slow "
              f"{1e3 * np.median(gaps[np.array(rates[:-1]) == s['slow_hz']]):.1f} ms")
    res.check(ep["steps"] == len(arm.sent) and ep["steps"] > 0,
              "every dispatched goal is counted", f"{ep['steps']} steps")
    check_force(res, run_dir, ep)
    grips = [a[f"{rc.ARM_KEY}_gripper"] for a in arm.sent]
    res.check(not all(g in (0.0, 1.0) for g in grips),
              "precision label stripped; gripper is the chunk's own column")

    # --- EAG, and an inference slower than the inf_delay rows
    t_f = 4
    args = _args(speed=2.0, episode_time_s=2.5, no_eag=False, no_precision=True)
    srv, arm, s, run_dir, ep = _run_sail(port + 1, args, fac_horizon=t_f, guided=True, delay_s=0.15)
    tags = _tags(arm)
    want = _sail_expected_tags(len(tags), inf_delay, execute_n)
    res.check(tags == want and len(tags) > inf_delay + execute_n,
              "a slow inference holds the arm instead of re-indexing the plan",
              f"{ep['notes'].get('inference_waits')} waits, {ep['notes'].get('inference_wait_s')} s held")
    res.check(ep["notes"].get("inference_waits", 0) > 0,
              "inference slower than inf_delay rows is counted")
    sent = [(i, np.asarray(g)) for i, g in enumerate(srv.guide_seen) if g is not None]
    ok = bool(sent)
    for i, g in sent:
        start = execute_n if i == 1 else inf_delay + execute_n
        got = [(int(round(x * 100)) // 100, int(round(x * 100)) % 100) for x in g[:, 6]]
        ok &= g.shape == (t_f, 8) and got == [(i - 1, start + r) for r in range(t_f)]
    res.check(ok and ep["guided_inferences"] == len(sent),
              "guidance is the previous plan's next fac_horizon rows, full width",
              f"{len(sent)} of {len(srv.guide_seen)} inferences guided")

    # --- the recorded run renders, each chunk drawn from the row it took over at
    from baselines import rollout_viz
    import pandas as pd
    args = _args(speed=2.0, episode_time_s=2.0, no_record=False, train_dataset="Offline/sail-viz")
    _, _, s, run_dir, ep = _run_sail(port + 2, args)
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
           for k, st in enumerate(steps) if st < len(goal)]
    res.check(bool(lag) and max(lag) <= reach,
              "each chunk is logged from the row it takes over at",
              f"goal at the chunk's frame is row {sorted(set(lag))} of the logged chunk")


def _run_bspline(port, args, arm=None, **server_kw):
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
        with operator():
            rc.run_episodes(args, arm, run_dir, record,
                            bsp_rollout.make_episode_fn(
                                client, kwargs, control_freq,
                                rc.check_camera_coverage(meta, arm, True)),
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
              "every goal carries the stock gain channels")
    spacing = [b - a for fr in srv.frames_seen if len(fr) == 2 for a, b in [fr]]
    res.check(bool(spacing) and all(1 <= d <= 3 for d in spacing),
              "the policy's two observations are distinct frames ~one demonstration step apart",
              f"camera-frame spacing {sorted(set(spacing))}")
    res.check(ep["max_lead_m"] < float(fc.policy("baselines.exec.max_lead_m")),
              "tracking arm stayed inside the lead bound", f"max lead {ep['max_lead_m']:.4f} m")

    # --- the lead monitor must ABORT, not clamp
    bound = float(fc.policy("baselines.exec.max_lead_m"))
    arm = FakeArm(ControlMode.EE_POS)
    arm.freeze = True          # the arm stops tracking; the plan runs on
    args = _args(control_freq=control_freq, episode_time_s=3.0)
    _, _, arm, _, _, record = _run_bspline(port + 1, args, arm=arm, travel=bound * 2.5)
    ep = record.episodes[0]
    aborted = ep["aborted"] is not None
    res.check(aborted, "a frozen arm aborts the episode", (ep["aborted"] or "")[:70])
    if aborted:
        last = np.array([arm.sent[-1][f"{rc.ARM_KEY}_{a}"] for a in "xyz"])
        pos, _ = arm._pose()
        res.check(np.linalg.norm(last - pos) > bound,
                  "the goal was never rescaled toward the arm (abort, not clamp)",
                  f"final divergence {np.linalg.norm(last - pos):.4f} m")


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
    p.add_argument("--only", choices=("pure", "outputs", "dispatch", "timing",
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
