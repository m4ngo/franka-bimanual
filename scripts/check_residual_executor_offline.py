#!/usr/bin/env python3
"""run_residual.py's executor for an EE_POS base, off the arm.

    python scripts/check_residual_executor_offline.py

The runner executes a LeRobot base the way reach_residual.py executes the
analytic reach base: absolute targets taken relative to the chunk anchor, the
residual summed there with FAST's rule, each target dispatched as the one-step
delta from the measured pose. This pins that equivalence -- the helpers are
compared against reach_residual's own functions, not a restatement of them --
and runs _run_episode against a fake arm that lands on every goal it is sent.
Exits PASS/FAIL.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "residual_wrapper"))
sys.path.insert(0, str(_REPO_ROOT))

import env_wrapper as ew  # noqa: E402
import reach_residual  # noqa: E402
import run_residual  # noqa: E402
from policy_wrapper import BasePolicy, ObservationHistory  # noqa: E402
from viz import EpisodeRecorder  # noqa: E402
from baselines.force_log import WrenchTrace  # noqa: E402

rng = np.random.default_rng(0)
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def rand_pose() -> np.ndarray:
    return np.concatenate([rng.uniform(0.3, 0.6, 3),
                           Rotation.random(random_state=int(rng.integers(1 << 31))).as_quat()])


def rot_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return (Rotation.from_quat(a) * Rotation.from_quat(b).inv()).magnitude() < tol


def check_helpers() -> None:
    print("helpers")
    anchor = rand_pose()
    chunk = np.zeros((10, 10))
    for i in range(10):
        chunk[i, :7] = rand_pose()
        chunk[i, 7] = rng.uniform(0, 1)
    rel = ew.chunk_to_relative(chunk, anchor[:3], anchor[3:7])
    poses = ew.relative_to_poses(rel, anchor[:3], anchor[3:7])
    check("relative -> poses inverts chunk_to_relative, anchor first",
          poses.shape == (11, 7) and np.allclose(poses[0], anchor)
          and np.allclose(poses[1:, :3], chunk[:, :3], atol=1e-6)
          and all(rot_close(poses[i + 1, 3:7], chunk[i, 3:7]) for i in range(10))
          and np.allclose(rel[:, 6:9], chunk[:, 7:10]))

    # reach's slots are [damping, kp, pos(3), rot(3), grip]; ours [pos, rot, grip, kp, kd]
    env = types.SimpleNamespace(osc_output_max=ew._POS_SCALE, osc_rot_output_max=ew._ROT_SCALE)
    reach_slots = np.hstack([rel[:, 8:9], rel[:, 7:8], rel[:, 0:6], rel[:, 6:7]])
    ref = reach_residual.chunk_poses(reach_slots, anchor[:3].astype(np.float64),
                                     Rotation.from_quat(anchor[3:7]), env)
    check("relative_to_poses == reach_residual.chunk_poses",
          np.allclose(ref[:, :3], poses[:, :3], atol=1e-6)
          and all(rot_close(ref[i, 3:], poses[i, 3:]) for i in range(11)))
    now = rand_pose()
    ref_d = reach_residual.target_to_delta(
        poses[3], {"robot0_eef_pos": now[:3], "robot0_eef_quat": now[3:7]}, env)
    check("target_to_delta == reach_residual.target_to_delta",
          np.allclose(ref_d, ew.target_to_delta(poses[3], now[:3], now[3:7])))

    res = rng.uniform(-0.5, 0.5, (5, 9))
    total = ew.compose_chunk(rel, res, bound=10.0)
    expect = rel.copy()
    for cols, rcols in (((0, 3), (2, 5)), ((3, 6), (5, 8))):
        expect[:5, cols[0]:cols[1]] = np.clip(rel[:5, cols[0]:cols[1]] + res[:, rcols[0]:rcols[1]], -10, 10)
    # StudentPredictor: the base contributes zero gains; the gripper sums in robosuite units.
    expect[:5, 7] = res[:, 1]
    expect[:5, 8] = res[:, 0]
    expect[:5, 6] = (1.0 - ((1.0 - 2.0 * rel[:5, 6]) + res[:, 8])) / 2.0
    check("compose_chunk is StudentPredictor's: clip(base + residual), zero base gains, "
          "gripper in robosuite units, residual's steps only",
          np.allclose(total, expect, atol=1e-6) and np.allclose(total[5:], rel[5:]))
    check("the residual sees the gripper in robosuite units (-1 = open)",
          np.allclose(ew.residual_input(np.array([[0] * 6 + [1.0, 0, 0]]))[0, 6], -1.0)
          and np.allclose(ew.residual_input(np.array([[0] * 6 + [0.0, 0, 0]]))[0, 6], 1.0))
    check("compose_chunk without a residual is the base",
          np.allclose(ew.compose_chunk(rel, np.empty((0, 9)), 10.0), rel))
    saturated = ew.compose_chunk(np.full((10, 9), 9.9, dtype=np.float32), np.full((5, 9), 0.5), 10.0)
    check("the composed bound holds", float(saturated.max()) <= 10.0)

    here = rand_pose()
    near, far, turned = here.copy(), here.copy(), here.copy()
    near[:3] += [0.03, 0, 0]
    far[:3] += [0.20, 0, 0]
    turned[3:7] = (Rotation.from_rotvec([0, 0, 0.2]) * Rotation.from_quat(here[3:7])).as_quat()
    a = ew.delta_action(ew.target_to_delta(near, here[:3], here[3:7]), 0.5, 0.0, 0.0)
    check("a target 3 cm away is one 3 cm delta", abs(a["r_x"] - 0.03) < 1e-9 and abs(a["r_qw"] - 1) < 1e-9)
    a = ew.delta_action(ew.target_to_delta(far, here[:3], here[3:7]), 0.5, 0.0, 0.0)
    check("a target 20 cm away saturates at one step", abs(a["r_x"] - ew._POS_SCALE) < 1e-9)
    a = ew.delta_action(ew.target_to_delta(turned, here[:3], here[3:7]), 0.5, 0.0, 0.0)
    check("a rotated target becomes the matching delta quaternion",
          np.allclose(Rotation.from_quat([a["r_qx"], a["r_qy"], a["r_qz"], a["r_qw"]]).as_rotvec(),
                      [0, 0, 0.2], atol=1e-9))


def check_classifier() -> None:
    print("action space from the checkpoint's stats")
    from safetensors.numpy import save_file

    class Stub(BasePolicy):
        def __init__(self, path: Path) -> None:
            self.path = path

    with tempfile.TemporaryDirectory() as tmp:
        for name, lo, hi, want in (("pose", [0.3, -0.1, 0.1], [0.7, 0.1, 0.35], "EE_POS"),
                                   ("delta", [-0.04, -0.04, -0.05], [0.04, 0.04, 0.05], "EE_DELTA")):
            d = Path(tmp) / name
            d.mkdir()
            save_file({"action.min": np.array(lo + [0] * 7, dtype=np.float32),
                       "action.max": np.array(hi + [0] * 7, dtype=np.float32)},
                      str(d / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
            got = Stub(d).action_space()
            check(f"{name} stats -> {want}", got == want, got)


class FakeArm:
    """Lands on every EE_DELTA goal it is sent, composed on the anchor it is handed.

    `drift` moves the arm between an observation and the goal that follows it,
    as the rig's arm moves during the camera reads.
    """

    def __init__(self, drift=(0.0, 0.0, 0.0)) -> None:
        self.pos = np.array([0.45, 0.0, 0.25])
        self.rot = Rotation.from_quat([0, 1, 0, 0])
        self.drift = np.asarray(drift, dtype=np.float64)
        self.steps = 0
        self.sent: list[dict] = []
        self.goals: list[np.ndarray] = []
        self.cameras: dict = {}
        self._cached = None
        self._r_robot_in_world = np.eye(3)
        self._t_robot_in_world = np.zeros(3)
        self.delta_pos = self.delta_rot = None

    def read_kinematic_state(self) -> dict:
        return {"r": (np.zeros(7), np.zeros(7), None, self.pos.copy(),
                      self.rot.as_quat(), np.zeros(6))}

    def get_observation(self) -> dict:
        self._cached = self.read_kinematic_state()
        obs = {f"r_joint_{i}": 0.0 for i in range(1, 8)}
        obs.update(r_gripper=1.0, _step=self.steps, _pos=self.pos.copy(), _quat=self.rot.as_quat())
        self.steps += 1
        self.pos = self.pos + self.drift
        return obs

    kin = property(lambda self: self._cached)
    last_full_point_cloud = property(lambda self: np.zeros((2048, 3), np.float32))
    # |F| equals the number of goals sent so far.
    last_ee_wrench = property(lambda self: {"r": np.array([0, 0, float(len(self.sent)), 0, 0, 0])})

    def cache_delta(self, dpos, drot) -> None:
        self.delta_pos, self.delta_rot = dpos, drot

    def send_action(self, a: dict, anchor: dict | None = None) -> None:
        self._cached = None
        _, _, _, pos, quat, _ = (anchor or self.read_kinematic_state())["r"]
        self.pos = pos + np.array([a["r_x"], a["r_y"], a["r_z"]])
        self.rot = Rotation.from_quat([a["r_qx"], a["r_qy"], a["r_qz"], a["r_qw"]]) * Rotation.from_quat(quat)
        self.goals.append(self.pos.copy())
        self.sent.append(dict(a))


class LineBase(ObservationHistory):
    """Absolute targets 2 cm apart along +x from the pose it observed last; keeps
    the step indices of every window it plans from."""

    n_obs_steps = 2

    def __init__(self) -> None:
        self.windows: list[list[int]] = []

    def infer(self, window: list[dict]) -> np.ndarray:
        self.windows.append([o["_step"] for o in window])
        start, q = window[-1]["_pos"], window[-1]["_quat"]
        return np.array([[*(start + [0.02 * (k + 1), 0, 0]), *q, 0.5, 0.0, 0.0] for k in range(10)])


class ConstResidual:
    """0.4 units (2 cm) of +y on every step, gains kp 0.3 / kd -0.1."""
    center_on_eef = False
    last_network_pcd = None
    composed_action_bound = 10.0
    teacher = None

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def infer(self, obs: dict) -> np.ndarray:
        self.chunks.append(np.array(obs["action_chunk"]))
        r = np.zeros((5, 9))
        r[:, 3], r[:, 1], r[:, 0] = 0.4, 0.3, -0.1
        return r


def check_history() -> None:
    print("observation history")
    h = ObservationHistory()
    h.n_obs_steps = 2
    h.observe("a")
    w1 = h.window()
    h.observe("b")
    w2 = h.window()
    h.observe("c")
    w3 = h.window()
    h.reset()
    w4 = h.window()
    h.observe("d")
    check("padded with the first observation, then one per step, oldest first",
          w1 == ["a", "a"] and w2 == ["a", "b"] and w3 == ["b", "c"])
    check("a window is a copy later observations do not touch", w2 == ["a", "b"])
    check("reset empties it and the next observation pads again", w4 == [] and h.window() == ["d", "d"])


def check_loop() -> None:
    print("_run_episode against a fake arm")
    # No terminal here: raw mode and key polling are stubbed out.
    run_residual.termios = types.SimpleNamespace(tcgetattr=lambda f: None, tcsetattr=lambda *a: None,
                                                 TCSADRAIN=0)
    run_residual.tty = types.SimpleNamespace(setraw=lambda f: None)
    run_residual._stdin_key_pressed = lambda: False

    def run(residual, wrench=None, drift=(0.0, 0.0, 0.0), infer_lead=1):
        arm, base, recorder = FakeArm(drift), LineBase(), EpisodeRecorder()
        run_residual._run_episode(arm, base, residual, dataset=None, episode_time_s=0.5, fps=50.0,
                                  recorder=recorder, infer_lead=infer_lead, wrench=wrench)
        return arm, base, recorder

    wrench = WrenchTrace("r")
    arm, base, _ = run(None, wrench)
    n = len(arm.sent)
    starts = list(range(0, n, ew._CHUNK_EXEC))
    check("one wrench sample per goal sent, read after it",
          [f[2] for f in wrench.force] == list(range(1, n + 1)) and not wrench.missing,
          f"{len(wrench.force)} samples for {n} goals")
    check("one chunk per chunk_exec steps", n >= 10 and len(base.windows) == len(starts),
          f"{n} steps, {len(base.windows)} chunks")
    check("the base plans each chunk from the two latest consecutive steps, as select_action queues them",
          base.windows == [[max(k - 1, 0), k] for k in starts], str(base.windows))
    check("base only: every step is the 2 cm target ahead, orientation held",
          np.allclose([a["r_x"] for a in arm.sent], 0.02, atol=1e-9)
          and np.allclose([a["r_y"] for a in arm.sent], 0)
          and all(abs(a["r_qw"] - 1) < 1e-9 for a in arm.sent))
    check("base only: gripper and gains are the base's",
          all(a["r_gripper"] == 0.5 and a["kp"] == 0.0 and a["kd"] == 0.0 for a in arm.sent))
    check("nothing rides on the robot's cached offset",
          arm.delta_pos is not None and not np.any(arm.delta_pos) and not np.any(arm.delta_rot))

    arm, base, _ = run(None, infer_lead=2)
    starts = list(range(0, len(arm.sent), ew._CHUNK_EXEC))
    check("a prefetched chunk plans from the two consecutive steps ending infer_lead steps early",
          base.windows[1:len(starts)] == [[k - 3, k - 2] for k in starts[1:]], str(base.windows))

    arm, _, recorder = run(None, drift=(0.0, 0.0, 0.004))
    err = np.abs(np.array(arm.goals) - np.array(recorder.total_desired_pos)).max()
    check("an arm moving between observation and send still gets exactly the planned target",
          err < 1e-6, f"max goal error {err * 1e3:.2f} mm")

    residual = ConstResidual()
    arm, base, _ = run(residual)
    ys = [a["r_y"] for a in arm.sent]
    check("residual sees the chunk relative to its anchor (first step 0.4 units of x)",
          all(abs(c[0, 0] - 0.4) < 1e-6 and abs(c[9, 0] - 4.0) < 1e-5 for c in residual.chunks))
    check("residual +y offset lands on the first step of each chunk and holds after",
          all(abs(ys[k] - (0.02 if k % ew._CHUNK_EXEC == 0 else 0.0)) < 1e-9 for k in range(len(ys))))
    check("residual gains drive the arm",
          all(abs(a["kp"] - 0.3) < 1e-6 and abs(a["kd"] + 0.1) < 1e-6 for a in arm.sent))
    check("x still advances 2 cm per step", np.allclose([a["r_x"] for a in arm.sent], 0.02, atol=1e-9))


def main() -> int:
    check_helpers()
    check_classifier()
    check_history()
    check_loop()
    print("PASS" if not failures else f"FAIL: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
