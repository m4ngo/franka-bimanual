"""Full-stack equivalence tests: this stack vs robosuite's real OSC controller.

These deliberately do NOT test osc_torque_controller in isolation. Each case runs
a policy action through the whole path that hardware sees --

    BimanualFranka.send_action        (gains remap, delta scaling, goal composition,
                                       goal-orientation rule, safety shaping)
      -> MultiRobotWrapper goal push  (wire encoding)
      -> _ArmSession.set_osc_goal     (quat -> matrix, mode switch)
      -> _ArmSession._compute_tau     (Jacobian, mass matrix, the control law)
      -> _ArmSession._limit           (clamp + torque rate limit)

-- and compares the torque against robosuite's actual
``OperationalSpaceController`` class fed the equivalent normalized action.

Two things are stubbed, neither of which is under test: mujoco (robosuite's
Controller base imports it, but update() is overridden to inject real state) and
pylibfranka (the server imports it at module scope; the compute path only needs
a RobotState-shaped object and a model).

Run standalone (no pytest needed):

    python tests/test_osc_stack.py
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parent.parent
_RS = str(_REPO / "multi-fast" / "robosuite" / "robosuite")
sys.path.insert(0, str(_REPO))


# --------------------------------------------------------------------------
# robosuite's real controller, loaded without mujoco
# --------------------------------------------------------------------------

def _load_robosuite():
    mj = types.ModuleType("mujoco")
    mj.mj_fullM = lambda *a, **k: None
    sys.modules["mujoco"] = mj
    for name, path in (("robosuite", _RS), ("robosuite.utils", _RS + "/utils"),
                       ("robosuite.controllers", _RS + "/controllers")):
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
    macros = types.ModuleType("robosuite.macros")
    macros.ENABLE_NUMBA, macros.CACHE_NUMBA, macros.SIMULATION_TIMESTEP = True, True, 0.002
    sys.modules["robosuite.macros"] = macros

    def load(modname, path):
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    load("robosuite.utils.numba", f"{_RS}/utils/numba.py")
    load("robosuite.utils.transform_utils", f"{_RS}/utils/transform_utils.py")
    load("robosuite.utils.control_utils", f"{_RS}/utils/control_utils.py")
    load("robosuite.controllers.base_controller", f"{_RS}/controllers/base_controller.py")
    return load("robosuite.controllers.osc", f"{_RS}/controllers/osc.py")


_osc_mod = _load_robosuite()
from robosuite.utils.control_utils import orientation_error as rs_orientation_error  # noqa: E402


# --------------------------------------------------------------------------
# pylibfranka stub, so the real server module imports on the workstation
# --------------------------------------------------------------------------

def _stub_pylibfranka():
    pf = types.ModuleType("pylibfranka")

    class _Torques:
        def __init__(self, tau):
            self.tau_J = list(tau)

    class _RealtimeConfig:
        kEnforce = 0
        kIgnore = 1

    pf.Torques = _Torques
    pf.RealtimeConfig = _RealtimeConfig
    pf.Robot = object
    sys.modules["pylibfranka"] = pf


_stub_pylibfranka()

from lerobot_robot_bimanual_franka import ControlMode, SingleArmFranka, SingleArmFrankaConfig  # noqa: E402
import lerobot_robot_bimanual_franka.bimanual_franka as bfmod  # noqa: E402
from lerobot_robot_bimanual_franka import pylibfranka_control as srv  # noqa: E402
from lerobot_robot_bimanual_franka import pylibfranka_shm as _shm  # noqa: E402
from lerobot_robot_bimanual_franka.franka_jacobian import fk_chain, zero_jacobian  # noqa: E402
from lerobot_robot_bimanual_franka.osc_torque_controller import (  # noqa: E402
    DELTA_POS_MAX, DELTA_ROT_MAX, KP_LIMITS, OSCTorqueController, resolve_gains,
)

RNG = np.random.default_rng(20260727)


# --------------------------------------------------------------------------
# Robot-state fixtures: self-consistent q / M / J / pose
# --------------------------------------------------------------------------

class FakeState:
    """RobotState-shaped, with libfranka's column-major O_T_EE."""

    def __init__(self, q, dq, ee_pos, R):
        self.q = tuple(float(x) for x in q)
        self.dq = tuple(float(x) for x in dq)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, ee_pos
        self.O_T_EE = tuple(float(x) for x in T.flatten(order="F"))
        self.tau_J_d = (0.0,) * 7


class FakeModel:
    def __init__(self, M, C):
        self._M, self._C = M, C

    def mass(self, _state):
        return tuple(float(x) for x in self._M.flatten(order="F"))

    def coriolis(self, _state):
        return tuple(float(x) for x in self._C)


def make_case(rng, ee_z=0.45):
    """A physically self-consistent state: pose from FK of q, SPD mass matrix."""
    q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.6]) + rng.normal(0, 0.25, 7)
    dq = rng.normal(0, 0.2, 7)
    chain = fk_chain(q)
    ee_pos = chain[7][:3, 3].copy()
    ee_pos[2] = ee_z                        # keep clear of the worktable brake
    R = chain[7][:3, :3].copy()
    A = rng.normal(0, 0.35, (7, 7))
    M = A @ A.T + np.eye(7) * 1.2
    C = rng.normal(0, 0.15, 7)
    J = zero_jacobian(q, ee_pos_base=ee_pos)
    return dict(q=q, dq=dq, ee_pos=ee_pos, R=R, M=M, C=C, J=J)


# --------------------------------------------------------------------------
# Our stack, end to end
# --------------------------------------------------------------------------

class _CapturingManager:
    def __init__(self, case):
        self.case = case
        self.osc_goals, self.joint_goals = [], []
        self.drivers = {"r": None}
        self.tuning = {}

    def _snap(self):
        c = self.case
        return (c["q"], c["dq"], c["J"], c["ee_pos"],
                Rotation.from_matrix(c["R"]).as_quat(), c["J"] @ c["dq"])

    def current_kinematic_state_batch(self, names, timeout_s=5.0):
        return {n: self._snap() for n in names}

    def current_kinematic_state(self, name, timeout_s=5.0):
        return self._snap()

    def move_osc_goal_batch(self, goals):
        self.osc_goals.append(goals)

    def move_joint_goal_batch(self, goals):
        self.joint_goals.append(goals)

    def move_joint_velocity_batch(self, vels, kd_scale=1.0):
        pass

    def set_tuning_all(self, **kw):
        self.tuning.update(kw)

    def stop_all_motion(self):
        pass

    def shutdown(self):
        pass

    @property
    def num_alive(self):
        return 1


class _FakeGripper:
    GRIPPER_TRUE_MAX_MM = 80.0
    position = 40.0

    def move(self, *a, **k):
        pass

    def home(self):
        return True

    def close(self):
        pass


class _PassThroughSafety:
    """ActionSafetyScreen is an addition robosuite has no equivalent of, so the
    parity tests must not run through it. It has its own test below."""

    goal_z_floor = -np.inf

    def shape_goal(self, goals, kin, kp, kd):
        return goals

    def shape_joint_goal(self, goals, kin, kp, damping_ratio=1.0):
        return goals


def make_robot(case, mode=ControlMode.EE_DELTA, safety=False, **cfg_kw):
    bfmod.BimanualFranka._make_gripper = lambda self, arm: _FakeGripper()
    # Pin sim gains: inheriting the tuned config would rewrite what parity means.
    cfg_kw.setdefault("kp_ori_scale", (1.0, 1.0, 1.0))
    cfg_kw.setdefault("kp_pos_scale", (1.0, 1.0, 1.0))
    cfg = SingleArmFrankaConfig(
        r_server_ip="x", r_robot_ip="y", r_gripper_ip="y", r_port=1,
        control_mode=mode, cameras={}, depth=False, depth_cam={}, **cfg_kw,
    )
    robot = SingleArmFranka(cfg)
    robot.robot_manager = _CapturingManager(case)
    robot._home_q["r"] = case["q"].copy()
    robot._osc_goal_ori["r"] = Rotation.from_matrix(case["R"])
    if not safety:
        robot.safety = _PassThroughSafety()
    return robot


def make_session(case, uncouple=True, friction_kc=0.0, joint_damping_kv=0.0):
    """A real ControlLoop with only the fields the compute path touches.

    The loop now runs in its own process (pylibfranka_control), so the law under
    test lives there; pylibfranka_server is a shared-memory proxy with no control
    code in it at all.
    """
    s = object.__new__(srv.ControlLoop)
    s.robot_ip = "test"
    s.ch = None
    s.model = FakeModel(case["M"], case["C"])
    s.osc = OSCTorqueController(num_joints=7, uncouple_pos_ori=uncouple)
    s.joint = srv.JointImpedanceController(num_joints=7)
    s._last_cmd_seq = -1.0
    s._goal_ts = 0.0
    s._last_mode = None
    s._stale = False
    s._last_J = None
    s._last_J_q = None
    s._last_tau = np.zeros(7)
    s._guard_ts = 0.0
    s.guard_trips = 0
    s.recovery_count = 0
    # Tuning reaches the loop through the goal block, so stash it for _goal_block.
    s._test_tuning = dict(uncouple=uncouple, friction_kc=friction_kc,
                          joint_damping_kv=joint_damping_kv)
    return s


def _goal_block(session, goal_pos, goal_quat, kp, kd, ns):
    """Build the shared-memory goal the control process would have read."""
    g = np.zeros(_shm.GOAL_SIZE)
    t = session._test_tuning
    g[_shm.G_MODE] = _shm.MODE_OSC
    g[_shm.G_CMD_SEQ] = session._last_cmd_seq + 1.0     # force a fresh apply
    g[_shm.G_POS] = goal_pos
    g[_shm.G_QUAT] = goal_quat
    g[_shm.G_KP] = kp
    g[_shm.G_KD] = kd
    g[_shm.G_NULLSPACE] = np.zeros(7) if ns is None else ns
    g[_shm.G_UNCOUPLE] = 1.0 if t["uncouple"] else 0.0
    g[_shm.G_FRICTION_KC] = t["friction_kc"]
    g[_shm.G_JOINT_DAMPING_KV] = t["joint_damping_kv"]
    return g


def our_torque(robot, session, action, case, ignore_action=False):
    """Drive the real send_action -> goal block -> _compute_tau path."""
    robot.send_action(dict(action), ignore_action=ignore_action)
    g = _goal_block(session, *robot.robot_manager.osc_goals[-1]["r"])
    state = FakeState(case["q"], case["dq"], case["ee_pos"], case["R"])
    tau, _q, _dq, _ = session._compute_tau(state, g)
    return tau


# --------------------------------------------------------------------------
# robosuite reference
# --------------------------------------------------------------------------

class _FakeSim:
    def forward(self):
        pass


class RefOSC(_osc_mod.OperationalSpaceController):
    def __init__(self, case, **kw):
        self._case = case
        super().__init__(
            sim=_FakeSim(), eef_name="eef",
            actuator_range=(-np.full(7, 1e4), np.full(7, 1e4)),
            joint_indexes={"joints": list(range(7)), "qpos": list(range(7)), "qvel": list(range(7))},
            **kw,
        )

    def update(self, force=False):
        c = self._case
        tw = c["J"] @ c["dq"]
        self.ee_pos = np.array(c["ee_pos"])
        self.ee_ori_mat = np.array(c["R"])
        self.ee_pos_vel, self.ee_ori_vel = tw[:3], tw[3:]
        self.joint_pos, self.joint_vel = np.array(c["q"]), np.array(c["dq"])
        self.J_pos, self.J_ori, self.J_full = c["J"][:3], c["J"][3:], np.array(c["J"])
        self.mass_matrix = np.array(c["M"])
        self.new_update = False

    @property
    def torque_compensation(self):
        # libfranka gravity-compensates internally; the hardware analogue of
        # mujoco's qfrc_bias is the Coriolis term alone.
        return np.array(self._case["C"])


def ref_torque(case, dpos_norm, drot_norm, a_kp, a_kd, uncouple=True, absolute=None):
    ctrl = RefOSC(
        case, input_max=1, input_min=-1,
        output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
        output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
        kp=150, damping_ratio=1, impedance_mode="variable",
        kp_limits=(0, 1500), damping_ratio_limits=(0, 10),
        control_ori=True, control_delta=absolute is None, uncouple_pos_ori=uncouple,
    )
    ctrl.initial_joint = np.array(case["q"])
    # The libero wrapper's exponential remap is what produces robosuite's raw gains.
    kp_raw = np.full(6, 150.0 * 10.0 ** a_kp)
    damp_raw = np.full(6, 1.0 * 10.0 ** a_kd)
    if absolute is None:
        ctrl.set_goal(np.concatenate([damp_raw, kp_raw, dpos_norm, drot_norm]))
    else:
        ctrl.set_goal(np.concatenate([damp_raw, kp_raw, absolute[:3], absolute[3:]]))
    return ctrl.run_controller()


def ee_delta_action(dpos_norm, drot_norm, a_kp=0.0, a_kd=0.0):
    """The action BimanualFranka expects: deltas already in output units."""
    dq = Rotation.from_rotvec(np.asarray(drot_norm) * DELTA_ROT_MAX).as_quat()
    dp = np.asarray(dpos_norm) * DELTA_POS_MAX
    return {"r_x": dp[0], "r_y": dp[1], "r_z": dp[2],
            "r_qx": dq[0], "r_qy": dq[1], "r_qz": dq[2], "r_qw": dq[3],
            "r_gripper": 0.0, "kp": a_kp, "kd": a_kd}


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1.0)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_gains_match_the_sim_wrapper():
    """resolve_gains == libero.py exponential remap + osc.py's clip."""
    for a_kp, a_kd in [(-1, -1), (-0.5, 0.3), (0, 0), (0.5, -0.5), (1, 1), (0.2, 0.9)]:
        kp, kd = resolve_gains(a_kp, a_kd)
        ref_kp = np.clip(150.0 * 10.0 ** a_kp, 0, 1500)
        ref_ratio = np.clip(1.0 * 10.0 ** a_kd, 0, 10)
        assert np.allclose(kp, ref_kp), (a_kp, kp, ref_kp)
        assert np.allclose(kd, 2 * np.sqrt(ref_kp) * ref_ratio), (a_kd, kd)


def test_delta_scaling_matches_scale_action():
    """Our clip-at-output-bound == osc.py's clip-then-scale."""
    ref = RefOSC(make_case(np.random.default_rng(0)), input_max=1, input_min=-1,
                 output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
                 output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
                 kp=150, damping_ratio=1, impedance_mode="variable",
                 kp_limits=(0, 1500), damping_ratio_limits=(0, 10))
    from lerobot_robot_bimanual_franka.osc_torque_controller import clip_delta
    rng = np.random.default_rng(5)
    for _ in range(50):
        raw = rng.uniform(-3, 3, 6)
        scaled = ref.scale_action(raw)
        ours_p, ours_r = clip_delta(raw[:3] * DELTA_POS_MAX, raw[3:] * DELTA_ROT_MAX)
        assert np.allclose(np.concatenate([ours_p, ours_r]), scaled, atol=1e-12)


def test_ee_delta_torque_matches_robosuite():
    """Whole EE_DELTA path vs robosuite's real controller, many states/actions."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for i in range(40):
        case = make_case(rng)
        dpos = np.zeros(3) if i % 5 == 4 else rng.uniform(-1, 1, 3)
        drot = np.zeros(3) if i % 2 == 0 else rng.uniform(-1, 1, 3)
        a_kp, a_kd = rng.uniform(-1, 1), rng.uniform(-1, 1)

        robot = make_robot(case)
        session = make_session(case)
        ours = our_torque(robot, session, ee_delta_action(dpos, drot, a_kp, a_kd), case)
        ref = ref_torque(case, dpos, drot, a_kp, a_kd)
        worst = max(worst, _rel(ours, ref))
        assert _rel(ours, ref) < 1e-5, f"trial {i}: rel err {_rel(ours, ref):.2e}"
    assert worst < 1e-5
    return f"worst relative error {worst:.2e} over 40 states"


def test_ee_pos_torque_matches_robosuite():
    """EE_POS: absolute pose goal, control_delta=False on the reference."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for i in range(20):
        case = make_case(rng)
        target_p = case["ee_pos"] + rng.uniform(-0.05, 0.05, 3)
        target_r = Rotation.from_rotvec(rng.uniform(-0.3, 0.3, 3)) * Rotation.from_matrix(case["R"])
        a_kp, a_kd = rng.uniform(-1, 1), rng.uniform(-1, 1)
        tq = target_r.as_quat()

        robot = make_robot(case, mode=ControlMode.EE_POS)
        session = make_session(case)
        action = {"r_x": target_p[0], "r_y": target_p[1], "r_z": target_p[2],
                  "r_qx": tq[0], "r_qy": tq[1], "r_qz": tq[2], "r_qw": tq[3],
                  "r_gripper": 0.0, "kp": a_kp, "kd": a_kd}
        ours = our_torque(robot, session, action, case)

        # robosuite absolute mode takes pos + axis-angle.
        absolute = np.concatenate([target_p, target_r.as_rotvec()])
        ref = ref_torque(case, None, None, a_kp, a_kd, absolute=absolute)
        worst = max(worst, _rel(ours, ref))
        assert _rel(ours, ref) < 1e-5, f"trial {i}: rel err {_rel(ours, ref):.2e}"
    return f"worst relative error {worst:.2e} over 20 states"


AXIS_LABELS = ("+X", "+Y", "+Z", "roll(X)", "pitch(Y)", "yaw(Z)")


def _per_axis_sweep(axis_index, is_rotation, uncouple, rng, trials=4):
    """Worst relative torque error for one isolated axis, swept over sign,
    magnitude, gains and robot configuration."""
    worst = 0.0
    for sign in (+1.0, -1.0):
        for mag in (0.1, 0.5, 1.0):
            for _ in range(trials):
                case = make_case(rng)
                dpos, drot = np.zeros(3), np.zeros(3)
                (drot if is_rotation else dpos)[axis_index] = sign * mag
                a_kp, a_kd = rng.uniform(-1, 1), rng.uniform(-1, 1)

                robot = make_robot(case)
                session = make_session(case, uncouple=uncouple)
                ours = our_torque(robot, session,
                                  ee_delta_action(dpos, drot, a_kp, a_kd), case)
                ref = ref_torque(case, dpos, drot, a_kp, a_kd, uncouple=uncouple)
                err = _rel(ours, ref)
                assert err < 1e-5, (
                    f"{AXIS_LABELS[axis_index + (3 if is_rotation else 0)]} "
                    f"sign={sign:+.0f} mag={mag} uncouple={uncouple}: rel err {err:.2e}"
                )
                worst = max(worst, err)
    return worst


def test_each_translation_axis_matches_robosuite():
    """+X, +Y, +Z in isolation, both signs, both uncouple_pos_ori settings."""
    rng = np.random.default_rng(41)
    notes = []
    for uncouple in (True, False):
        per = [_per_axis_sweep(i, False, uncouple, rng) for i in range(3)]
        notes.append("uncouple=%s " % uncouple
                     + " ".join(f"{AXIS_LABELS[i]} {per[i]:.1e}" for i in range(3)))
    return " | ".join(notes)


def test_each_rotation_axis_matches_robosuite():
    """roll, pitch, yaw in isolation -- the axes that stalled on hardware."""
    rng = np.random.default_rng(42)
    notes = []
    for uncouple in (True, False):
        per = [_per_axis_sweep(i, True, uncouple, rng) for i in range(3)]
        notes.append("uncouple=%s " % uncouple
                     + " ".join(f"{AXIS_LABELS[i + 3]} {per[i]:.1e}" for i in range(3)))
    return " | ".join(notes)


def test_each_axis_goal_pose_matches_robosuite():
    """The goal itself, per axis, not just the torque it produces.

    A goal built on the wrong axis could still yield a matching torque if the
    reference were fed the same wrong goal, so check the composition directly
    against set_goal_position / set_goal_orientation.
    """
    rng = np.random.default_rng(43)
    for axis in range(6):
        for sign in (+1.0, -1.0):
            case = make_case(rng)
            dpos, drot = np.zeros(3), np.zeros(3)
            if axis < 3:
                dpos[axis] = sign * 0.7
            else:
                drot[axis - 3] = sign * 0.7

            robot = make_robot(case)
            robot.send_action(ee_delta_action(dpos, drot))
            goal_p, goal_q = robot.robot_manager.osc_goals[-1]["r"][:2]

            ref = RefOSC(case, input_max=1, input_min=-1,
                         output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
                         output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
                         kp=150, damping_ratio=1, impedance_mode="variable",
                         kp_limits=(0, 1500), damping_ratio_limits=(0, 10))
            ref.set_goal(np.concatenate([np.full(6, 1.0), np.full(6, 150.0), dpos, drot]))

            assert np.allclose(goal_p, ref.goal_pos, atol=1e-12), (
                f"{AXIS_LABELS[axis]} sign={sign:+.0f}: goal_pos {goal_p} vs {ref.goal_pos}")
            ours_R = Rotation.from_quat(goal_q).as_matrix()
            assert np.max(np.abs(ours_R - ref.goal_ori)) < 1e-5, (
                f"{AXIS_LABELS[axis]} sign={sign:+.0f}: goal_ori diverged")


def test_each_axis_moves_the_commanded_direction():
    """Sanity on the physics, not just agreement.

    With uncouple_pos_ori=False the task-space response is exactly the commanded
    acceleration, so a single-axis goal must produce acceleration on that axis
    and zero on the other five.

    Note the rotation channel is kp * orientation_error(), and osc.py's
    orientation_error is the sin-based 0.5*sum(rc_i x rd_i) formula -- for a
    single-axis rotation of angle t it returns sin(t), not t. At the 0.5 rad
    envelope bound that is a 4% shortfall. Faithful to sim, so expected here.
    """
    rng = np.random.default_rng(44)
    for axis in range(6):
        case = make_case(rng)
        case["dq"] = np.zeros(7)          # no damping term, isolate the kp response
        dpos, drot = np.zeros(3), np.zeros(3)
        if axis < 3:
            dpos[axis] = 1.0
        else:
            drot[axis - 3] = 1.0

        session = make_session(case, uncouple=False)
        tau = our_torque(make_robot(case), session, ee_delta_action(dpos, drot), case)
        accel = case["J"] @ np.linalg.inv(case["M"]) @ (tau - case["C"])

        if axis < 3:
            want = np.concatenate([dpos * DELTA_POS_MAX * 150.0, np.zeros(3)])
        else:
            goal_R = Rotation.from_rotvec(drot * DELTA_ROT_MAX).as_matrix() @ case["R"]
            want = np.concatenate([np.zeros(3), 150.0 * rs_orientation_error(goal_R, case["R"])])
        assert np.allclose(accel, want, rtol=1e-6, atol=1e-6), (
            f"{AXIS_LABELS[axis]}: accel {np.round(accel, 3)} vs expected {np.round(want, 3)}")

        # The commanded channel must dominate; the other five must be zero.
        on_axis = accel[axis]
        off = np.delete(accel, axis)
        assert abs(on_axis) > 1e-6, f"{AXIS_LABELS[axis]}: no response on its own axis"
        assert np.max(np.abs(off)) < 1e-6 * max(abs(on_axis), 1.0), (
            f"{AXIS_LABELS[axis]}: cross-axis leak {np.max(np.abs(off)):.2e}")


def test_orientation_error_is_sin_based_like_robosuite():
    """Pin the behaviour the previous test's expectation depends on: our port
    and osc.py agree that a t-radian single-axis error reports sin(t)."""
    R0 = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    from lerobot_robot_bimanual_franka.osc_torque_controller import orientation_error as ours_oe
    for t in (0.05, 0.2, 0.5, 1.0):
        for axis in range(3):
            v = np.zeros(3); v[axis] = t
            goal = Rotation.from_rotvec(v).as_matrix() @ R0
            ours = ours_oe(goal, R0)
            theirs = rs_orientation_error(goal, R0)
            assert np.allclose(ours, theirs, atol=1e-12), (t, axis)
            assert np.isclose(ours[axis], np.sin(t), atol=1e-9), (
                f"expected sin({t})={np.sin(t):.4f}, got {ours[axis]:.4f}")


def _advance(case, tau, dt=0.01, dq_cap=1.5):
    """Integrate the arm forward one step and re-derive the pose from FK.

    Makes the trajectory test a genuine walk through configuration space rather
    than the same static comparison repeated. Both controllers are handed this
    same evolved state, so the comparison stays fair; M and C are held fixed
    because their physical accuracy is irrelevant to an equivalence check.
    """
    ddq = np.linalg.inv(case["M"]) @ (tau - case["C"])
    case["dq"] = np.clip(case["dq"] + ddq * dt, -dq_cap, dq_cap)
    case["q"] = case["q"] + case["dq"] * dt
    chain = fk_chain(case["q"])
    case["ee_pos"] = chain[7][:3, 3].copy()
    case["R"] = chain[7][:3, :3].copy()
    case["J"] = zero_jacobian(case["q"], ee_pos_base=case["ee_pos"])


def test_ee_delta_trajectory_over_all_axis_combinations():
    """A whole EE_DELTA trajectory: every combination of translation and rotation
    axis signs, driven step by step, with the arm state evolving between steps.

    This is the test that would catch a divergence the single-step cases cannot:
    goal_ori is carried across steps, so any disagreement about when it is
    rewritten compounds down the trajectory instead of cancelling.
    """
    rng = np.random.default_rng(61)
    case = make_case(rng)
    robot = make_robot(case)
    session = make_session(case)
    ref = RefOSC(case, input_max=1, input_min=-1,
                 output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
                 output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
                 kp=150, damping_ratio=1, impedance_mode="variable",
                 kp_limits=(0, 1500), damping_ratio_limits=(0, 10))
    ref.initial_joint = case["q"].copy()

    levels = (-1.0, 0.0, 1.0)
    combos = [(np.array(t, dtype=float), np.array(r, dtype=float))
              for t in itertools.product(levels, repeat=3)
              for r in itertools.product(levels, repeat=3)]
    assert len(combos) == 3 ** 6 == 729

    worst, worst_at = 0.0, None
    worst_goal_ori = 0.0
    for step, (dpos, drot) in enumerate(combos):
        a_kp, a_kd = float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3))
        kp_raw, damp_raw = np.full(6, 150.0 * 10.0 ** a_kp), np.full(6, 1.0 * 10.0 ** a_kd)

        ours = our_torque(robot, session, ee_delta_action(dpos, drot, a_kp, a_kd), case)
        ref.set_goal(np.concatenate([damp_raw, kp_raw, dpos, drot]))
        theirs = ref.run_controller()

        err = _rel(ours, theirs)
        if err > worst:
            worst, worst_at = err, (step, dpos.astype(int).tolist(), drot.astype(int).tolist())
        assert err < 1e-5, f"step {step} dpos={dpos} drot={drot}: rel err {err:.2e}"

        # The carried goal orientation must stay in lockstep too.
        ours_R = Rotation.from_quat(robot.robot_manager.osc_goals[-1]["r"][1]).as_matrix()
        goal_err = float(np.max(np.abs(ours_R - ref.goal_ori)))
        worst_goal_ori = max(worst_goal_ori, goal_err)
        assert goal_err < 1e-5, f"step {step}: goal_ori diverged by {goal_err:.2e}"

        _advance(case, ours)

    assert np.all(np.isfinite(case["q"])), "trajectory left the valid region"
    return (f"729 steps, worst tau {worst:.2e} at step {worst_at[0]} "
            f"dpos={worst_at[1]} drot={worst_at[2]}, worst goal_ori {worst_goal_ori:.2e}")


def test_ee_delta_trajectory_uncoupled_variant():
    """Same trajectory shape under uncouple_pos_ori=False, the setting teleop
    actually runs, over a shuffled walk so the state history differs."""
    rng = np.random.default_rng(62)
    case = make_case(rng)
    robot = make_robot(case)
    session = make_session(case, uncouple=False)
    ref = RefOSC(case, input_max=1, input_min=-1,
                 output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
                 output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
                 kp=150, damping_ratio=1, impedance_mode="variable",
                 kp_limits=(0, 1500), damping_ratio_limits=(0, 10),
                 uncouple_pos_ori=False)
    ref.initial_joint = case["q"].copy()

    levels = (-1.0, 0.0, 1.0)
    combos = [(np.array(t, dtype=float), np.array(r, dtype=float))
              for t in itertools.product(levels, repeat=3)
              for r in itertools.product(levels, repeat=3)]
    rng.shuffle(combos)

    worst = 0.0
    for step, (dpos, drot) in enumerate(combos):
        ours = our_torque(robot, session, ee_delta_action(dpos, drot), case)
        ref.set_goal(np.concatenate([np.full(6, 1.0), np.full(6, 150.0), dpos, drot]))
        theirs = ref.run_controller()
        err = _rel(ours, theirs)
        worst = max(worst, err)
        assert err < 1e-5, f"step {step} dpos={dpos} drot={drot}: rel err {err:.2e}"
        _advance(case, ours)
    return f"729 shuffled steps, uncouple=False, worst tau {worst:.2e}"


def test_uncouple_flag_matches_robosuite_both_ways():
    rng = np.random.default_rng(3)
    for uncouple in (True, False):
        for i in range(10):
            case = make_case(rng)
            dpos, drot = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            robot = make_robot(case)
            session = make_session(case, uncouple=uncouple)
            ours = our_torque(robot, session, ee_delta_action(dpos, drot), case)
            ref = ref_torque(case, dpos, drot, 0.0, 0.0, uncouple=uncouple)
            assert _rel(ours, ref) < 1e-5, f"uncouple={uncouple} trial {i}"


def test_goal_orientation_held_across_zero_rotation_steps():
    """osc.py rewrites goal_ori only on a nonzero rotation delta; so must we."""
    rng = np.random.default_rng(4)
    case = make_case(rng)
    robot = make_robot(case)
    ref = RefOSC(case, input_max=1, input_min=-1,
                 output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
                 output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
                 kp=150, damping_ratio=1, impedance_mode="variable",
                 kp_limits=(0, 1500), damping_ratio_limits=(0, 10))
    kp_raw, damp_raw = np.full(6, 150.0), np.full(6, 1.0)

    seq = [([0.3, 0, 0], [0, 0, 0.4]), ([0.3, 0, 0], [0, 0, 0]),
           ([0, 0.2, 0], [0, 0, 0]), ([0, 0, 0.1], [0.2, 0, 0]), ([0, 0, 0], [0, 0, 0])]
    for dpos, drot in seq:
        robot.send_action(ee_delta_action(dpos, drot))
        ref.set_goal(np.concatenate([damp_raw, kp_raw, dpos, drot]))
        ours_q = robot.robot_manager.osc_goals[-1]["r"][1]
        ours_R = Rotation.from_quat(ours_q).as_matrix()
        # robosuite's quat2mat rounds through float32, hence the loose tol.
        assert np.max(np.abs(ours_R - ref.goal_ori)) < 1e-5, (dpos, drot)


def test_nullspace_reference_is_the_home_configuration():
    rng = np.random.default_rng(6)
    case = make_case(rng)
    robot = make_robot(case)
    robot.send_action(ee_delta_action([0.1, 0, 0], [0, 0, 0]))
    ns = robot.robot_manager.osc_goals[-1]["r"][4]
    assert np.allclose(ns, case["q"])


def test_control_decimation_matches_robosuite_substep_rate():
    """The law must run at the sim's rate, not the robot's.

    robosuite calls run_controller() once per mujoco substep -- 500 Hz at
    macros.SIMULATION_TIMESTEP = 2 ms -- while set_goal fires at the 20 Hz
    policy rate. Our RT loop ticks at 1 kHz, so it must recompute every 2nd
    tick to land on the same 500 Hz. Running it every tick is both less
    faithful and ~150 us/tick we do not have.
    """
    assert srv._CONTROL_DECIMATION == 2
    assert 1000.0 / srv._CONTROL_DECIMATION == 1.0 / 0.002


def test_impedance_mode_is_variable_not_fixed():
    """Gains must come from the action every step.

    Under osc.py's "fixed" mode the gain channels are ignored, so the two modes
    are distinguishable by whether changing action kp changes the torque. The
    sysid sim reference was generated under "fixed" -- identical at action 0,
    but a gain sweep against it compares a varying real arm to a constant sim.
    """
    from lerobot_robot_bimanual_franka.osc_torque_controller import IMPEDANCE_MODE
    assert IMPEDANCE_MODE == "variable"

    rng = np.random.default_rng(71)
    case = make_case(rng)
    lo = our_torque(make_robot(case), make_session(case),
                    ee_delta_action([0.5, 0, 0], [0, 0, 0], a_kp=-0.5), case)
    hi = our_torque(make_robot(case), make_session(case),
                    ee_delta_action([0.5, 0, 0], [0, 0, 0], a_kp=+0.5), case)
    assert not np.allclose(lo, hi), "action kp had no effect -- this is fixed-mode behaviour"

    # And the ratio is the gain ratio: kp scales the position-error term.
    kp_lo, _ = resolve_gains(-0.5, 0.0)
    kp_hi, _ = resolve_gains(+0.5, 0.0)
    assert np.isclose(kp_hi[0] / kp_lo[0], 10.0), "exponential remap is not 10**a"


def test_defaults_are_exact_parity():
    """Every knob that alters the CONTROL LAW must be a no-op by default.

    friction_kc is not one: it cancels a plant term the sim lacks. It still
    defaults to 0 server-side, so an unconfigured client gets the raw law.
    """
    cfg = SingleArmFrankaConfig(r_server_ip="x", r_robot_ip="y", r_gripper_ip="y", r_port=1,
                                control_mode=ControlMode.EE_DELTA, cameras={}, depth=False,
                                depth_cam={})
    assert cfg.uncouple_pos_ori is True
    assert cfg.ee_translation_fudge == 1.0 and cfg.ee_rotation_fudge == 1.0
    assert 0.0 <= cfg.friction_kc <= 1.0
    # Tuning is allowed, but only inside osc.py's action space.
    kp_tuned, kd_tuned = resolve_gains(0.0, 0.0, cfg.kp_ori_scale,
                                       kp_pos_scale=cfg.kp_pos_scale)
    assert np.all(kp_tuned <= KP_LIMITS[1] + 1e-9)
    # The invariant that keeps a stiffness tune from becoming a speed tune.
    kp_sim, kd_sim = resolve_gains(0.0, 0.0)
    assert np.allclose(kp_tuned / kd_tuned, kp_sim / kd_sim), (
        "a per-axis gain scale changed the slew rate, not just the stiffness")
    from lerobot_robot_bimanual_franka import pylibfranka_server as _srv
    assert _srv._FRICTION_KC == 0.0
    assert _srv._JOINT_DAMPING_KV == 0.0
    kp, kd = resolve_gains(0.0, 0.0)
    assert np.allclose(kp, 150.0) and np.allclose(kd, 2 * np.sqrt(150.0))


def test_hardware_knobs_only_perturb_when_enabled():
    """kp_ori_scale / friction leave translation untouched and are bounded."""
    rng = np.random.default_rng(7)
    case = make_case(rng)
    case["dq"] = np.zeros(7)   # with motion, kd_ori legitimately changes tau
    base = our_torque(make_robot(case), make_session(case),
                      ee_delta_action([0.5, 0, 0], [0, 0, 0]), case)
    scaled = our_torque(make_robot(case, kp_ori_scale=(1.0, 1.0, 10.0)), make_session(case),
                        ee_delta_action([0.5, 0, 0], [0, 0, 0]), case)
    # Pure translation with a held (zero-error) orientation: ori gains cannot matter.
    assert _rel(scaled, base) < 1e-9, "kp_ori_scale changed a pure-translation torque"

    fr = our_torque(make_robot(case), make_session(case, friction_kc=0.6),
                    ee_delta_action([0.5, 0, 0], [0, 0, 0]), case)
    bound = 0.6 * srv._FRICTION_COULOMB
    assert np.all(np.abs(fr - base) <= bound + 1e-9), "friction exceeded its saturation bound"


def test_speed_guard_cuts_back_then_brakes():
    """The last-resort guard. Not client-tunable by design: every exposed knob
    (kp, kp_ori_scale, uncouple_pos_ori, the fudges, friction_kc) multiplies
    commanded torque and several compose, so bounding the command is not enough
    -- a sweep once demanded ~110 Nm against a 69.6 Nm clamp, and a saturated
    clamp IS maximum-force motion. This bounds the resulting speed instead.
    """
    case = make_case(np.random.default_rng(0))
    tau = np.full(7, 60.0)
    trip = srv._JOINT_VELOCITY_LIMITS * srv._JOINT_VELOCITY_TRIP

    # Joint-speed term in isolation (no Jacobian -> no EE term).
    s = make_session(case)
    s._last_J = None
    assert np.allclose(srv.ControlLoop._speed_guard(s, tau.copy(), trip * 0.5), tau)
    prev = np.inf
    for f in (1.05, 1.2, 1.35, 1.49):
        mag = float(np.max(np.abs(srv.ControlLoop._speed_guard(s, tau.copy(), trip * f))))
        assert mag < prev, f"cutback not monotonic at {f}x"
        prev = mag
    for f in (1.6, 3.0, 10.0):
        dq = trip * f
        out = srv.ControlLoop._speed_guard(s, tau.copy(), dq)
        assert np.all(out * dq <= 1e-9), f"did not brake at {f}x"
        assert np.all(np.abs(out) <= srv._TAU_LIMIT + 1e-9), (
            f"brake torque {np.max(np.abs(out)):.1f} Nm exceeds the clamp at {f}x")
    assert s.guard_trips > 0

    # EE speed trips independently, and SHOULD fire before joint limits do --
    # a modest joint speed near an extended pose is a fast end effector.
    s2 = make_session(case)
    s2._last_J = case["J"]
    assert np.allclose(srv.ControlLoop._speed_guard(s2, tau.copy(), np.zeros(7)), tau)
    dq_ee = np.linalg.pinv(case["J"]) @ np.array([2.0, 0, 0, 0, 0, 0])   # 2 m/s EE
    out = srv.ControlLoop._speed_guard(s2, tau.copy(), dq_ee)
    assert not np.allclose(out, tau), "EE-speed term did not trip"
    assert np.all(np.abs(out) <= srv._TAU_LIMIT + 1e-9)


def test_speed_guard_clears_the_sim_action_envelope():
    """The guard must sit OUTSIDE the motion a +/-1 sim action asks for.

    Inside it, the guard rescales the control law instead of bounding a runaway.
    """
    kp, kd = resolve_gains(0.0, 0.0)
    v_lin = (kp[0] / kd[0]) * DELTA_POS_MAX
    v_ang = (kp[3] / kd[3]) * DELTA_ROT_MAX
    worst = 0.0
    for seed in range(24):
        case = make_case(np.random.default_rng(seed))
        s = make_session(case)
        s._last_J = case["J"]
        J_pinv = np.linalg.pinv(case["J"])
        for axis in range(6):
            twist = np.zeros(6)
            twist[axis] = v_lin if axis < 3 else v_ang
            dq = J_pinv @ twist
            if np.max(np.abs(dq) / srv._JOINT_VELOCITY_LIMITS) > 0.98:
                continue          # the arm's own rated velocity, not our envelope
            tau = np.full(7, 5.0)
            out = srv.ControlLoop._speed_guard(s, tau.copy(), dq)
            assert np.allclose(out, tau), (
                f"guard fired inside the sim envelope (seed {seed}, axis {axis}, "
                f"max dq/limit {np.max(np.abs(dq) / srv._JOINT_VELOCITY_LIMITS):.2f})")
            worst = max(worst, float(np.max(np.abs(dq) / srv._JOINT_VELOCITY_LIMITS)))
    return f"sim-parity peak joint speed {worst:.2f} of rated, guard trips at {srv._JOINT_VELOCITY_TRIP}"


def test_friction_feedforward_cannot_drive_an_idle_arm():
    """Zero commanded torque must mean zero assist -- the anti-walking invariant.

    EE_DELTA re-anchors goal_pos on the measured pose, so residual friction is
    the only thing holding the arm at zero command; a dq-signed assist cancels it
    and the arm walks.
    """
    kc = 0.9
    coriolis = np.full(7, 0.4)
    assert np.allclose(srv._friction_feedforward(kc, coriolis.copy(), coriolis), 0.0), (
        "assist is nonzero with no control torque -- the arm can drive itself")

    # Direction follows the command, and it saturates so it can only ever assist.
    for sign in (+1.0, -1.0):
        ff = srv._friction_feedforward(kc, coriolis + sign * 3.0, coriolis)
        assert np.all(sign * ff > 0.0), "assist did not follow the commanded torque"
        assert np.all(np.abs(ff) <= kc * srv._FRICTION_COULOMB + 1e-12)

    # Enough authority to matter: a command at the friction floor must clear it.
    cmd = 0.3 * srv._FRICTION_COULOMB
    total = cmd + srv._friction_feedforward(kc, coriolis + cmd, coriolis)
    assert np.all(total > srv._FRICTION_COULOMB * 0.9), "assist too weak to break away"

    assert np.allclose(srv._friction_feedforward(0.0, coriolis + 3.0, coriolis), 0.0)


def test_friction_assist_is_osc_only():
    """Joint impedance relies on friction to stay quiet; assisting there buzzes."""
    case = make_case(np.random.default_rng(11))
    case["dq"] = np.zeros(7)
    session = make_session(case, friction_kc=0.9)
    state = FakeState(case["q"], case["dq"], case["ee_pos"], case["R"])

    g = np.zeros(_shm.GOAL_SIZE)
    g[_shm.G_CMD_SEQ] = 1.0
    g[_shm.G_FRICTION_KC] = 0.9
    g[_shm.G_JOINT_Q] = case["q"] + 0.05      # a standing hold error
    g[_shm.G_JOINT_KP], g[_shm.G_JOINT_RATIO] = 1.0, 1.0
    for mode in (_shm.MODE_JOINT, _shm.MODE_HOLD):
        session._last_cmd_seq = -1.0
        session._last_mode = None
        g[_shm.G_MODE] = mode
        tau = session._compute_tau(state, g.copy())[0]
        session._last_cmd_seq = -1.0
        session._last_mode = None
        g[_shm.G_FRICTION_KC] = 0.0
        plain = session._compute_tau(state, g.copy())[0]
        g[_shm.G_FRICTION_KC] = 0.9
        assert np.allclose(tau, plain), f"friction assist leaked into mode {mode}"


def test_speed_guard_is_not_reachable_from_the_client():
    """set_tuning must not expose the safety envelope."""
    import inspect
    from lerobot_robot_bimanual_franka import pylibfranka_server as _srv
    sig = inspect.signature(_srv._ArmSession.set_tuning)
    for p in sig.parameters:
        assert "veloc" not in p and "guard" not in p and "brake" not in p, p


def test_torque_limit_and_rate_limit():
    """_limit clamps per joint and respects libfranka's 1000 Nm/s."""
    class _Dur:
        def to_sec(self):
            return 1e-3

    huge = np.array([500.0, -500, 500, -500, 500, -500, 500])
    out = srv.ControlLoop._limit(huge, np.zeros(7), _Dur())
    assert np.all(np.abs(out) <= 1.0 + 1e-12), "rate limit not applied from tau_prev=0"
    out2 = srv.ControlLoop._limit(huge, huge, _Dur())
    assert np.all(np.abs(out2) <= srv._TAU_LIMIT + 1e-9), "per-joint clamp not applied"
    nan = srv.ControlLoop._limit(np.full(7, np.nan), np.zeros(7), _Dur())
    assert np.all(np.isfinite(nan)), "NaN torque not sanitised"


def test_worktable_brake_never_commands_below_the_floor():
    rng = np.random.default_rng(8)
    case = make_case(rng, ee_z=0.12)
    robot = make_robot(case, safety=True)
    robot.send_action(ee_delta_action([0, 0, -1.0], [0, 0, 0]))
    goal_z = robot.robot_manager.osc_goals[-1]["r"][0][2]
    assert goal_z >= robot.safety.goal_z_floor - 1e-12


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            note = fn()
            print(f"  PASS  {name}" + (f"   [{note}]" if note else ""))
        except AssertionError as e:
            failed.append(name)
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f"  FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
