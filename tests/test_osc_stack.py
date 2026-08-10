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
    DAMPING_RATIO_LIMITS, DELTA_POS_MAX, DELTA_ROT_MAX, KP_LIMITS, OSCTorqueController,
    resolve_gains,
)

RNG = np.random.default_rng(20260727)

# Measured on the right arm via libfranka Model.mass at the q below, so tests that
# care about conditioning use a real mass matrix rather than a synthetic SPD one.
REAL_Q = np.array([-0.064549, -0.010316, -0.206631, -2.232503, 0.296601, 2.048460, -0.124818])
REAL_M = np.array([
    [1.426654, 0.097576, 1.399142, -0.015600, -0.039852, -0.046474, 0.002390],
    [0.097576, 2.030185, 0.092572, -0.983097, -0.009118, -0.118825, -0.001640],
    [1.399142, 0.092572, 1.404409, -0.013515, -0.040171, -0.046279, 0.002405],
    [-0.015600, -0.983097, -0.013515, 1.121601, 0.010825, 0.141208, 0.000126],
    [-0.039852, -0.009118, -0.040171, 0.010825, 0.024808, 0.002881, 0.000871],
    [-0.046474, -0.118825, -0.046279, 0.141208, 0.002881, 0.041403, 0.000133],
    [0.002390, -0.001640, 0.002405, 0.000126, 0.000871, 0.000133, 0.001934],
])



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

    def shape_goal(self, goals):
        return goals


# Every config field that reaches the control path, at its sim value. Parity is
# DEFINED at these, so anything tuned on the rig must be pinned here or the suite
# silently starts asserting "the rig matches the rig" -- which is how
# ee_rotation_fudge=0.35 turned 8 of these tests red with the stack untouched.
# test_every_hardware_knob_is_pinned checks this list is still complete.
_SIM_PARITY_KNOBS = {
    "kp_ori_scale": (1.0, 1.0, 1.0),
    "kp_pos_scale": (1.0, 1.0, 1.0),
    "kd_ori_scale": (1.0, 1.0, 1.0),
    "kd_pos_scale": (1.0, 1.0, 1.0),
    "ee_translation_fudge": 1.0,
    "ee_rotation_fudge": 1.0,
    "use_noise": False,
}

# Reach the control loop through make_session, not the robot config.
_SESSION_KNOBS = {"friction_kc", "friction_kc_joint_pos", "friction_kc_joint_neg"}

# Transport, framing and perception: cannot reach the torque path. noise_*_scale
# is control-path but gated by use_noise, which is pinned above.
_INERT_FIELDS = {
    "active_arms", "calibration_dir", "cameras", "control_mode", "depth",
    "depth_cam", "depth_center_arm", "depth_crop_radius_m", "id",
    "noise_pos_scale", "noise_rot_scale",
    "r_gripper_ip", "r_gripper_port", "r_port", "r_robot_ip", "r_server_ip",
    # rig_profile selects arms/cameras/base poses. It reaches the torque path
    # only through the safety screen, which the parity harness bypasses.
    "rig_profile",
    "world_in_robot_quat_wxyz", "world_in_robot_translation_m",
}


def make_robot(case, mode=ControlMode.EE_DELTA, safety=False, **cfg_kw):
    bfmod.BimanualFranka._make_gripper = lambda self, arm: _FakeGripper()
    for knob, value in _SIM_PARITY_KNOBS.items():
        cfg_kw.setdefault(knob, value)
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


def make_session(case, uncouple=True, friction_kc=0.0):
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
    s.clamp_trips = 0
    s.recovery_count = 0
    # uncoupling is an OSCTorqueController constructor arg (osc.py's own), set
    # above; only friction_kc still reaches the loop through the goal block.
    s._test_tuning = dict(friction_kc=friction_kc)
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
    g[_shm.G_FRICTION_KC_POS] = t["friction_kc"]
    g[_shm.G_FRICTION_KC_NEG] = t["friction_kc"]
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


def test_gain_constants_match_the_sim_controller_config():
    """The remap's constants come from the sim's own controller block.

    resolve_gains hardcodes kp=150, kp_limits=[0,1500], damping=1,
    damping_limits=[0,10] and derives exp_scale = limit_max / default the way
    libero.py's wrapper does. Those numbers live in cfg/fast_default.yaml, not
    osc_pose.json (whose "fixed" / kp_limits [0,300] are overridden), so read
    them back rather than trusting the comment.
    """
    import re
    cfg = _REPO / "multi-fast" / "cfg" / "fast_default.yaml"
    if not cfg.exists():
        return "skipped: multi-fast not checked out"
    block = re.search(r"^controller:\n((?:[ \t]+.*\n|\n)+)", cfg.read_text(), re.M)
    assert block, "no controller: block in fast_default.yaml"
    text = block.group(1)

    def num(key):
        m = re.search(rf"^\s+{key}:\s*(\[.*?\]|[-\d.eE+]+)", text, re.M)
        assert m, f"{key} missing from the controller block"
        return eval(m.group(1))  # noqa: S307 -- a float or a 2-list of floats

    kp, kp_lim = num("kp"), num("kp_limits")
    damping, damping_lim = num("damping"), num("damping_limits")

    from lerobot_robot_bimanual_franka.osc_torque_controller import (
        DAMPING_EXP_SCALE, DAMPING_RATIO_LIMITS, DEFAULT_DAMPING_RATIO, DEFAULT_KP, KP_EXP_SCALE,
    )
    assert DEFAULT_KP == kp, f"kp {DEFAULT_KP} vs sim {kp}"
    assert tuple(KP_LIMITS) == tuple(kp_lim), f"kp_limits {KP_LIMITS} vs sim {kp_lim}"
    assert DEFAULT_DAMPING_RATIO == damping, f"damping {DEFAULT_DAMPING_RATIO} vs sim {damping}"
    assert tuple(DAMPING_RATIO_LIMITS) == tuple(damping_lim)
    # libero.py: exp_scale = limits[1] / default, applied as scale ** action.
    assert np.isclose(KP_EXP_SCALE, kp_lim[1] / kp)
    assert np.isclose(DAMPING_EXP_SCALE, damping_lim[1] / damping)
    return f"kp={kp} {kp_lim}, damping={damping} {damping_lim}, exp_scale={KP_EXP_SCALE:g}"


def test_gain_channels_sweep_against_robosuite():
    """kp and kd actions, swept in isolation against robosuite's real set_goal.

    The other tests randomise the gains alongside the delta; this pins the gain
    channel on its own -- both ends of the action range, both channels, at a
    fixed pose error -- so a remap error cannot hide behind a small delta.
    """
    rng = np.random.default_rng(77)
    case = make_case(rng)
    dpos, drot = np.array([0.6, -0.4, 0.3]), np.array([0.2, 0.5, -0.3])
    worst = 0.0
    for a_kp in (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0):
        for a_kd in (-1.0, -0.4, 0.0, 0.4, 1.0):
            ours = our_torque(make_robot(case), make_session(case),
                              ee_delta_action(dpos, drot, a_kp, a_kd), case)
            ref = ref_torque(case, dpos, drot, a_kp, a_kd)
            err = _rel(ours, ref)
            assert err < 1e-5, f"a_kp={a_kp} a_kd={a_kd}: rel err {err:.2e}"
            worst = max(worst, err)

    # The endpoints land exactly on the sim's clip limits, so a policy that
    # saturates its gain channel gets the same stiffness the sim would give.
    kp_hi, kd_hi = resolve_gains(1.0, 1.0)
    kp_lo, _ = resolve_gains(-1.0, 0.0)
    assert np.allclose(kp_hi, KP_LIMITS[1]), f"a_kp=+1 gave {kp_hi[0]}, not {KP_LIMITS[1]}"
    assert np.allclose(kp_lo, 150.0 / 10.0)
    assert np.allclose(kd_hi, 2 * np.sqrt(KP_LIMITS[1]) * 10.0)
    return f"35 (a_kp, a_kd) pairs, worst rel err {worst:.2e}"


def test_kd_scales_damp_without_stiffening():
    """kd_*_scale is the knob for an axis that oscillates; kp_*_scale is not.

    The settling speed is v = (kp/kd)*delta. kp_*_scale re-derives the ratio as
    sqrt(kp6/kp), so kp/kd is invariant and raising it only stiffens an already
    lively axis. kd_*_scale multiplies the damping ratio, which is the only path
    that lowers kp/kd.
    """
    kp0, kd0 = resolve_gains(0.0, 0.0)

    # A kd scale: damping up by exactly the factor, stiffness untouched.
    for s in (1.5, 2.0, 4.0):
        kp, kd = resolve_gains(0.0, 0.0, kd_pos_scale=s)
        assert np.allclose(kp, kp0), "kd_pos_scale moved the stiffness"
        assert np.allclose(kd[:3], kd0[:3] * s), f"kd_pos_scale={s} did not scale kd"
        assert np.allclose(kd[3:], kd0[3:]), "kd_pos_scale leaked into orientation"
        assert np.allclose((kp[:3] / kd[:3]) * s, kp0[:3] / kd0[:3]), "slew rate wrong"

    kp, kd = resolve_gains(0.0, 0.0, kd_ori_scale=3.0)
    assert np.allclose(kd[3:], kd0[3:] * 3.0) and np.allclose(kd[:3], kd0[:3])
    assert np.allclose(kp, kp0), "kd_ori_scale moved the stiffness"

    # By contrast a kp scale cannot calm anything: kp/kd is unchanged.
    kp, kd = resolve_gains(0.0, 0.0, kp_pos_scale=4.0)
    assert np.allclose(kp[:3] / kd[:3], kp0[:3] / kd0[:3]), (
        "kp_pos_scale changed the slew rate; it is a stiffness knob only")
    assert np.all(kp[:3] > kp0[:3])

    # Per-axis, and still inside osc.py's damping_ratio_limits at the extreme.
    kp, kd = resolve_gains(0.0, 0.0, kd_pos_scale=(1.0, 2.0, 3.0))
    assert np.allclose(kd[:3] / kd0[:3], [1.0, 2.0, 3.0])
    _, kd_max = resolve_gains(0.0, 1.0, kd_pos_scale=100.0)   # ratio already at the cap
    assert np.allclose(kd_max[:3], 2 * np.sqrt(150.0) * DAMPING_RATIO_LIMITS[1]), (
        "a kd scale escaped the sim's damping_ratio_limits")
    return "damping-only, per axis, clipped at the sim's damping_ratio_limits"


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


def test_every_hardware_knob_is_pinned():
    """Every config field is classified, and the pinned ones actually land.

    Without this, adding a tuning knob quietly redefines parity as "the rig
    agrees with the rig": the harness inherits the rig's value, feeds it to both
    sides, and the comparison passes while the arm and the sim disagree.
    """
    import dataclasses
    fields = {f.name for f in dataclasses.fields(SingleArmFrankaConfig)}
    unclassified = fields - set(_SIM_PARITY_KNOBS) - _SESSION_KNOBS - _INERT_FIELDS
    assert not unclassified, (
        f"unclassified config fields: {sorted(unclassified)} -- add each to "
        "_SIM_PARITY_KNOBS (pinned at its sim value), _SESSION_KNOBS or _INERT_FIELDS")

    # And the pinning reaches the state send_action reads, not just the config.
    robot = make_robot(make_case(np.random.default_rng(0)))
    assert robot._trans_fudge == 1.0 and robot._rot_fudge == 1.0
    assert np.allclose(robot._kp_ori_scale, 1.0) and np.allclose(robot._kp_pos_scale, 1.0)
    assert robot.config.use_noise is False


def test_defaults_are_exact_parity():
    """Pin the deliberate deviations from osc.py.

    friction_kc cancels a plant term the sim lacks; the gain scales retune a
    block without leaving osc.py's own action space. Those are the only two
    deviations left -- everything else is bit-exact robosuite.
    """
    cfg = SingleArmFrankaConfig(r_server_ip="x", r_robot_ip="y", r_gripper_ip="y", r_port=1,
                                control_mode=ControlMode.EE_DELTA, cameras={}, depth=False,
                                depth_cam={})
    assert cfg.ee_translation_fudge == 1.0 and cfg.ee_rotation_fudge == 1.0, (
        f"delta fudge deviates from sim: translation={cfg.ee_translation_fudge} "
        f"rotation={cfg.ee_rotation_fudge}. The policy's own delta is being "
        "rescaled before it reaches the OSC goal, so the arm tracks a different "
        "command than the sim it was trained in.")
    # Above 1.0 the assist injects more than the friction it cancels, so the net
    # plant friction goes negative, and the band's incremental gain (1 + kc/0.20)
    # multiplies the OSC law -- and the measured dq inside it -- by that much.
    assert 0.0 <= cfg.friction_kc <= 1.0, (
        f"friction_kc={cfg.friction_kc} over-compensates: assist saturates at "
        f"{cfg.friction_kc:.1f}x breakaway against ~0.7x kinetic while sliding, and the "
        f"assist's incremental gain is {1.0 + cfg.friction_kc / 0.20:.0f}x "
        f"(vs {1.0 + 1.0 / 0.20:.0f}x at 1.0, {1.0:.0f}x at exact osc.py)")
    # Tuning is allowed, but only inside osc.py's action space.
    kp_tuned, kd_tuned = resolve_gains(0.0, 0.0, cfg.kp_ori_scale, cfg.kd_ori_scale,
                                       kp_pos_scale=cfg.kp_pos_scale,
                                       kd_pos_scale=cfg.kd_pos_scale)
    assert np.all(kp_tuned <= KP_LIMITS[1] + 1e-9)
    # The invariant that keeps a STIFFNESS tune from becoming a speed tune: the
    # kp scales must leave kp/kd alone, and the kd scales must move it by exactly
    # their own factor and nothing more.
    kp_sim, kd_sim = resolve_gains(0.0, 0.0)
    kd_scale6 = np.concatenate([np.broadcast_to(cfg.kd_pos_scale, 3),
                                np.broadcast_to(cfg.kd_ori_scale, 3)])
    assert np.allclose((kp_tuned / kd_tuned) * kd_scale6, kp_sim / kd_sim), (
        "a gain scale changed the slew rate by something other than kd_*_scale")
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


def test_torque_limits_are_the_only_thing_that_rescales_tau():
    """One layer, and it is _enforce_limits. Anything else that scales tau on its
    way to the joints is a second envelope the tuner cannot see: it makes a gain
    change stop mattering somewhere, with nothing in the log to say where.

    Asserted structurally, because a reintroduced guard would pass every
    numerical test in this file -- it only fires outside their range.
    """
    import inspect
    body = inspect.getsource(srv.ControlLoop.run)
    sends = [ln.strip() for ln in body.splitlines() if "writeOnce" in ln]
    assert len(sends) == 1, f"expected one write path, found {len(sends)}"
    # The tau that is written must come straight out of the limiter.
    assert "tau = self._enforce_limits(" in body
    assert "writeOnce(pylibfranka.Torques(tau.tolist()))" in sends[0]

    for name in dir(srv.ControlLoop):
        assert "guard" not in name and "brake" not in name, (
            f"ControlLoop.{name} looks like a second limit layer")
    for name in dir(srv):
        assert "VELOCITY" not in name and "BRAKE" not in name, (
            f"pylibfranka_control.{name} looks like a velocity envelope")

    # And the client cannot install one.
    from lerobot_robot_bimanual_franka import pylibfranka_server as _srv
    assert set(inspect.signature(_srv._ArmSession.set_tuning).parameters) == {
        "self", "friction_kc"}


def test_friction_feedforward_cannot_drive_an_idle_arm():
    """Zero commanded torque must mean zero assist -- the anti-walking invariant.

    EE_DELTA re-anchors goal_pos on the measured pose, so residual friction is
    the only thing holding the arm at zero command; a dq-signed assist cancels it
    and the arm walks.
    """
    kc = 0.9
    coriolis = np.full(7, 0.4)
    assert np.allclose(srv._friction_feedforward(kc, kc, coriolis.copy(), coriolis), 0.0), (
        "assist is nonzero with no control torque -- the arm can drive itself")

    # Direction follows the command, and it saturates so it can only ever assist.
    for sign in (+1.0, -1.0):
        ff = srv._friction_feedforward(kc, kc, coriolis + sign * 3.0, coriolis)
        assert np.all(sign * ff > 0.0), "assist did not follow the commanded torque"
        assert np.all(np.abs(ff) <= kc * srv._FRICTION_COULOMB + 1e-12)

    # Enough authority to matter: a command at the friction floor must clear it.
    cmd = 0.3 * srv._FRICTION_COULOMB
    total = cmd + srv._friction_feedforward(kc, kc, coriolis + cmd, coriolis)
    assert np.all(total > srv._FRICTION_COULOMB * 0.9), "assist too weak to break away"

    assert np.allclose(srv._friction_feedforward(0.0, 0.0, coriolis + 3.0, coriolis), 0.0)


def test_joint_damping_pole_is_inside_the_law_rate():
    """JOINT_POS / home() / hold damping must stay representable at 500 Hz.

    kp is deliberately not weighted by M (the wrist could not break away
    otherwise), but kd inherited that: at M_jj ~ 0.01-0.05 the raw kd puts the
    velocity pole at 600-1500 rad/s, so -kd*dq alternates sign tick to tick and
    the wrist bang-bangs its torque clamp. That is the homing vibration.
    """
    from lerobot_robot_bimanual_franka.osc_torque_controller import (
        DEFAULT_JOINT_KD, DEFAULT_JOINT_KP, JOINT_KD_POLE_MAX, JointImpedanceController,
    )
    # Representative FR3 inertia diagonal: heavy proximal joints, light wrist.
    M = np.diag([2.0, 2.0, 0.8, 0.6, 0.05, 0.02, 0.01])
    t_eff = srv._CONTROL_DECIMATION * 1e-3 + 1.5e-3     # law period + write-first delay

    ctrl = JointImpedanceController(num_joints=7)
    ctrl.set_goal(goal_q=np.zeros(7), kp=1.0, damping_ratio=1.0)
    dq = np.ones(7)
    tau = ctrl.run_controller(np.zeros(7), dq, M, np.zeros(7), position_hold=True)
    kd_eff = -tau                                        # zero error, zero coriolis

    crit = kd_eff * t_eff / np.diag(M)
    assert np.all(crit < 1.0), f"velocity loop not stable at 500 Hz: kd*T/M = {crit}"

    # Damping only. The stiffness must stay un-weighted or the wrist stalls.
    tau_e = ctrl.run_controller(np.full(7, 0.1), np.zeros(7), M, np.zeros(7),
                                position_hold=True)
    assert np.allclose(tau_e, DEFAULT_JOINT_KP * -0.1), "the cap touched the stiffness"

    # A cap, never a boost, and the heavy joints keep their tuned damping.
    assert np.all(kd_eff <= DEFAULT_JOINT_KD + 1e-9), "the cap raised a joint's damping"
    assert np.allclose(kd_eff[:4], DEFAULT_JOINT_KD[:4]), (
        f"cap bit a joint it should not have: {kd_eff[:4]}")

    zeta = kd_eff / (2.0 * np.sqrt(DEFAULT_JOINT_KP * np.diag(M)))
    assert np.all(zeta > 0.4), f"capped damping left a joint underdamped: {zeta}"

    # DEFAULT_JOINT_KD is now sized to sit just inside the cap at nominal
    # inertia, so the cap is a pose-varying safety net rather than the mechanism.
    # Keep it under test by handing it an inertia well below nominal.
    light = np.diag([2.0, 2.0, 0.8, 0.6, 0.005, 0.002, 0.001])
    tau_l = ctrl.run_controller(np.zeros(7), dq, light, np.zeros(7), position_hold=True)
    assert np.all(-tau_l[4:] < DEFAULT_JOINT_KD[4:]), "the cap did not bite on a light wrist"
    assert np.all(-tau_l * t_eff / np.diag(light) < 1.0), "still unstable at low inertia"
    return (f"kd {np.round(kd_eff, 2)}, zeta {np.round(zeta, 2)}, "
            f"worst kd*T/M {crit.max():.2f} (cap {JOINT_KD_POLE_MAX:.0f} rad/s)")


def test_homing_speed_budget_reaches_every_joint():
    """home()'s ramp must be able to drive every joint at a usable speed.

    The budget is qdot_max = frac*tau_limit/(kd*(1+margin)) -- it falls out of
    kd, so a kd sized for the proximal joints throttles the wrist. At kd=25 that
    was 0.13 rad/s on joint 6, and a 1 rad wrist move could not finish inside the
    5 s default max_time_s: homing reported 'failed to converge' while looking
    converged, because it genuinely had not arrived.
    """
    from lerobot_robot_bimanual_franka.osc_torque_controller import (
        DEFAULT_JOINT_KD, DEFAULT_JOINT_KP, JOINT_TORQUE_LIMITS,
    )
    qdot_max = np.minimum(
        bfmod.HOME_MAX_QDOT,
        bfmod.HOME_TAU_FRACTION * np.asarray(JOINT_TORQUE_LIMITS)
        / (bfmod.HOME_IMPEDANCE_KD * (1.0 + bfmod.HOME_LEAD_MARGIN)))

    assert np.all(qdot_max >= 0.4 * bfmod.HOME_MAX_QDOT), (
        f"a joint is throttled to {np.round(qdot_max, 3)} rad/s; the wrist cannot home")

    # The ramp's lead has to clear breakaway friction, or the joint never starts.
    max_lead = bfmod.HOME_LEAD_MARGIN * qdot_max * DEFAULT_JOINT_KD / DEFAULT_JOINT_KP
    tau_at_lead = DEFAULT_JOINT_KP * max_lead
    assert np.all(tau_at_lead > srv._FRICTION_COULOMB), (
        f"stall lead is under breakaway: {np.round(tau_at_lead, 2)} Nm "
        f"vs {srv._FRICTION_COULOMB} Nm")
    # ...without asking for more torque than the joint has.
    assert np.all(tau_at_lead <= np.asarray(JOINT_TORQUE_LIMITS) * bfmod.HOME_TAU_FRACTION + 1e-9)

    # And a full-scale move fits inside the default timeout with margin.
    slowest = float(np.min(qdot_max))
    assert 1.0 / slowest < 0.6 * 5.0, (
        f"a 1 rad move takes {1.0 / slowest:.1f}s against the 5 s default max_time_s")
    return (f"qdot_max {np.round(qdot_max, 3)} rad/s, "
            f"1 rad worst case {1.0 / slowest:.1f}s")


def test_friction_assist_is_memoryless():
    """The assist must stay a pure function of (kc, tau, coriolis).

    Its incremental gain is 1 + kc/band -- 6x at the defaults -- so it multiplies
    the whole OSC law near the Coriolis level, and tau carries the measured dq.
    That genuinely does amplify encoder noise. The fix is NOT to filter it: a
    first-order lag inside a gain-(1+g) path is a lag network, pole at wc and
    zero at wc(1+g), and a 25 Hz corner costs 39 deg of phase at 52 Hz. Deployed
    to hardware that made the shake worse, not better. Reduce the gain instead
    (_FRICTION_TAU_EPS), or kc, which set_tuning can sweep live.
    """
    kc, coriolis = 0.9, np.zeros(7)
    hi = 3.0 * srv._FRICTION_COULOMB
    lo = -3.0 * srv._FRICTION_COULOMB

    # Same input, same output, whatever came before it.
    first = srv._friction_feedforward(kc, kc, hi, coriolis)
    for _ in range(50):
        srv._friction_feedforward(kc, kc, lo, coriolis)
    assert np.allclose(srv._friction_feedforward(kc, kc, hi, coriolis), first), (
        "the assist carries state across calls -- it must add no phase")

    # A reversal completes in one call rather than being approached over several.
    assert np.allclose(srv._friction_feedforward(kc, kc, lo, coriolis), -first), (
        "reversal is not instantaneous")

    # And the loop holds no assist state of its own.
    assert not hasattr(srv.ControlLoop, "_assist"), (
        "ControlLoop regained assist state; see _friction_feedforward")

    band = srv._FRICTION_TAU_EPS[0] / srv._FRICTION_COULOMB[0]
    return f"pure map, band {band:.2f}*Fc -> {1.0 + kc / band:.1f}x incremental gain"


def test_friction_assist_is_osc_only():
    """Joint impedance relies on friction to stay quiet; assisting there buzzes."""
    case = make_case(np.random.default_rng(11))
    case["dq"] = np.zeros(7)
    session = make_session(case, friction_kc=0.9)
    state = FakeState(case["q"], case["dq"], case["ee_pos"], case["R"])

    g = np.zeros(_shm.GOAL_SIZE)
    g[_shm.G_CMD_SEQ] = 1.0
    g[_shm.G_FRICTION_KC_POS] = 0.9
    g[_shm.G_FRICTION_KC_NEG] = 0.9
    g[_shm.G_JOINT_Q] = case["q"] + 0.05      # a standing hold error
    g[_shm.G_JOINT_KP], g[_shm.G_JOINT_RATIO] = 1.0, 1.0
    for mode in (_shm.MODE_JOINT, _shm.MODE_HOLD):
        session._last_cmd_seq = -1.0
        session._last_mode = None
        g[_shm.G_MODE] = mode
        tau = session._compute_tau(state, g.copy())[0]
        session._last_cmd_seq = -1.0
        session._last_mode = None
        g[_shm.G_FRICTION_KC_POS] = 0.0
        g[_shm.G_FRICTION_KC_NEG] = 0.0
        plain = session._compute_tau(state, g.copy())[0]
        g[_shm.G_FRICTION_KC_POS] = 0.9
        g[_shm.G_FRICTION_KC_NEG] = 0.9
        assert np.allclose(tau, plain), f"friction assist leaked into mode {mode}"


def test_friction_assist_is_directional():
    """Each joint carries two gains, selected by the sign of the COMMANDED torque.

    The hypothesis this exists for: breakaway on this arm differs by which way a
    joint turns, so a single symmetric gain either under-assists the hard
    direction or over-assists the easy one. Over-assist is negative damping, so
    the two failures are not symmetric and one gain cannot avoid both.
    """
    coriolis = np.full(7, 0.4)
    kc_pos = np.array([2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    kc_neg = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    drive = 10.0 * srv._FRICTION_TAU_EPS          # well past the tanh band

    push = srv._friction_feedforward(kc_pos, kc_neg, coriolis + drive, coriolis)
    pull = srv._friction_feedforward(kc_pos, kc_neg, coriolis - drive, coriolis)

    # Joint 0: 2.0 pushing, 0.5 pulling -> a 4x asymmetry, correctly oriented.
    assert push[0] > 0 and pull[0] < 0, "assist must follow the commanded direction"
    assert np.isclose(abs(push[0]) / abs(pull[0]), 4.0, rtol=1e-6), (
        f"expected the 2.0/0.5 gain split, got {abs(push[0]) / abs(pull[0]):.3f}")
    # Joint 1: zero gain one way, full gain the other. A per-direction zero must
    # be reachable -- that is what "this direction needs no help" looks like.
    assert np.isclose(push[1], 0.0), "kc_pos=0 must give no assist in + direction"
    assert abs(pull[1]) > 0.5 * srv._FRICTION_COULOMB[1]
    # Joints 2-6 are symmetric here and must stay symmetric.
    assert np.allclose(push[2:], -pull[2:]), "symmetric joints became asymmetric"


def test_directional_assist_is_continuous_through_zero():
    """Switching gain on the torque sign must not put a step at zero command.

    The gain switches exactly where tanh() is zero, so the product is continuous
    even though the two sides have different slopes. Selecting on dq instead
    would switch where the assist is at full magnitude, which IS a step -- and
    would also cancel the friction holding a still arm.
    """
    coriolis = np.zeros(7)
    kc_pos, kc_neg = np.full(7, 3.0), np.full(7, 0.2)     # a 15x split
    eps = 1e-9
    lo = srv._friction_feedforward(kc_pos, kc_neg, np.full(7, -eps), coriolis)
    hi = srv._friction_feedforward(kc_pos, kc_neg, np.full(7, +eps), coriolis)
    assert np.max(np.abs(hi - lo)) < 1e-6, (
        f"assist jumps by {np.max(np.abs(hi - lo)):.2e} Nm across zero command")
    assert np.allclose(srv._friction_feedforward(kc_pos, kc_neg, coriolis, coriolis), 0.0), (
        "zero command must still mean zero assist -- the anti-walk invariant")


def test_symmetric_directional_gains_reproduce_the_scalar_assist():
    """Equal gains both ways == the old single-gain behaviour, exactly.

    This is what makes the 14 parameters a strict generalisation: the config
    ships both vectors equal, so nothing changes until someone splits them.
    """
    rng = np.random.default_rng(17)
    coriolis = rng.normal(0, 0.3, 7)
    for kc_val in (0.0, 0.5, 1.0, 2.5):
        kc = np.full(7, kc_val)
        for _ in range(5):
            tau = coriolis + rng.normal(0, 2.0, 7)
            got = srv._friction_feedforward(kc, kc, tau, coriolis)
            want = kc * srv._FRICTION_COULOMB * np.tanh(
                (tau - coriolis) / srv._FRICTION_TAU_EPS)
            assert np.allclose(got, want), f"symmetric gains diverged at kc={kc_val}"

    # And the config ships them equal, so today's arm is unchanged.
    import franka_config as fc_
    assert np.allclose(fc_.control("tuning.friction_kc_joint_pos"),
                       fc_.control("tuning.friction_kc_joint_neg")), (
        "the shipped config splits the directions; that is a deliberate rig trim, "
        "but it means the arm no longer matches the symmetric baseline")


def test_directional_gains_reach_the_control_loop():
    """The (2, 7) block survives config -> robot -> goal block -> assist.

    A widened shm field is exactly the kind of change that silently half-lands:
    the writer fills 14 floats and the reader still slices 7.
    """
    assert _shm.G_FRICTION_KC_POS.stop == _shm.G_FRICTION_KC_NEG.start
    assert _shm.G_FRICTION_KC_NEG.stop <= _shm.G_RUNNING, "friction field overruns G_RUNNING"
    assert _shm.G_RUNNING < _shm.GOAL_SIZE

    case = make_case(np.random.default_rng(5))
    pos = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    neg = (0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    robot = make_robot(case, friction_kc=1.0,
                       friction_kc_joint_pos=pos, friction_kc_joint_neg=neg)
    kc = robot.friction_kc
    assert kc.shape == (2, 7), f"friction_kc is {kc.shape}, expected (2, 7)"
    assert np.allclose(kc[0], pos) and np.allclose(kc[1], neg)

    # And it lands in the goal block the loop reads, both rows distinct.
    g = np.zeros(_shm.GOAL_SIZE)
    g[_shm.G_FRICTION_KC_POS] = kc[0]
    g[_shm.G_FRICTION_KC_NEG] = kc[1]
    assert not np.allclose(g[_shm.G_FRICTION_KC_POS], g[_shm.G_FRICTION_KC_NEG])
    return f"(2, 7) through config -> robot -> shm, pos[0]={kc[0][0]} neg[0]={kc[1][0]}"


def test_directional_friction_estimator_and_its_confound():
    """The measurement math behind `measure_joint_friction.py --directional`.

    Synthesises sweeps with a KNOWN asymmetry and a KNOWN model bias and checks
    the estimator recovers them -- including the negative result, which is the
    important half: from a single pose the two are perfectly confounded and no
    amount of averaging separates them. That is why the script needs
    --gravity-flip-joint, and why its plain output is an upper bound.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mjf", str(Path(__file__).resolve().parent.parent
                    / "scripts" / "measure_joint_friction.py"))
    mjf = importlib.util.module_from_spec(spec)
    sys.modules["_mjf"] = mjf
    spec.loader.exec_module(mjf)

    F_pos, F_neg, bias = 1.30, 0.70, 0.25     # ground truth
    sym_true, asym_true = 0.5 * (F_pos + F_neg), 0.5 * (F_pos - F_neg)

    # One pose: tau_up = +F_pos + b, tau_dn = -F_neg + b
    sym, half_sum = mjf.split_direction(F_pos + bias, -F_neg + bias)
    assert np.isclose(sym, sym_true), "symmetric friction must be bias-free"
    assert np.isclose(half_sum, asym_true + bias), "half-sum should carry asym + bias"

    # Single pose -> not separable, and the estimate is asym+bias, an upper bound.
    one = mjf.asymmetry_from_poses([half_sum], [+1.0])
    assert one["separable"] is False
    assert np.isclose(one["asymmetry"], asym_true + bias)
    assert one["asymmetry"] > asym_true, "the confounded figure must overstate, not understate"

    # Two poses with opposite gravity sign -> b flips, A does not, so both fall out.
    s_flip = asym_true - bias
    two = mjf.asymmetry_from_poses([half_sum, s_flip], [+1.0, -1.0])
    assert two["separable"] is True
    assert np.isclose(two["asymmetry"], asym_true), (
        f"recovered asymmetry {two['asymmetry']:.4f}, expected {asym_true:.4f}")
    assert np.isclose(two["bias"], bias), (
        f"recovered bias {two['bias']:.4f}, expected {bias:.4f}")

    # A genuinely symmetric joint under a large bias must NOT read as directional.
    sym2, hs2 = mjf.split_direction(1.0 + 0.9, -1.0 + 0.9)
    clean = mjf.asymmetry_from_poses([hs2, 0.0 - 0.9], [+1.0, -1.0])
    assert np.isclose(clean["asymmetry"], 0.0, atol=1e-9), (
        "a symmetric joint read as directional -- the bias leaked into the estimate")
    return (f"recovered asym {two['asymmetry']:.2f}/bias {two['bias']:.2f} from two poses; "
            f"one pose overstates as {one['asymmetry']:.2f}")


def test_set_tuning_failure_is_fatal_not_a_warning():
    """A failed tuning push must stop the run, not log and continue.

    Server sessions are keyed by robot_ip and outlive their client, so a
    swallowed set_tuning leaves the arm on whatever assist an earlier script
    set. BimanualFranka.connect() pushes unconditionally precisely to stop that
    -- but set_tuning used to route through _push, which downgrades every
    exception to a warning, so the guard did nothing. A real sysid run then
    continued against a stale NUC and would have measured a controller nobody
    configured.

    Every OTHER call in RobotDriver is best-effort on purpose: goal pushes are
    per-tick and the next one corrects a dropped one. This one is not.
    """
    import inspect
    from lerobot_robot_bimanual_franka import franka_process as fp

    # The call, not the word: the docstring explains why _push is wrong here.
    src = inspect.getsource(fp.RobotDriver.set_tuning)
    assert "self._push(" not in src, (
        "set_tuning routes through _push again, which swallows failures into a "
        "warning and lets a run proceed against an unconfigured controller")

    class _AngryRoot:
        def set_tuning(self, *a):
            raise ValueError("operands could not be broadcast together ... (14,) and (7,)")

    d = object.__new__(fp.RobotDriver)
    d.robot_ip, d._root = "1.2.3.4", _AngryRoot()
    try:
        d.set_tuning(friction_kc=np.ones((2, 7)))
    except RuntimeError as e:
        msg = str(e)
    else:
        raise AssertionError("set_tuning swallowed a server-side failure")
    assert "1.2.3.4" in msg, "the error must name which arm failed"
    assert "deploy_nuc_server" in msg, (
        "a 14-element rejection means a stale NUC; the error should say so rather "
        f"than surfacing a raw numpy broadcast message. Got: {msg}")
    return "raises RuntimeError naming the arm and the redeploy"


def test_torque_limit_and_rate_limit():
    """_enforce_limits clamps per joint and respects libfranka's 1000 Nm/s.

    This is now the ONLY bound between the control law and the joints, so a
    regression here has nothing behind it.
    """
    class _Dur:
        def to_sec(self):
            return 1e-3

    huge = np.array([500.0, -500, 500, -500, 500, -500, 500])
    out = srv.ControlLoop._enforce_limits(huge, np.zeros(7), _Dur())
    assert np.all(np.abs(out) <= 1.0 + 1e-12), "rate limit not applied from tau_prev=0"
    out2 = srv.ControlLoop._enforce_limits(huge, huge, _Dur())
    assert np.all(np.abs(out2) <= srv._TAU_LIMIT + 1e-9), "per-joint clamp not applied"
    nan = srv.ControlLoop._enforce_limits(np.full(7, np.nan), np.zeros(7), _Dur())
    assert np.all(np.isfinite(nan)), "NaN torque not sanitised"
    # The clamp is the datasheet limit itself: no safety factor in between, so
    # what the log calls a clamp trip is the joint's real ceiling.
    from lerobot_robot_bimanual_franka.osc_torque_controller import JOINT_TORQUE_LIMITS
    assert np.allclose(srv._TAU_LIMIT, JOINT_TORQUE_LIMITS)


def test_worktable_brake_never_commands_below_the_floor():
    rng = np.random.default_rng(8)
    case = make_case(rng, ee_z=0.12)
    robot = make_robot(case, safety=True)
    robot.send_action(ee_delta_action([0, 0, -1.0], [0, 0, 0]))
    goal_pos, goal_quat = robot.robot_manager.osc_goals[-1]["r"][:2]
    # The floor is a WORLD-frame plane under the EE collision sphere's bottom,
    # not a base-frame goal z: the goal's own orientation decides where the
    # sphere sits, and the arm's base pose lifts it into world.
    bottom = robot.safety.sphere_bottom_world_z("r", goal_pos, goal_quat)
    assert bottom >= robot.safety.goal_z_floor - 1e-12


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
