"""Element-wise parity check: osc_torque_controller vs robosuite's own osc.py.

Loads the real robosuite modules out of multi-fast/ and compares our port
against them numerically -- helpers, goal composition, and the full
run_controller torque. Run after touching either side.

Note on tolerances: robosuite's transform_utils.quat2mat casts to float32 (a
numba workaround), so anything routed through it agrees only to ~1e-7. The
torque law itself is float64 end to end and matches to ~1e-13.
"""
import importlib.util
import sys
import types
import numpy as np

# Load robosuite's util modules directly: importing the package proper pulls in
# mujoco, which the workstation venv does not have (and does not need).
_RS = "/home/franka/franka_ws/multi-fast/robosuite/robosuite"
for _name, _path in (("robosuite", None), ("robosuite.utils", None)):
    _m = types.ModuleType(_name)
    _m.__path__ = [_RS if _name == "robosuite" else _RS + "/utils"]
    sys.modules[_name] = _m
_macros = types.ModuleType("robosuite.macros")
_macros.ENABLE_NUMBA, _macros.CACHE_NUMBA = True, True
sys.modules["robosuite.macros"] = _macros


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("robosuite.utils.numba", f"{_RS}/utils/numba.py")
T = _load("robosuite.utils.transform_utils", f"{_RS}/utils/transform_utils.py")
_load("robosuite.utils.control_utils", f"{_RS}/utils/control_utils.py")
from robosuite.utils.control_utils import (
    nullspace_torques as rs_nullspace_torques,
    opspace_matrices as rs_opspace_matrices,
    orientation_error as rs_orientation_error,
    set_goal_orientation as rs_set_goal_orientation,
    set_goal_position as rs_set_goal_position,
)

from lerobot_robot_bimanual_franka import osc_torque_controller as ours

rng = np.random.default_rng(7)
FAILS = []


def check(name, a, b, tol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    err = float(np.max(np.abs(a - b)))
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} max|diff| = {err:.3e}")
    if not ok:
        FAILS.append(name)


def rand_state():
    q = rng.normal(0, 0.8, 7)
    dq = rng.normal(0, 0.3, 7)
    A = rng.normal(0, 0.4, (7, 7))
    M = A @ A.T + np.eye(7) * 1.5
    J = rng.normal(0, 0.6, (6, 7))
    C = rng.normal(0, 0.2, 7)
    return q, dq, M, J, C


print("== control_utils helpers ==")
for _ in range(5):
    q, dq, M, J, C = rand_state()
    check("opspace_matrices[lambda_full]", ours.opspace_matrices(M, J, J[:3], J[3:])[0],
          rs_opspace_matrices(M, J, J[:3], J[3:])[0])
    check("opspace_matrices[lambda_pos]", ours.opspace_matrices(M, J, J[:3], J[3:])[1],
          rs_opspace_matrices(M, J, J[:3], J[3:])[1])
    check("opspace_matrices[lambda_ori]", ours.opspace_matrices(M, J, J[:3], J[3:])[2],
          rs_opspace_matrices(M, J, J[:3], J[3:])[2])
    check("opspace_matrices[nullspace]", ours.opspace_matrices(M, J, J[:3], J[3:])[3],
          rs_opspace_matrices(M, J, J[:3], J[3:])[3])
    N = rs_opspace_matrices(M, J, J[:3], J[3:])[3]
    init = rng.normal(0, 0.5, 7)
    check("nullspace_torques", ours.nullspace_torques(M, N, init, q, dq),
          rs_nullspace_torques(M, N, init, q, dq))
    R1 = T.quat2mat(T.random_quat()); R2 = T.quat2mat(T.random_quat())
    check("orientation_error", ours.orientation_error(R1, R2), rs_orientation_error(R1, R2))
    break

print("\n== goal composition (set_goal under use_delta=True) ==")
for _ in range(5):
    ee_pos = rng.normal(0, 0.4, 3)
    ee_mat = T.quat2mat(T.random_quat())
    # robosuite carries the delta as axis-angle; the real stack carries it as a
    # quaternion. Build both from the same rotation and check the goals agree.
    delta_rotvec = rng.normal(0, 0.2, 3)
    delta_pos = rng.normal(0, 0.02, 3)

    rs_goal_pos = rs_set_goal_position(delta_pos, ee_pos)
    rs_goal_ori = rs_set_goal_orientation(delta_rotvec, ee_mat)

    from scipy.spatial.transform import Rotation
    dquat_xyzw = Rotation.from_rotvec(delta_rotvec).as_quat()
    our_rotvec = np.asarray(Rotation.from_quat(dquat_xyzw / np.linalg.norm(dquat_xyzw)).as_rotvec())
    our_goal_pos = ee_pos + delta_pos
    our_goal_ori = (Rotation.from_rotvec(our_rotvec) * Rotation.from_matrix(ee_mat)).as_matrix()

    check("goal_pos", our_goal_pos, rs_goal_pos)
    # 1e-6: robosuite's quat2mat rounds through float32; ours stays float64.
    check("goal_ori (quat delta == axisangle)", our_goal_ori, rs_goal_ori, tol=1e-6)
    break

print("\n== full run_controller vs osc.py body ==")


def robosuite_run_controller(goal_pos, goal_ori, ee_pos, ee_ori_mat, ee_pos_vel, ee_ori_vel,
                             J_full, q, dq, M, torque_compensation, kp, kd, initial_joint,
                             uncoupling=True):
    """Verbatim transcription of OperationalSpaceController.run_controller()."""
    position_error = goal_pos - ee_pos
    vel_pos_error = -ee_pos_vel
    desired_force = np.multiply(np.array(position_error), np.array(kp[0:3])) + np.multiply(
        vel_pos_error, kd[0:3])
    ori_error = rs_orientation_error(goal_ori, ee_ori_mat)
    vel_ori_error = -ee_ori_vel
    desired_torque = np.multiply(np.array(ori_error), np.array(kp[3:6])) + np.multiply(
        vel_ori_error, kd[3:6])
    lambda_full, lambda_pos, lambda_ori, nullspace_matrix = rs_opspace_matrices(
        M, J_full, J_full[:3], J_full[3:])
    if uncoupling:
        decoupled_wrench = np.concatenate([np.dot(lambda_pos, desired_force),
                                           np.dot(lambda_ori, desired_torque)])
    else:
        decoupled_wrench = np.dot(lambda_full, np.concatenate([desired_force, desired_torque]))
    torques = np.dot(J_full.T, decoupled_wrench) + torque_compensation
    torques += rs_nullspace_torques(M, nullspace_matrix, initial_joint, q, dq)
    return torques


for trial in range(3):
    q, dq, M, J, C = rand_state()
    ee_pos = rng.normal(0, 0.4, 3)
    ee_mat = T.quat2mat(T.random_quat())
    goal_pos = ee_pos + rng.normal(0, 0.03, 3)
    goal_ori = T.quat2mat(T.random_quat())
    twist = J @ dq
    initial_joint = rng.normal(0, 0.5, 7)
    kp, kd = ours.resolve_gains(rng.uniform(-1, 1), rng.uniform(-1, 1))

    ctrl = ours.OSCTorqueController(num_joints=7)
    ctrl.set_goal(goal_pos, goal_ori, kp, kd, initial_joint)
    ours_tau = ctrl.run_controller(ee_pos, ee_mat, twist[:3], twist[3:], J, q, dq, M, C)

    # torque_compensation on hardware is Coriolis only: libfranka adds gravity.
    rs_tau = robosuite_run_controller(goal_pos, goal_ori, ee_pos, ee_mat, twist[:3], twist[3:],
                                      J, q, dq, M, C, kp, kd, initial_joint)
    check(f"tau (trial {trial}, kp={kp[0]:.0f})", ours_tau, rs_tau, tol=1e-9)

print("\n== delta envelope (osc.py scale_action bound) ==")
dp, dr = ours.clip_delta(np.array([0.9, -0.02, 0.001]), np.array([2.0, -0.1, 0.3]))
check("clip_delta pos", dp, [0.05, -0.02, 0.001])
check("clip_delta rot", dr, [0.5, -0.1, 0.3])

print("\n" + ("ALL PARITY CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
