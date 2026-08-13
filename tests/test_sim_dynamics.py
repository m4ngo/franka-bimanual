"""sim_dynamics.mass_matrix vs robosuite's own, over the full joint range.

Needs the multi-fast venv (robosuite + mujoco), not the workstation one:

    PYTHONPATH=franka_config multi-fast/.venv/bin/python tests/test_sim_dynamics.py

The point is that `sim_dynamics.py`'s baked model constants are TRANSCRIBED from
robosuite's Panda XML and would otherwise rot silently if robosuite were bumped.
This regenerates the comparison from the live install; it is the only thing standing
between a robosuite upgrade and a control law that quietly stops matching the sim it
claims to emulate.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "lerobot_robot_bimanual_franka"))
# Imported directly: the package __init__ pulls in lerobot, which the sim venv lacks.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "lerobot_robot_bimanual_franka",
                                "lerobot_robot_bimanual_franka"))
import sim_dynamics  # noqa: E402

N_SAMPLES = 1000
TOL = 1e-10


def _env():
    import robosuite
    from robosuite.controllers import load_controller_config
    return robosuite.make(
        "Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20,
        controller_configs=load_controller_config(default_controller="OSC_POSE"))


def main() -> int:
    env = _env()
    env.reset()
    r = env.robots[0]
    c = r.controller
    sim = env.sim
    lo = sim.model.jnt_range[r._ref_joint_indexes][:, 0]
    hi = sim.model.jnt_range[r._ref_joint_indexes][:, 1]

    # 1. the plant constants sim_dynamics advertises must be the ones mujoco has
    got = sim.model.dof_armature[r._ref_joint_vel_indexes]
    assert np.allclose(got, sim_dynamics.ARMATURE), f"armature drifted: {got}"
    got = sim.model.dof_frictionloss[r._ref_joint_vel_indexes]
    assert np.allclose(got, sim_dynamics.FRICTIONLOSS_NM), f"frictionloss drifted: {got}"
    got = sim.model.dof_damping[r._ref_joint_vel_indexes]
    assert np.allclose(got, sim_dynamics.DAMPING_NMS_RAD), f"damping drifted: {got}"
    print("plant constants match mujoco")

    # 2. the mass matrix itself, over the whole reachable range
    rng = np.random.default_rng(0)
    worst = worst_rel = 0.0
    for _ in range(N_SAMPLES):
        q = lo + (hi - lo) * rng.random(7)
        sim.data.qpos[r._ref_joint_pos_indexes] = q
        sim.data.qvel[r._ref_joint_vel_indexes] = 0.0
        sim.forward()
        c.update(force=True)
        ref = np.array(c.mass_matrix)
        err = np.abs(sim_dynamics.mass_matrix(q) - ref).max()
        worst = max(worst, err)
        worst_rel = max(worst_rel, err / np.abs(ref).max())
    print(f"{N_SAMPLES} random q: worst abs {worst:.3e}, worst rel {worst_rel:.3e}")
    assert worst < TOL, f"M_sim disagrees with robosuite by {worst:.3e} (tol {TOL})"

    # 3. qfrc_bias: gravity + Coriolis, the term robosuite clips on top of
    worst = 0.0
    for _ in range(N_SAMPLES):
        q = lo + (hi - lo) * rng.random(7)
        dq = rng.normal(0.0, 1.0, 7)
        sim.data.qpos[r._ref_joint_pos_indexes] = q
        sim.data.qvel[r._ref_joint_vel_indexes] = dq
        sim.forward()
        ref = np.array(sim.data.qfrc_bias[r._ref_joint_vel_indexes])
        worst = max(worst, np.abs(sim_dynamics.bias(q, dq) - ref).max())
    print(f"{N_SAMPLES} random (q, dq): worst |bias - qfrc_bias| {worst:.3e}")
    assert worst < 1e-9, f"bias disagrees with mujoco by {worst:.3e}"
    sim.data.qvel[r._ref_joint_vel_indexes] = 0.0

    # 4. lambda is what the law actually consumes, and it inverts M -- check the
    #    quantity that matters, not just M, at the pose the sysid ladder anchors on
    q = np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.7853981634])
    sim.data.qpos[r._ref_joint_pos_indexes] = q
    sim.data.qvel[r._ref_joint_vel_indexes] = 0.0
    sim.forward()
    c.update(force=True)
    J = np.array(c.J_full)
    for name, M in (("robosuite", np.array(c.mass_matrix)),
                    ("sim_dynamics", sim_dynamics.mass_matrix(q))):
        lam = np.linalg.pinv(J @ np.linalg.inv(M) @ J.T)
        print(f"  lambda_full diag ({name:12s}) {np.round(np.diag(lam), 6)}")
    lam_ref = np.linalg.pinv(J @ np.linalg.inv(np.array(c.mass_matrix)) @ J.T)
    lam_got = np.linalg.pinv(J @ np.linalg.inv(sim_dynamics.mass_matrix(q)) @ J.T)
    err = np.abs(lam_got - lam_ref).max()
    assert err < 1e-8, f"lambda disagrees by {err:.3e}"

    # 4. it has to fit in the 500 Hz law tick
    import time
    sim_dynamics.mass_matrix(q)
    t0 = time.perf_counter()
    for _ in range(2000):
        sim_dynamics.mass_matrix(q)
    us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"mass_matrix: {us:.1f} us/call")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
