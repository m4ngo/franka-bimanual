#!/usr/bin/env python3
"""End-to-end equivalence: robosuite's real OSC class vs this stack's pipeline.

Not a transcription check -- this instantiates the actual
robosuite.controllers.osc.OperationalSpaceController, feeds it the same robot
state, and diffs the torque against what BimanualFranka + pylibfranka_server
produce for the same policy action. It therefore covers set_goal (action
scaling, the gains remap, the goal-orientation rule) as well as run_controller.

mujoco is stubbed: Controller.update() is overridden to inject the real robot
state, so no simulator is ever touched.

    python scripts/check_osc_e2e.py            # synthetic states
    python scripts/check_osc_e2e.py --live     # pull one state off the arm
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

_RS = "/home/franka/franka_ws/multi-fast/robosuite/robosuite"


def _load_robosuite():
    """Load robosuite's controller modules without importing mujoco."""
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


osc_mod = _load_robosuite()

from lerobot_robot_bimanual_franka.osc_torque_controller import (  # noqa: E402
    DELTA_POS_MAX, DELTA_ROT_MAX, OSCTorqueController, clip_delta, resolve_gains,
)


class _FakeSim:
    def forward(self):
        pass


class RealStateOSC(osc_mod.OperationalSpaceController):
    """robosuite's controller, reading a supplied robot state instead of a sim."""

    def __init__(self, state, **kw):
        self._state = state
        super().__init__(
            sim=_FakeSim(), eef_name="eef", actuator_range=(-np.full(7, 1e3), np.full(7, 1e3)),
            joint_indexes={"joints": list(range(7)), "qpos": list(range(7)), "qvel": list(range(7))},
            **kw,
        )

    def update(self, force=False):
        s = self._state
        self.ee_pos = np.array(s["ee_pos"])
        self.ee_ori_mat = np.array(s["ee_ori_mat"])
        self.ee_pos_vel = np.array(s["ee_pos_vel"])
        self.ee_ori_vel = np.array(s["ee_ori_vel"])
        self.joint_pos = np.array(s["q"])
        self.joint_vel = np.array(s["dq"])
        self.J_pos = np.array(s["J"][:3])
        self.J_ori = np.array(s["J"][3:])
        self.J_full = np.array(s["J"])
        self.mass_matrix = np.array(s["M"])
        self.new_update = False

    @property
    def torque_compensation(self):
        # libfranka gravity-compensates internally, so the hardware analogue of
        # mujoco's qfrc_bias is the Coriolis term alone.
        return np.array(self._state["coriolis"])


def our_pipeline(state, action_dpos_norm, action_drot_norm, a_kp, a_kd, goal_ori_held):
    """BimanualFranka._osc_goal_delta -> server run_controller, for one policy step."""
    kp, kd = resolve_gains(a_kp, a_kd)
    # BimanualFranka receives deltas already in output units (the teleop/policy
    # scales them), where robosuite scales normalized ones inside set_goal.
    dpos = np.array(action_dpos_norm) * DELTA_POS_MAX
    drot = np.array(action_drot_norm) * DELTA_ROT_MAX
    dpos, drot = clip_delta(dpos, drot)

    ee_r = Rotation.from_matrix(state["ee_ori_mat"])
    if goal_ori_held is None or np.any(drot != 0.0):
        goal_r = Rotation.from_rotvec(drot) * ee_r
    else:
        goal_r = goal_ori_held
    goal_p = np.array(state["ee_pos"]) + dpos

    ctrl = OSCTorqueController(num_joints=7)
    ctrl.set_goal(goal_p, goal_r.as_matrix(), kp, kd, np.array(state["q"]))
    tau = ctrl.run_controller(
        state["ee_pos"], state["ee_ori_mat"], state["ee_pos_vel"], state["ee_ori_vel"],
        state["J"], state["q"], state["dq"], state["M"], state["coriolis"],
    )
    return tau, goal_r


def synthetic_state(rng):
    q = rng.normal(0, 0.8, 7)
    dq = rng.normal(0, 0.3, 7)
    A = rng.normal(0, 0.4, (7, 7))
    J = rng.normal(0, 0.6, (6, 7))
    return {
        "q": q, "dq": dq, "M": A @ A.T + np.eye(7) * 1.5, "J": J,
        "coriolis": rng.normal(0, 0.2, 7),
        "ee_pos": rng.normal(0, 0.4, 3),
        "ee_ori_mat": Rotation.random(random_state=int(rng.integers(1e6))).as_matrix(),
        "ee_pos_vel": (J @ dq)[:3], "ee_ori_vel": (J @ dq)[3:],
    }


def live_state():
    import rpyc
    from lerobot_robot_bimanual_franka.franka_jacobian import zero_jacobian
    conn = rpyc.connect("192.168.3.10", 18812, config={"sync_request_timeout": 30, "allow_pickle": True})
    conn.root.init_robot("192.168.201.10", True)
    q, dq, _, ee_pos, ee_quat, _, _ = conn.root.read_state("192.168.201.10")
    q, dq, ee_pos = np.array(q), np.array(dq), np.array(ee_pos)
    c = rpyc.classic.connect("192.168.3.10", 18812)
    c.execute("""
import __main__ as _m
_s = _m.FrankaTorqueService._sessions['192.168.201.10']
_st = _s.robot.read_once()
_o = (tuple(float(x) for x in _s.model.mass(_st)), tuple(float(x) for x in _s.model.coriolis(_st)),
      tuple(float(x) for x in _st.O_T_EE))
""")
    M = np.array(c.namespace["_o"][0]).reshape(7, 7, order="F")
    C = np.array(c.namespace["_o"][1])
    R = np.array(c.namespace["_o"][2]).reshape(4, 4, order="F")[:3, :3]
    c.close(); conn.close()
    J = zero_jacobian(q, ee_pos_base=ee_pos)
    return {"q": q, "dq": dq, "M": M, "J": J, "coriolis": C, "ee_pos": ee_pos,
            "ee_ori_mat": R, "ee_pos_vel": (J @ dq)[:3], "ee_ori_vel": (J @ dq)[3:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use one state read off the arm")
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    fails = []
    print(f"{'action (dpos_norm | drot_norm | kp,kd)':<46}{'max|tau_ours - tau_robosuite|':>32}")
    print("-" * 78)

    for t in range(args.trials):
        state = live_state() if args.live else synthetic_state(rng)
        dpos_n = np.zeros(3) if t % 4 == 3 else rng.uniform(-1, 1, 3)
        drot_n = np.zeros(3) if t % 2 == 0 else rng.uniform(-1, 1, 3)
        a_kp, a_kd = rng.uniform(-1, 1), rng.uniform(-1, 1)

        rs = RealStateOSC(
            state, input_max=1, input_min=-1,
            output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
            output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
            kp=150, damping_ratio=1, impedance_mode="variable",
            kp_limits=(0, 1500), damping_ratio_limits=(0, 10),
            control_ori=True, control_delta=True, uncouple_pos_ori=True,
        )
        # The libero wrapper's exponential remap is what produces robosuite's raw gains.
        kp_raw = np.full(6, 150.0 * 10.0 ** a_kp)
        damp_raw = np.full(6, 1.0 * 10.0 ** a_kd)
        rs.set_goal(np.concatenate([damp_raw, kp_raw, dpos_n, drot_n]))
        tau_rs = rs.run_controller()

        tau_ours, _ = our_pipeline(state, dpos_n, drot_n, a_kp, a_kd, None)
        err = float(np.max(np.abs(tau_ours - tau_rs)))
        scale = max(float(np.max(np.abs(tau_rs))), 1.0)
        ok = err / scale < 1e-6
        if not ok:
            fails.append(t)
        label = f"dp={np.round(dpos_n,2)} dr={np.round(drot_n,2)} kp={a_kp:+.2f}"
        print(f"{label:<46}{err:>20.3e}  {'OK' if ok else 'MISMATCH'}")

    print("\n" + ("END-TO-END EQUIVALENT" if not fails else f"MISMATCH on trials {fails}"))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
