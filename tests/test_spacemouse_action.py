"""SpaceMouse must emit the same EE_DELTA action a sim base policy emits.

The controller tests prove the stack matches robosuite given an action. These
prove the teleop *produces* the right action: full stick deflection is a
normalized +/-1 policy action, both channels share one device->base rotation,
and the resulting goal pose matches what robosuite's set_goal builds from the
equivalent normalized action.

    python tests/test_spacemouse_action.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the controller suite's robosuite loader, pylibfranka stub and fixtures.
from test_osc_stack import (  # noqa: E402
    DELTA_POS_MAX, DELTA_ROT_MAX, ControlMode, RefOSC, make_case, make_robot,
    make_session, our_torque, ref_torque, _rel,
)

from lerobot_teleoperator_spacemouse import SpaceMouse, SpaceMouseConfig  # noqa: E402
from lerobot_teleoperator_spacemouse.lerobot_teleoperator_spacemouse.spacemouse import (  # noqa: E402
    ANGULAR_DEVICE_TO_BASE, LINEAR_DEVICE_TO_BASE, _apply_deadzone,
)


class _FakeState:
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=(0, 0), t=0.0):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.buttons, self.t = list(buttons), t


class _FakeDevice:
    """Returns one fixed state; t never advances so the drain loop exits at once."""

    def __init__(self, state):
        self.state = state

    def read(self):
        return self.state

    def close(self):
        pass


def make_teleop(state, **cfg_kw):
    cfg = SpaceMouseConfig(prefix="r_", use_delta=True, **cfg_kw)
    tel = SpaceMouse(cfg)
    tel._device = _FakeDevice(state)
    return tel


def action_vec(act):
    dp = np.array([act["r_x"], act["r_y"], act["r_z"]])
    dq = np.array([act["r_qx"], act["r_qy"], act["r_qz"], act["r_qw"]])
    return dp, Rotation.from_quat(dq).as_rotvec()


def test_full_deflection_equals_one_normalized_policy_action():
    """Stick at the stop == robosuite output_max == clip_delta's bound."""
    tel = make_teleop(_FakeState(y=1.0), deadzone=0.0)
    dp, _ = action_vec(tel.get_action())
    assert np.isclose(dp[0], DELTA_POS_MAX), dp
    tel = make_teleop(_FakeState(roll=1.0), deadzone=0.0)
    _, dr = action_vec(tel.get_action())
    assert np.isclose(np.linalg.norm(dr), DELTA_ROT_MAX), dr
    assert np.isclose(dr[0], DELTA_ROT_MAX), dr


def test_each_channel_uses_its_own_device_to_base_rotation():
    """Each channel maps through its declared matrix.

    The two are NOT the same: pyspacemouse's angular triple is not reported on
    the same axes as its linear triple on this device, so deriving the angular
    map from the linear one swaps roll and pitch at the robot. Confirmed on
    hardware -- see the note on the constants.
    """
    cfg = SpaceMouseConfig(prefix="r_", use_delta=True)
    t_signs = np.asarray(cfg.translation_signs, dtype=np.float64)
    r_signs = np.asarray(cfg.rotation_signs, dtype=np.float64)

    for i, axis in enumerate("xyz"):
        lin = make_teleop(_FakeState(**{axis: 1.0}), deadzone=0.0)
        dp, _ = action_vec(lin.get_action())
        expect = (LINEAR_DEVICE_TO_BASE @ np.eye(3)[i]) * t_signs
        assert np.allclose(dp / np.linalg.norm(dp), expect, atol=1e-9), f"linear {axis}"

    for i, axis in enumerate(("roll", "pitch", "yaw")):
        ang = make_teleop(_FakeState(**{axis: 1.0}), deadzone=0.0)
        _, dr = action_vec(ang.get_action())
        expect = (ANGULAR_DEVICE_TO_BASE @ np.eye(3)[i]) * r_signs
        assert np.allclose(dr / np.linalg.norm(dr), expect, atol=1e-9), f"angular {axis}"


def test_device_maps_are_proper_rotations():
    """A reflection would mirror a channel; the import-time guard must hold."""
    for name, m in (("linear", LINEAR_DEVICE_TO_BASE), ("angular", ANGULAR_DEVICE_TO_BASE)):
        assert np.isclose(np.linalg.det(m), 1.0), name
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12), name


def test_roll_and_pitch_are_not_swapped():
    """Regression: deriving the angular map from the linear one put roll on the
    robot's Y axis and pitch on X, which is what the operator felt as swapped."""
    _, roll_dr = action_vec(make_teleop(_FakeState(roll=1.0), deadzone=0.0).get_action())
    _, pitch_dr = action_vec(make_teleop(_FakeState(pitch=1.0), deadzone=0.0).get_action())
    assert int(np.argmax(np.abs(roll_dr))) == 0, f"roll must drive base X, got {roll_dr}"
    assert int(np.argmax(np.abs(pitch_dr))) == 1, f"pitch must drive base Y, got {pitch_dr}"


def test_deadzone_kills_crosstalk_but_keeps_full_range():
    """The measured yaw->linear cross-talk (~0.23) must be suppressed."""
    dz = SpaceMouseConfig().deadzone
    assert _apply_deadzone(np.array([0.5 * dz]), dz)[0] == 0.0
    assert np.isclose(_apply_deadzone(np.array([1.0]), dz)[0], 1.0)
    assert np.isclose(_apply_deadzone(np.array([-1.0]), dz)[0], -1.0)
    # Monotone and sign-preserving.
    v = np.linspace(-1, 1, 101)
    out = _apply_deadzone(v, dz)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all(np.sign(out) * np.sign(v) >= 0)


def test_neutral_stick_emits_exactly_zero():
    """osc.py holds goal_ori only on an EXACTLY zero rotation delta, so a
    centred stick must produce exact zeros, not float dust."""
    act = make_teleop(_FakeState()).get_action()
    dp, dr = action_vec(act)
    assert np.all(dp == 0.0), dp
    assert np.all(dr == 0.0), dr


def test_spacemouse_action_drives_the_same_torque_as_the_sim_policy():
    """End to end at the DEFAULT config: whatever the device emits, our torque
    equals robosuite's for the same action.

    The reference action is recovered from the emitted action itself (divide by
    output_max, the exact inverse of the scale), not recomputed from the raw
    device axes -- so this tests the controller path for whatever the teleop
    produces, deadzone and all, rather than assuming the mapping.
    test_both_channels_share_one_device_to_base_rotation pins the mapping.
    """
    rng = np.random.default_rng(31)
    worst = 0.0
    for i in range(30):
        case = make_case(rng)
        dev = rng.uniform(-1, 1, 6)
        st = _FakeState(x=dev[0], y=dev[1], z=dev[2], roll=dev[3], pitch=dev[4], yaw=dev[5])

        act = make_teleop(st).get_action()          # default config: deadzone on
        dp, dr = action_vec(act)
        # No clipping can have occurred: |normalized| <= 1 => |dp| <= output_max.
        assert np.all(np.abs(dp) <= DELTA_POS_MAX + 1e-12)
        assert np.all(np.abs(dr) <= DELTA_ROT_MAX + 1e-12)

        ours = our_torque(make_robot(case), make_session(case), act, case)
        ref = ref_torque(case, dp / DELTA_POS_MAX, dr / DELTA_ROT_MAX,
                         act["kp"], act["kd"])
        worst = max(worst, _rel(ours, ref))
        assert _rel(ours, ref) < 1e-5, f"trial {i}: rel err {_rel(ours, ref):.2e}"
    return f"worst relative error {worst:.2e} over 30 deflections, default config"


def test_spacemouse_sequence_matches_robosuite_including_orientation_hold():
    """A multi-step teleop session, including ticks where the stick is centred.

    This is where the two implementations could silently diverge: osc.py rewrites
    goal_ori only on an exactly-zero rotation delta, so a released stick must
    keep the held orientation on BOTH sides across the whole sequence.
    """
    rng = np.random.default_rng(32)
    case = make_case(rng)
    robot = make_robot(case)
    session = make_session(case)
    ref = RefOSC(case, input_max=1, input_min=-1,
                 output_max=(DELTA_POS_MAX,) * 3 + (DELTA_ROT_MAX,) * 3,
                 output_min=(-DELTA_POS_MAX,) * 3 + (-DELTA_ROT_MAX,) * 3,
                 kp=150, damping_ratio=1, impedance_mode="variable",
                 kp_limits=(0, 1500), damping_ratio_limits=(0, 10))
    kp_raw, damp_raw = np.full(6, 150.0), np.full(6, 1.0)

    # push / twist / release / push again / release
    seq = [(0.0, 0.6, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.7),
           (0.0,) * 6, (0.4, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0,) * 6,
           (0.0, 0.0, 0.5, 0.3, 0.0, 0.0), (0.0,) * 6]
    worst = 0.0
    for step, dev in enumerate(seq):
        st = _FakeState(x=dev[0], y=dev[1], z=dev[2], roll=dev[3], pitch=dev[4], yaw=dev[5])
        act = make_teleop(st).get_action()
        dp, dr = action_vec(act)

        ours = our_torque(robot, session, act, case)
        ref.set_goal(np.concatenate([damp_raw, kp_raw, dp / DELTA_POS_MAX, dr / DELTA_ROT_MAX]))
        theirs = ref.run_controller()
        worst = max(worst, _rel(ours, theirs))
        assert _rel(ours, theirs) < 1e-5, f"step {step} dev={dev}: {_rel(ours, theirs):.2e}"

        # And the held goal orientations must still agree.
        ours_R = Rotation.from_quat(robot.robot_manager.osc_goals[-1]["r"][1]).as_matrix()
        assert np.max(np.abs(ours_R - ref.goal_ori)) < 1e-5, f"goal_ori diverged at step {step}"
    return f"worst relative error {worst:.2e} over a 7-step session"


def test_spacemouse_absolute_mode_matches_robosuite():
    """use_delta=False feeds EE_POS; robosuite's control_delta=False equivalent."""
    rng = np.random.default_rng(33)
    cfg = SpaceMouseConfig(prefix="r_", use_delta=False, deadzone=0.0)
    worst = 0.0
    for i in range(15):
        case = make_case(rng)
        tel = SpaceMouse(cfg)
        tel._device = _FakeDevice(_FakeState(y=0.4, yaw=0.3))
        tel.seed_state(case["ee_pos"], Rotation.from_matrix(case["R"]).as_quat())
        act = tel.get_action()

        robot = make_robot(case, mode=ControlMode.EE_POS)
        ours = our_torque(robot, make_session(case), act, case)

        target_p = np.array([act["r_x"], act["r_y"], act["r_z"]])
        target_r = Rotation.from_quat([act["r_qx"], act["r_qy"], act["r_qz"], act["r_qw"]])
        ref = ref_torque(case, None, None, act["kp"], act["kd"],
                         absolute=np.concatenate([target_p, target_r.as_rotvec()]))
        worst = max(worst, _rel(ours, ref))
        assert _rel(ours, ref) < 1e-5, f"trial {i}: rel err {_rel(ours, ref):.2e}"
    return f"worst relative error {worst:.2e} over 15 poses"


DEVICE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def test_each_device_axis_drives_the_same_torque_as_robosuite():
    """Push one device axis at a time, both signs, several deflections, and
    compare our torque against robosuite's for the action produced."""
    rng = np.random.default_rng(51)
    per = {}
    for i, axis in enumerate(DEVICE_AXES):
        worst = 0.0
        for sign in (+1.0, -1.0):
            for mag in (0.25, 0.6, 1.0):
                for _ in range(3):
                    case = make_case(rng)
                    st = _FakeState(**{axis: sign * mag})
                    act = make_teleop(st).get_action()      # default config
                    dp, dr = action_vec(act)
                    ours = our_torque(make_robot(case), make_session(case), act, case)
                    ref = ref_torque(case, dp / DELTA_POS_MAX, dr / DELTA_ROT_MAX,
                                     act["kp"], act["kd"])
                    err = _rel(ours, ref)
                    assert err < 1e-5, f"{axis} sign={sign:+.0f} mag={mag}: {err:.2e}"
                    worst = max(worst, err)
        per[axis] = worst
    return " ".join(f"{a} {per[a]:.1e}" for a in DEVICE_AXES)


def test_each_device_axis_lands_on_the_expected_base_axis():
    """Which robot axis each device axis drives, with sign.

    The sign trims are part of the convention: this puck reports yaw inverted.
    """
    cfg = SpaceMouseConfig(prefix="r_", use_delta=True)
    expected = {}
    for i, axis in enumerate(DEVICE_AXES):
        m = LINEAR_DEVICE_TO_BASE if i < 3 else ANGULAR_DEVICE_TO_BASE
        signs = cfg.translation_signs if i < 3 else cfg.rotation_signs
        vec = (m @ np.eye(3)[i % 3]) * np.asarray(signs, dtype=np.float64)
        expected[axis] = (int(np.argmax(np.abs(vec))), float(np.sign(vec[np.argmax(np.abs(vec))])))

    rows = []
    for i, axis in enumerate(DEVICE_AXES):
        act = make_teleop(_FakeState(**{axis: 1.0})).get_action()
        dp, dr = action_vec(act)
        got = dp if i < 3 else dr
        k = int(np.argmax(np.abs(got)))
        sgn = float(np.sign(got[k]))
        want_k, want_sgn = expected[axis]
        assert k == want_k and sgn == want_sgn, (
            f"device {axis} -> base {'XYZ'[k]}{'+' if sgn > 0 else '-'}, "
            f"expected {'XYZ'[want_k]}{'+' if want_sgn > 0 else '-'}")
        # Nothing should leak onto the other two axes of that channel.
        leak = np.max(np.abs(np.delete(got, k))) / abs(got[k])
        assert leak < 1e-12, f"device {axis}: {leak:.2e} leak onto other base axes"
        kind = "lin" if i < 3 else "ang"
        rows.append(f"{axis}->{kind} {'XYZ'[k]}{'+' if sgn > 0 else '-'}")
    return " ".join(rows)


def test_gripper_buttons_latch_and_open_wins():
    assert make_teleop(_FakeState(buttons=(1, 0))).get_action()["r_gripper"] < 0
    assert make_teleop(_FakeState(buttons=(0, 1))).get_action()["r_gripper"] > 0
    assert make_teleop(_FakeState(buttons=(1, 1))).get_action()["r_gripper"] > 0
    assert make_teleop(_FakeState()).get_action()["r_gripper"] == 0.0


def test_absolute_mode_integrates_the_same_deltas():
    """use_delta=False must integrate in the base frame, matching how
    set_goal_orientation composes delta onto the current orientation."""
    cfg = SpaceMouseConfig(prefix="r_", use_delta=False, deadzone=0.0)
    tel = SpaceMouse(cfg)
    tel._device = _FakeDevice(_FakeState(y=0.5, yaw=0.4))
    tel.seed_state(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    pos = np.zeros(3)
    rot = Rotation.identity()
    t_signs = np.asarray(cfg.translation_signs, dtype=np.float64)
    r_signs = np.asarray(cfg.rotation_signs, dtype=np.float64)
    step_p = (LINEAR_DEVICE_TO_BASE @ np.array([0.0, 0.5, 0.0])) * t_signs * DELTA_POS_MAX
    step_r = Rotation.from_rotvec(
        (ANGULAR_DEVICE_TO_BASE @ np.array([0.0, 0.0, 0.4])) * r_signs * DELTA_ROT_MAX)
    for _ in range(4):
        act = tel.get_action()
        pos = pos + step_p
        rot = step_r * rot
        got_p = np.array([act["r_x"], act["r_y"], act["r_z"]])
        got_r = Rotation.from_quat([act["r_qx"], act["r_qy"], act["r_qz"], act["r_qw"]])
        assert np.allclose(got_p, pos, atol=1e-12), (got_p, pos)
        assert np.allclose(np.abs(np.dot(got_r.as_quat(), rot.as_quat())), 1.0, atol=1e-9)


def test_gains_channel_is_neutral():
    """kp/kd of 0.0 means the sim defaults (150, critically damped)."""
    act = make_teleop(_FakeState()).get_action()
    assert act["kp"] == 0.0 and act["kd"] == 0.0


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
