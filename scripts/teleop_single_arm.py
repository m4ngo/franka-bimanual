#!/usr/bin/env python3
"""Single-arm teleop across all four leader/mode combinations.

    spacemouse_delta   SpaceMouse -> EE_DELTA   per-step delta, the policy's own action
    spacemouse_ee      SpaceMouse -> EE_POS     integrated absolute target pose
    gello_ee           GELLO      -> EE_POS     absolute pose via FR3 forward kinematics
    gello              GELLO      -> JOINT_POS  joint setpoints

Why this exists instead of another `lerobot-teleoperate` wrapper: the two
absolute modes need the leader's target seeded from the arm's real EE pose
between `connect()` and the first `get_action()`, and `lerobot-teleoperate` has
no hook there. Unseeded, a SpaceMouse in EE_POS commands `teleop.initial_pos` on
step one -- an absolute pose the arm jumps to. The loop itself is LeRobot's,
imported rather than reimplemented.

Which physical arm the rig drives and which leader the operator holds are
SEPARATE settings, both on the rig profile in config/rig.yaml (arms /
teleop_device); ports and hidraw paths are in config/teleop.yaml. The profile
used when --rig is omitted is rig.yaml's `default_single_arm_profile`, which is
also what scripts/single_arm_teleop.sh passes through -- neither side restates it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import franka_config as fc  # noqa: E402
import numpy as np  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.scripts.lerobot_teleoperate import teleop_loop  # noqa: E402
from lerobot.teleoperators import make_teleoperator_from_config  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

from lerobot_robot_bimanual_franka import (  # noqa: E402
    ControlMode, SingleArmFrankaConfig, SingleArmRightConfig,
)
from lerobot_teleoperator_gello import GelloConfig, GelloEEConfig  # noqa: E402
from lerobot_teleoperator_spacemouse import SpaceMouseConfig  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_RIG = fc.default_single_arm_profile()
# The exposed key prefix, which is NOT the physical arm -- see config/rig.yaml.
# Every rig in _RIGS must expose it; main() rejects one that does not.
_ARM_KEY = next(iter(fc.profile(_DEFAULT_RIG).arms))

# Rig profile -> config class. Which PHYSICAL arm each drives is in config/rig.yaml,
# never here. Both expose the same `r_` key prefix, which is what lets the leader
# wiring below be shared; main() rejects a rig that does not.
_RIGS = {
    "single_arm_franka": SingleArmFrankaConfig,
    "single_arm_right": SingleArmRightConfig,
}

def _spacemouse(device: str, teleop_id: str, use_delta: bool, **_) -> SpaceMouseConfig:
    return SpaceMouseConfig(
        id=teleop_id,
        hidraw_path=fc.teleop(f"spacemouse.devices.{device}.hidraw_path"),
        prefix=f"{_ARM_KEY}_",
        use_delta=use_delta,
    )


def _gello(cls, device: str, teleop_id: str, use_noise: bool = False):
    return cls(
        id=teleop_id,
        side=_ARM_KEY,
        port=fc.teleop(f"gello.devices.{device}.port"),
        use_noise=use_noise,
    )


# mode -> (robot control mode, leader factory). The leader's UNITS and the
# robot's action schema are chosen together here and nowhere else, which is the
# point of the table: they are not independent, and every earlier script let them
# be set separately. Pairing spacemouse(use_delta=True) with EE_POS, which the
# record script's two independent flags allowed, sends a 5 cm per-tick delta as an
# absolute goal pose.
MODES = {
    "spacemouse_delta": (ControlMode.EE_DELTA,
                         lambda **kw: _spacemouse(use_delta=True, **kw)),
    "spacemouse_ee":    (ControlMode.EE_POS,
                         lambda **kw: _spacemouse(use_delta=False, **kw)),
    "gello_ee":         (ControlMode.EE_POS,
                         lambda **kw: _gello(GelloEEConfig, **kw)),
    "gello":            (ControlMode.JOINT_POS,
                         lambda **kw: _gello(GelloConfig, **kw)),
}


def build_leader(mode: str, device: str, teleop_id: str, use_noise: bool = False):
    """Leader config for one mode. Shared with the recording scripts so the
    mode -> (control mode, units) pairing has exactly one definition."""
    return MODES[mode][1](device=device, teleop_id=teleop_id, use_noise=use_noise)


def _measured_ee(robot):
    """The arm's EE pose, from the snapshot get_observation just took.

    LeRobot's teleop_loop calls robot.get_observation() immediately before
    teleop.get_action(), so robot.kin is populated and this costs no round-trip.
    Returns None rather than reading over RPyC if it is not -- a missed tick of
    anti-windup is cheap, a blocking read on the teleop path is not.
    """
    kin = robot.kin
    if not kin or _ARM_KEY not in kin:
        return None
    _, _, _, pos, quat_xyzw, _ = kin[_ARM_KEY]
    return pos, quat_xyzw


def seed_leader(teleop, robot) -> None:
    """Point an integrating leader at the arm's real EE pose, and keep it there.

    A no-op for leaders that do not integrate: GELLO derives an absolute pose from
    the operator's own joint angles every tick, so it has neither state to seed nor
    an integrator to wind up. Only the SpaceMouse carries a target forward.

    Seeding fixes the first step; binding the pose source fixes every step after
    it. Without the bind, a held stick advances the EE_POS target 0.05 m per tick
    whether or not the arm follows -- see SpaceMouse.bind_pose_source.
    """
    seed = getattr(teleop, "seed_state", None)
    if not callable(seed):
        return
    kin = robot.robot_manager.current_kinematic_state_batch(list(robot.active_arms))
    _, _, _, pos, quat_xyzw, _ = kin[_ARM_KEY]
    seed(pos, quat_xyzw)
    teleop.bind_pose_source(lambda: _measured_ee(robot))
    logger.info("seeded %s from the arm's EE at %s (lead capped at %.3f m / %.2f rad)",
                type(teleop).__name__, np.round(pos, 4),
                teleop.config.max_lead_m, teleop.config.max_lead_rad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=sorted(MODES), help="leader and control mode")
    parser.add_argument("--rig", choices=sorted(_RIGS), default=_DEFAULT_RIG,
                        help="rig profile; which physical arm it drives is in config/rig.yaml")
    parser.add_argument("--teleop-device", default=None,
                        help="left|right; defaults to the rig profile's teleop_device")
    parser.add_argument("--fps", type=int, default=fc.control_fps())
    parser.add_argument("--duration", type=float, default=None, help="seconds; default unlimited")
    parser.add_argument("--display-data", action="store_true")
    args = parser.parse_args()

    init_logging()
    rig = args.rig
    arm_key = next(iter(fc.profile(rig).arms))
    if arm_key != _ARM_KEY:
        parser.error(f"rig {rig!r} exposes key {arm_key!r}, not {_ARM_KEY!r}; the leader "
                     "prefix and gripper wiring assume the latter.")
    device = args.teleop_device or getattr(fc.profile(rig), "teleop_device", None)
    if device is None:
        parser.error(
            f"rig profile {rig!r} has no teleop_device; set one in config/rig.yaml "
            "or pass --teleop-device. Both leaders enumerate identically, so the wrong "
            "one connects cleanly and then does nothing."
        )

    control_mode = MODES[args.mode][0]
    logger.info("%s: %s-hand leader driving %s in %s",
                args.mode, device, rig, control_mode.value)

    robot = make_robot_from_config(_RIGS[rig](control_mode=control_mode))
    teleop = make_teleoperator_from_config(
        build_leader(args.mode, device, f"{args.mode}_{_ARM_KEY}_teleop")
    )
    processors = make_default_processors()

    teleop.connect()
    robot.connect()
    try:
        seed_leader(teleop, robot)
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=args.fps,
            display_data=args.display_data,
            duration=args.duration,
            teleop_action_processor=processors[0],
            robot_action_processor=processors[1],
            robot_observation_processor=processors[2],
        )
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
