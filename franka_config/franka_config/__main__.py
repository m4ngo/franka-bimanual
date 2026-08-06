"""Command-line access to config/*.yaml, so shell scripts read the same values.

    python -m franka_config get control.rates.control_fps
    python -m franka_config arm right            # shell exports for one arm
    python -m franka_config rig bimanual_franka  # shell exports for a profile
    python -m franka_config cameras --field ip   # one value per line
    python -m franka_config dump control         # whole section as JSON
"""

from __future__ import annotations

import argparse
import json
import sys

from . import _loader, schema


def _emit(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return " ".join(_emit(v) for v in value)
    if value is None:
        return ""
    return str(value)


def _cmd_get(args) -> int:
    print(_emit(_loader.get(args.path)))
    return 0


def _cmd_dump(args) -> int:
    # get() takes a dotted path and returns the whole file for a bare section
    # name, so `dump control` and `dump control.torque` both work.
    print(json.dumps(_loader.get(args.section), indent=2))
    return 0


def _cmd_arm(args) -> int:
    spec = schema.arm(args.name)
    prefix = args.prefix or spec.default_key.upper()
    rows = {
        f"{prefix}_SERVER_IP": spec.server_ip,
        f"{prefix}_ROBOT_IP": spec.robot_ip,
        f"{prefix}_GRIPPER_IP": spec.gripper.ip,
        f"{prefix}_GRIPPER_KIND": spec.gripper.kind,
        f"{prefix}_PORT": spec.rpyc_port,
        f"{prefix}_GRIPPER_PORT": spec.gripper_rpyc_port,
        f"{prefix}_NUC_HOST": spec.nuc_host,
        f"{prefix}_SSH": spec.ssh_target,
    }
    for key, value in rows.items():
        print(f"{key}={_emit(value)}")
    return 0


def _cmd_rig(args) -> int:
    profile = schema.profile(args.name)
    for key, arm_name in profile.arms.items():
        spec = schema.arm(arm_name)
        up = key.upper()
        print(f"{up}_ARM={arm_name}")
        print(f"{up}_SERVER_IP={spec.server_ip}")
        print(f"{up}_ROBOT_IP={spec.robot_ip}")
        print(f"{up}_GRIPPER_IP={spec.gripper.ip}")
        print(f"{up}_PORT={spec.rpyc_port}")
    print(f"CONTROL_MODE={profile.control_mode}")
    print(f"CONTROL_FPS={schema.control_fps()}")
    print(f"CAMERAS={' '.join(profile.cameras)}")
    return 0


def _cmd_cameras(args) -> int:
    for key in schema.camera_keys():
        spec = schema.camera(key)
        if args.field:
            print(_emit(getattr(spec, args.field)))
        else:
            print(f"{key} {spec.type} {spec.name} {spec.ip} {spec.role} {spec.mount}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="franka_config", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("get", help="print one dotted config value")
    p.add_argument("path", help="e.g. control.rates.control_fps")
    p.set_defaults(func=_cmd_get)

    p = sub.add_parser("dump", help="print a whole section as JSON")
    p.add_argument("section")
    p.set_defaults(func=_cmd_dump)

    p = sub.add_parser("arm", help="shell exports for one physical arm")
    p.add_argument("name", help="left | right")
    p.add_argument("--prefix", default=None, help="variable prefix (default: the arm's key)")
    p.set_defaults(func=_cmd_arm)

    p = sub.add_parser("rig", help="shell exports for a rig profile")
    p.add_argument("name")
    p.set_defaults(func=_cmd_rig)

    p = sub.add_parser("cameras", help="list cameras")
    p.add_argument("--field", default=None, help="print only this CameraSpec field, one per line")
    p.set_defaults(func=_cmd_cameras)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"franka_config: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
