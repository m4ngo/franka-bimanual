# config/

The single source of truth for every environment constant in this workspace:
where the world origin is, where each robot base sits, camera intrinsics and
extrinsics, control gains, safety limits, and the control rate.

Nothing in this repo (outside `multi-fast/`, which is a reference submodule)
should hardcode any of these values. If you find a duplicate, delete it and
read from here instead.

## Files

| File | Contents |
|---|---|
| `world.yaml` | World frame definition, calibration-frame origin, per-arm `robot_base_in_world`, worktable height (world frame), sim alignment |
| `arms.yaml` | Per-arm NUC hosts, robot/gripper IPs, RPyC ports, gripper kind; home-pose directory |
| `cameras.yaml` | `cam_1` … `cam_6`: type, name, IP, serial, role/mount, resolution, intrinsics + extrinsics |
| `control.yaml` | Every control constant: rates, velocity-domain PD gains, the whole `torque:` block (OSC schedule, delta envelope, joint impedance, torque/speed limits, friction, RT loop), fudge factors, worktable brake, Franka wire dynamics, homing budget, WSG params |
| `rig.yaml` | Robot profiles — which arms and cameras each robot config exposes, under which key prefixes, and which leader device drives it |
| `teleop.yaml` | GELLO and SpaceMouse device ports, scales, sign flips, calibration positions |
| `policy.yaml` | Residual/policy normalization contract shared by sysid, residual_wrapper and trained checkpoints |
| `calibration.yaml` | ChArUco board spec and IO layout for `frames/camera/camera_calibration.py` |

## Reading it

Python — via the `franka_config` package (installed editable, see
`scripts/local_module_check.sh`):

```python
import franka_config as fc

fc.control_fps()                     # 20
fc.arm("right").robot_ip             # 192.168.201.10
fc.camera("cam_2").calibration.cam_in_world   # Pose, camera -> world
fc.robot_base_in_world("left")       # Pose, robot base -> world
fc.control("torque.osc.default_kp")  # 150.0
fc.home_q(key="r")                   # from home_poses/<default>.json
```

Shell — via the same loader, so the two can't drift:

```bash
source scripts/_config.sh
FPS=$(cfg control.rates.control_fps)
eval "$(cfg_arm right)"        # R_SERVER_IP, R_ROBOT_IP, R_PORT, R_SSH, …
eval "$(cfg_rig bimanual_franka)"
```

`FRANKA_CONFIG_DIR` overrides which directory is read, for running against an
alternate rig without editing tracked files.

## Conventions

- **Quaternions are wxyz (scalar first) everywhere in this directory.** scipy is
  scalar-last; `franka_config.quat_wxyz_to_xyzw` is the only place to convert.
  Mixing the two silently produced a 180-degree world rotation in the past.
- **Poses are named `<child>_in_<parent>`** and mean
  `p_parent = R @ p_child + t`. `robot_base_in_world` maps base coordinates
  into world — it is *not* inverted by consumers.
- **World is floor-origin.** z = 0 is the floor, the table top is 0.904, and the
  arm bases are at 0.912. Camera extrinsics are stored in the calibration frame
  — origin at the ChArUco board **centre**, on the table — and lifted by
  `world.calib_origin_in_world`. That single transform is the *only* place the
  board-to-world offset is expressed; the calibration script has no origin knob,
  so moving the world origin never invalidates a calibration.
- **The worktable brake is world-frame.** `world.worktable.height_m` is a single
  plane; `ActionSafetyScreen` lifts each arm's EE through that arm's
  `robot_base_in_world` before comparing, so no per-arm threshold exists.
- **The EE is a sphere, not a point.** `control.yaml`'s
  `worktable_brake.ee_sphere` gives a TOOL-frame centre (rotates with the
  gripper) and a radius; clearance is measured from the sphere's lowest point.
  An arm with a different gripper can override it with its own `ee_sphere`
  under `arms.yaml`. The default radius is a conservative 0.10 m — measure the
  real jaw extent and shrink it to recover workspace near the table.
- **The leader device is not derived from the arm.** `rig.yaml`'s
  `teleop_device` names which physical GELLO/SpaceMouse the operator holds; it
  is independent of which follower arm the profile drives. Both SpaceMice
  enumerate as identical HID devices, so an inferred device connects
  successfully and then silently does nothing.
- **`config/` holds data only.** The loader lives in `franka_config/`.
- **A constant defined twice is a bug, even when the values agree.** If a module
  reads a value from here and then assigns a literal to the same name, the
  literal wins and this file is decoration. Delete the literal.
- **`torque:` is read on both machines.** The NUC has no `franka_config`, so
  `scripts/deploy_nuc_server.sh` resolves that block here and ships it as
  `nuc_control_config.py`; `torque_config.py` is what both sides read it
  through. Derived quantities (the action→gain exponential bases) are computed
  from `torque.osc.*`, never configured separately.

## After editing

Values are cached per process, so restart anything running. To check what the
stack will actually see:

```bash
python -m franka_config dump control
python -m franka_config dump control.torque   # dotted paths work too
python -m franka_config cameras
```

**Editing `torque:` is not enough on its own** — re-run
`scripts/deploy_nuc_server.sh <mario|luigi>` or the arm keeps running the gains
from the last deploy.

Recalibrating a camera: run `python frames/camera/camera_calibration.py --cam cam_2`;
it prints a ready-to-paste `cameras.yaml` calibration block in the calibration
frame (board-centre origin — no world offset baked in, in any axis).
