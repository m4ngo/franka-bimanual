# CLAUDE.md

Real-world control stack for a **bimanual Franka FR3** setup at TRI. The
workspace is a collection of LeRobot plugin packages plus shell scripts that
drive the standard `lerobot-teleoperate` / `lerobot-record` / `lerobot-replay`
/ `lerobot-train` CLIs against this hardware.

See `README.md` for hardware-specific bring-up procedures (FCI enablement,
control-box power, NUC server, calibration). This file is the codebase-level
orientation.

## Hardware topology

Three machines, two arms, two grippers, six cameras:

- **`franka@deepblue`** workstation (Tailscale). All scripts run here.
  - venv at `~/.venv` — activate before any script.
  - `~/franka_data/` holds rollouts, datasets, trained policies. **Never
    commit data into the repo.**
  - `~/franka_ws/` is this workspace.
- **`mario@192.168.3.10`** NUC controls the **right** arm
  (`192.168.201.10`, Schunk WSG at `192.168.2.20`, RPyC port `18812`).
- **`luigi@192.168.3.11`** NUC controls the **left** arm
  (`192.168.200.2`, native Franka Hand, RPyC port `18813`).
- Six GigE cameras (4× Basler ARV, 2× FRAMOS D415e).

These values are stated here for orientation only — `config/arms.yaml` and
`config/cameras.yaml` are authoritative, and `python -m franka_config arm right`
prints what the stack will actually use.

Each NUC runs `./run_server.sh`, which launches `pylibfranka_server.py` pinned
to **CPUs 0-1**. The server is only an RPyC → shared-memory proxy; it spawns one
`pylibfranka_control.py` child per arm under `chrt -f 80`, pinned to whichever
core takes the fewest device interrupts. That child owns the **1 kHz torque
control loop** (libfranka `readOnce`/`writeOnce`) and the OSC law; the
workstation pushes goals at policy rate over RPyC. Deploy with
`scripts/deploy_nuc_server.sh <mario|luigi>`. The franky/net_franky
`start_control.sh` is superseded — franky has no torque interface.

**Three placement rules keep the loop real-time**, all found the hard way by
watching `control_command_success_rate`. Breaking any one of them brings back
`communication_constraints_violation` aborts, which latch a stiff joint hold
mid-trajectory and read as a jerk:

- Nothing else may share the loop's core. The RPyC server and the gripper server
  are pinned to 0-1 for this reason — `SCHED_FIFO` wins the CPU but does not
  stop their socket work from landing NET_RX softirqs on the loop's core.
- The loop's core is chosen at spawn from `/proc/interrupts`
  (`torque.loop.rt_cpu_candidates`). Both the arm's UDP and the workstation's
  RPyC ride one NIC here whose MSI-X queues sit on most of 2-7; steering the
  IRQs away instead would need root.
- The loop writes before it computes. See `pylibfranka_control.py`'s docstring —
  the robot grades response latency, not tick duration.

## Central configuration — read this first

**`config/*.yaml` is the single source of truth for every environment constant**:
world origin, per-arm base poses, arm IPs/ports, camera intrinsics/extrinsics,
control gains, **every torque-loop constant**, safety limits, fudge factors, and
the control rate. Nothing outside `config/` should hardcode any of them. See
[config/README.md](config/README.md) for the file-by-file breakdown.

Read it through the `franka_config` package (never by parsing YAML yourself):

```python
import franka_config as fc
fc.control_fps()                    # 20 — one rate for teleop/record/rollout/sysid
fc.arm("right").robot_ip
fc.robot_base_in_world("left")      # Pose: p_world = R @ p_base + t
fc.camera("cam_2").calibration.cam_in_world
fc.control("torque.osc.default_kp")
fc.home_q(key="r")                  # from home_poses/<default>.json
```

Shell scripts use the same loader via `scripts/_config.sh` (`cfg`, `cfg_arm`,
`cfg_rig`, `cfg_cameras`) so the two can never drift.

Hard rules:

- **Quaternions in `config/` are wxyz (scalar first).** scipy is scalar-last;
  `fc.quat_wxyz_to_xyzw` is the only conversion point. Confusing the two once
  produced a silent 180° world rotation.
- **Poses are `<child>_in_<parent>`**, meaning `p_parent = R @ p_child + t`.
  `robot_base_in_world` maps base → world and is **never inverted** by consumers.
- **World is floor-origin**: z=0 floor, table top `world.worktable.height_m`
  (what the safety brake compares against), arm bases 0.912. Camera extrinsics
  are stored in the calibration frame — origin at the board **centre** — and
  lifted by `world.calib_origin_in_world`, which is the single place the
  board-to-world offset lives. Never bake an offset into an extrinsic.
- **Home configurations live only in `home_poses/*.json`.** No home `q` vector
  belongs in Python.
- Robot/teleop/camera dataclass fields get their defaults from YAML via
  `default_factory`, so every existing `--robot.*` / `--teleop.*` CLI override
  still works unchanged.
- **A constant defined twice is a bug, even when the values agree.** The
  osc→main merge left both sides' definitions in several modules and the second
  one silently won, so `config/control.yaml` was being read and then discarded.
  If you see a config read followed by a literal assignment to the same name,
  delete the literal.

### Getting config onto the NUC

The NUC env is numpy + pylibfranka only — `franka_config` is not installed
there, and a YAML parse plus a directory search has no business in the process
that owns the 1 kHz loop. So `scripts/deploy_nuc_server.sh` resolves
`control.yaml`'s `torque:` block on the workstation and writes it next to the
server as `nuc_control_config.py`.
[torque_config.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/torque_config.py)
is what both sides read it through: `franka_config` when importable, that
generated file otherwise, and a hard error if neither — never a baked-in
default, because a silently stale gain is the failure this indirection exists
to prevent.

**A gain change therefore only reaches the arm once you re-run the deploy
script.** The workstation copy is not what runs the loop.

## Package layout

The repo is **six editable packages** in a flat layout plus a `scripts/`
directory:

```
franka_ws/
├── config/                           # YAML: the environment's single source of truth
├── franka_config/                    # loader + typed accessors for config/ (import this)
├── lerobot_robot_bimanual_franka/    # follower: two FR3 arms + grippers + 6 cameras
├── lerobot_teleoperator_gello/       # leader: joint-mode and EE-mode GELLOs
├── lerobot_teleoperator_spacemouse/  # leader: 3Dconnexion SpaceMice (EE only)
├── lerobot_camera_arv/               # Aravis GigE cameras (Basler BFS)
├── lerobot_camera_framos/            # FRAMOS D415e via FRAMOS librealsense2
├── scripts/                          # bash wrappers around lerobot-* CLIs
├── tests/                            # full-stack equivalence tests vs robosuite
├── sysid/                            # gain/friction identification against sim references
├── residual_wrapper/                 # residual policy runner on top of the follower
├── home_poses/                       # named home configurations (JSON)
└── frames/                           # per-camera reference snapshots + calibration
```

Each package self-registers with LeRobot's config registries via
`@RobotConfig.register_subclass`, `@TeleoperatorConfig.register_subclass`, and
`@CameraConfig.register_subclass` decorators — that is how the
`--robot.type=bimanual_franka` / `--teleop.type=bimanual_gello` strings on the
CLI resolve to classes here.

Setup commands live in [scripts/local_module_check.sh](scripts/local_module_check.sh)
— it installs the six packages with `uv pip install --no-deps -e ... -C
editable_mode=compat` (`franka_config` first) plus the non-PyPI deps (FRAMOS-built
`pyrealsense2` from `~/librealsense2/wrappers/python/`, `net_franky`,
`PyGObject<3.52`, `pyspacemouse`, `dynamixel-sdk`).

## Robot stack — `lerobot_robot_bimanual_franka`

- [bimanual_franka.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/bimanual_franka.py)
  — the `Robot` subclass. Holds a `MultiRobotWrapper`, two grippers, six
  cameras, and an `ActionSafetyScreen`. `get_observation()` reads cameras in a
  thread pool while batch-querying both arms' kinematic state in parallel;
  it caches that snapshot so the immediate `send_action()` skips a redundant
  RPyC round-trip (past `observation.kin_cache_max_age_s` it re-reads instead —
  EE_DELTA anchors on the measured pose, so a stale anchor silently eats part
  of the commanded delta).
- [franka_process.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/franka_process.py)
  — the RPyC client. Ships *goals*, not per-tick commands: `send_osc_goal`,
  `send_joint_goal` and `send_joint_velocity` are non-blocking pushes, and
  `get_kinematic_state` is a non-blocking read of whatever the server last
  published. **Tuples, not lists** on the wire — brine encodes immutable values;
  lists cross as netrefs that cost a round-trip per element and spam
  `AttributeError` when numpy probes `__array__`.
- [pylibfranka_server.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/pylibfranka_server.py)
  — runs **on the NUC**, not here. The RPyC face of the stack: it owns no robot
  connection, only the shm channel and the control child's lifetime. The service
  **must** keep its `SlaveService` base or `FrankaGripper`'s
  `rpyc.classic.connect` breaks.
- [pylibfranka_control.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/pylibfranka_control.py)
  — runs **on the NUC**, one process per arm. **Read the docstring before
  editing.** Owns the `ActiveControlBase` and the RT loop, recomputing tau from
  the held goal. Constraints encoded there: libfranka matrices are column-major
  (`order="F"`); `Model.zero_jacobian` returns all zeros on this build (re-checked
  under pylibfranka, still zero) so the analytic `franka_jacobian` is used
  instead; torque rate limiting against `state.tau_J_d` is mandatory; the law runs
  at **500 Hz, not 1 kHz**, because that is robosuite's substep rate; the write
  precedes the compute; the speed guard's envelope must stay **outside** the one
  a ±1 sim action produces (`v = (kp/kd)·delta` = 0.31 m/s / 3.06 rad/s at the
  default kp, and 3.06 rad/s runs the wrist at ~0.9 of rated) or it silently
  rescales the control law; and recoverable errors (reflex,
  `communication_constraints_violation`, UDP timeout) re-arm torque control
  holding the pose the arm actually ended up in.
- [pylibfranka_shm.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/pylibfranka_shm.py)
  — the goal/state block the two share, with a seqlock so neither ever blocks the
  other. Only the creator may unlink it: Python's `resource_tracker` otherwise
  destroys the segment when *any* attached process exits, so a restarting control
  child would take the server's channel down with it.
- [osc_torque_controller.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/osc_torque_controller.py)
  — robosuite's `OperationalSpaceController` ported 1:1 in
  `impedance_mode="variable"`. Deployed to the NUC alongside the server, so it
  stays numpy-only with no *unguarded* package-relative imports — the one import
  it has (`torque_config`) uses the same relative/flat try-except the other NUC
  modules use. Verify changes with `scripts/check_osc_parity.py`, which diffs it
  against robosuite's real modules.
- [torque_config.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/torque_config.py)
  — resolves `control.yaml`'s `torque:` block on either side of the RPyC link.
  See "Getting config onto the NUC" above.
- [rig_config.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/rig_config.py)
  — the bridge from `config/rig.yaml` profiles to concrete camera configs and
  per-arm connection fields. Keeps `franka_config` free of LeRobot imports.
- [safety.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/safety.py)
  — `ActionSafetyScreen` applies pre-dispatch shaping to goal poses (torque
  paths) or velocity commands (the remaining velocity callers). Every threshold
  comes from `config/control.yaml` and `config/world.yaml`:
  - **Worktable brake**: caps downward motion by `sqrt(2·MAX_DECEL·clearance)`
    so the EE can stop before reaching `worktable.height_m + distance_min_m`.
    `shape_goal` / `shape_joint_goal` express that envelope as a bound on the
    downward position error (the impedance loop settles at `v = (kp/kd)·error`)
    plus a hard floor the goal can never cross; `shape_ee` / `shape_joint` keep
    the velocity-domain form.
  - All four reason in **world frame**: the screen is constructed with each arm's
    `robot_base_in_world`, so one table plane covers arms whose bases differ in
    height or yaw. The EE is a **sphere**, not a point
    (`worktable_brake.ee_sphere`, per-arm overridable in `arms.yaml`): its centre
    is a TOOL-frame offset that rotates with the gripper, clearance is measured
    from the sphere's bottom, and the centre's velocity carries the `ω × r` lever
    term — so a tilted or rotating gripper can't graze the table while the TCP
    still reads clear. `goal_z_floor` is a **world-frame plane compared against
    `sphere_bottom_world_z`**, not a base-frame goal-z bound. Corrections are
    applied along world-up only; joint mode scales the whole vector to preserve
    direction.
  - L2-norm clamps on joint velocity and EE linear/angular velocity
    (`limits.*`). These bound the *velocity-domain* callers only; the torque
    paths are bounded by the NUC's speed guard instead.
  - Bimanual arm-repel is **not yet implemented** (noted in the module
    docstring).
- [wsg.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/wsg.py)
  — Schunk WSG (GCL protocol) gripper driver. One socket, two threads
  (reader + sender) coordinated by a `Condition`. Designed so `move()` only
  hits the wire when the target actually changes (coalesced by
  `target_change_thresh_mm`, capped at `min_move_interval_s`) — this keeps
  motion plans from carpet-bombing the gripper, which is what makes it feel
  laggy. Blocking ops (`HOME`/`GRIP`/`RELEASE`) use a token-matched
  `_Waiter` queue. `bye()` is called on `close()` to avoid latching FAST STOP.
- [franka_gripper.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/franka_gripper.py)
  — the native Franka Hand, over its own RPyC port. Selected automatically when
  an arm's gripper IP equals its robot IP.

### Robot action schema

`BimanualFranka` has three modes selected by `control_mode`. All land on torque;
`send_action` is robosuite's `set_goal` half, the NUC runs `run_controller`.

- **`JOINT_POS`**: keys `l_joint_1…l_joint_7, l_gripper, r_joint_1…`. Dispatched
  as a joint-impedance goal at `torque.joint_impedance.kp`.
- **`EE_POS`**: keys `l_{x,y,z,qx,qy,qz,qw,gripper}` and `r_*`, an absolute pose
  used directly as the OSC goal.
- **`EE_DELTA`**: same keys as a delta. `goal = ee_pos + delta`, rebuilt from the
  *current* pose every policy step (never accumulated onto the previous goal),
  delta clipped to `torque.delta.*` (osc_pose.json's ±0.05 m / ±0.5 rad).
  Clip **then** fudge: the other order lets `clip_delta` eat the fudge.

The `kp`/`kd` action entries use the sim's exponential remap, matching
`multi-fast/utils/envs/libero.py`: `kp = 150·10^a_kp` clipped to [0, 1500],
`damping_ratio = 1·10^a_kd` clipped to [0, 10], `kd = 2√kp·ratio`. The bases are
**derived** from `torque.osc.{default_kp,kp_limits,…}`, never configured
separately — a second copy is how it drifts from the sim.

Gripper values are normalised to `[0, 1]` against `gripper.wsg.true_max_mm`
(110.0 mm).

Camera frames are `observation.image_width` × `observation.image_height` RGB
and exposed in the observation under `cam_1` … `cam_6` — those are the
canonical keys everywhere; `role`/`mount` in `config/cameras.yaml` are metadata
and never part of the key. Camera failures degrade to `blank_frame()` (last
known image) rather than raising.

**Rig profiles.** `config/rig.yaml` maps each robot config to its arms and
cameras. `single_arm_franka` exposes the **left** FR3 under the `r_` key prefix
(deliberate: it keeps previously recorded `r_*` datasets readable), so the key
prefix is not the physical arm — always resolve through the profile.

## Teleop stack

Three leaders, all emitting the **same `l_` / `r_` key prefixes** the follower
expects. The convention: every bimanual teleop is a thin shell over two
single-arm instances; per-arm calibrations live in `{id}_left.json` /
`{id}_right.json` under the `calibration_dir`.

- **Joint-mode GELLO** ([gello.py](lerobot_teleoperator_gello/lerobot_teleoperator_gello/gello.py))
  reads 8 Dynamixel motors per arm (7 joints + gripper) over `DynamixelMotorsBus`
  at 57.6 kbps. `_process_action` normalises raw counts to radians using the
  calibration offsets, then to `[0, 1]` for the gripper. An async reader
  thread keeps `latest_action` fresh with EMA smoothing (alpha defaults 0.99).
  Joint signs and `calibration_position` live in `GelloLeaderFields` —
  hardware-specific defaults from `config/teleop.yaml`.
- **EE-mode GELLO** ([gello_ee.py](lerobot_teleoperator_gello/lerobot_teleoperator_gello/gello_ee.py))
  subclasses `Gello` and runs the joint reading through
  [franka_fk.py](lerobot_teleoperator_gello/lerobot_teleoperator_gello/franka_fk.py)
  (modified-DH Craig convention) to emit absolute EE poses. Output keys are
  `{x, y, z, qx, qy, qz, qw, gripper}`. `BimanualGelloEE.seed_from_robot()`
  is a bring-up helper that logs the FK output against the robot's actual EE
  state for sanity-checking before teleop starts.
- **SpaceMouse** ([spacemouse.py](lerobot_teleoperator_spacemouse/lerobot_teleoperator_spacemouse/spacemouse.py))
  integrates twist into an absolute EE pose; the device output is read in a
  drain-loop because pyspacemouse processes one HID report per `read()` and
  the device emits separate reports per channel at ~100 Hz — without draining,
  the queue builds up and the robot keeps tracking the "old" twist after the
  joystick is released. Buttons latch the gripper target (left=close,
  right=open). Always call `seed_state()` (or `BimanualSpaceMouse.seed_from_robot()`)
  before the first `get_action()` so the integrated pose starts at the actual
  arm pose.
  - **The device mounting lives in `LINEAR_DEVICE_TO_BASE` /
    `ANGULAR_DEVICE_TO_BASE`, not in the sign trims.** `teleop.yaml`'s
    `translation_signs` is identity because that matrix already encodes this
    rig; a `-1` on top of it swaps X and Y rather than mirroring one axis.
    `_validate_device_map()` rejects a reflection at import so a channel can
    never end up mirror-imaged.
  - Full deflection is exactly a normalized ±1 policy action (`translation_scale`
    / `rotation_scale` are osc_pose.json's `output_max`), so teleop and policy
    drive the controller through identical units.

## Camera stack

- **ARV** ([arv.py](lerobot_camera_arv/lerobot_camera_arv/arv.py)) — Aravis GigE
  for Basler BFS. Captures at 8× the configured output resolution
  (`DOWNSCALE_FACTOR = 8`) and software-downsamples with `INTER_AREA` to avoid
  on-camera ROI cropping. **Frame-drain pattern**: Aravis buffers form a FIFO,
  so `_fetch_frame` blocks for one frame then drains any newer ready buffers
  (recycling the older one back to the camera) so we always decode the freshest
  image. The same pattern is needed for FRAMOS — RealSense's
  `wait_for_frames()` does the equivalent internally.
- **FRAMOS** ([framos.py](lerobot_camera_framos/lerobot_camera_framos/framos.py))
  — D415e (RealSense over GigE) via the FRAMOS-built `pyrealsense2`. RGB+depth
  share one IP on different GVSP channels. The D415e only supports stream
  FPS in `{6, 15, 30, 60, 90}`; `_snap_stream_fps` rounds whatever LeRobot
  asks for (often 20) to the nearest supported value.
  - `get_cropped_point_cloud` crops to a world-axis-aligned **box** (matching
    the sim collect crop, `STUDENT_INPUT_PARITY.md` F8), then drops isolated
    points via `_filter_lone_points`. That filter is an **O(N) numpy voxel
    bucket count**, not a KD-tree radius query and not open3d's
    `remove_radius_outlier` — the latter returned a different keep-set
    run-to-run on identical input, which is irreproducible policy input.
    Off by default (`cameras.yaml` `defaults.framos.blob_filter`) because it
    changes what the policy sees.

Both camera modules guard the heavy import with `try/except` in `__init__.py`
so config-only consumers can import the configs without pulling in Aravis or
librealsense2. **That guard also swallows a genuine syntax error in
`framos.py`** and leaves a stale `.pyc` importable — if the FRAMOS camera
silently stops existing, run `python -m compileall lerobot_camera_framos`
before believing the import.

## Scripts

Bash wrappers around the LeRobot CLIs. All scripts assume the venv is active and
read hosts/ports/rates from `config/` via `scripts/_config.sh`.

| Script | What it does | Mode |
|---|---|---|
| `teleop.sh` | Bimanual GELLO joint-mode teleop | `JOINT_POS` |
| `gello_ee_teleop.sh` | Bimanual GELLO EE-mode teleop (FR3 FK on leader) | `EE_POS` |
| `spacemouse_teleop.sh` | Bimanual SpaceMouse EE-mode teleop | `EE_POS` |
| `single_arm_delta_teleop.sh` | Single-arm SpaceMouse EE-delta teleop | `EE_DELTA` |
| `record_data.sh <repo_id> <n_eps> <task> <out_dir> <resume>` | Record GELLO joint teleop dataset → HuggingFace | joint |
| `ee_record_data.sh <repo_id> <n_eps> <task> <out_dir> <resume>` | Record GELLO EE teleop dataset | EE |
| `replay.sh <repo_id> <episode>` | Replay one episode of a recorded dataset | joint |
| `train.sh <repo_id> <policy_repo> <bs> <steps> <policy_type> <resume> <config>` | Train a policy with wandb logging, upload to HF | — |
| `rollout_policy.sh <repo_id> <n_eps> <policy_repo> <out_dir>` | Roll out a policy and log trajectories | EE |
| `home_pose.py` | Save / drive named home configurations (`home_poses/*.json`) | joint |
| `openpi_client_franka.py` | Single-arm OpenPI inference client; DROID-style joint-velocity observations to a remote websocket policy | joint |
| `deploy_nuc_server.sh <mario\|luigi>` | Resolve `torque:` config, copy the torque server + controller to a NUC, restart under `chrt -f 80` | — |
| `check_osc_parity.py` | Diff `osc_torque_controller` against robosuite's real `osc.py` / `control_utils.py` | — |
| `check_osc_e2e.py` | Same, but through the whole `send_action` → server path | — |
| `check_osc_axes.py` | Move the arm one OSC axis at a time; reports commanded-vs-measured | EE |
| `sweep_sim_match.py` | Search `friction_kc` + the delta fudges against a sim reference | EE |
| `../sysid/sweep_gains.py` | Search the OSC gain scales (per axis) against one sim trajectory; rejects trials that saturate the torque clamp | EE |
| `check_spacemouse.py` | Print raw SpaceMouse channels and the base-frame delta they become | — |
| `measure_joint_friction.py` | Per-joint Coulomb/viscous friction; sets `torque.friction.coulomb_nm` | joint |
| `local_module_check.sh` | Editable-install + uninstall recipe for all six packages | — |

Device paths (GELLO USB ports, SpaceMouse hidraw nodes) and motion scales come
from `config/teleop.yaml`. **Which** device a single-arm rig uses is a separate
setting — `teleop_device` on the rig profile in `config/rig.yaml` — because the
leader the operator holds is independent of which follower arm it drives. Both
SpaceMice enumerate as identical HID devices, so grabbing the wrong one
connects cleanly and then does nothing.

`scripts/old/` holds pre-LeRobot prototypes; don't depend on them.

## Conventions worth following

- **Never hardcode an environment constant.** IPs, ports, gains, limits,
  transforms, rates, device paths, torque-loop constants — `config/*.yaml`, read
  via `franka_config` (or `torque_config` on the NUC side).
- **Tuples over lists at the RPyC boundary.** See `franka_process.py`.
- **The control law runs on the NUC; the goal is set here.** Anything that must
  respond faster than a policy period (damping, Coriolis, torque limiting) belongs
  in `pylibfranka_control.py`; anything at policy rate (goal composition, safety
  shaping, gains) belongs in `bimanual_franka.py`. Re-deploy the NUC side after
  touching it or after editing `torque:` — the workstation copy is not what runs.
- **Pure (action, state) → action transforms in `safety.py`.** Don't push
  side effects into `ActionSafetyScreen`; it is intended to be a pre-dispatch
  shaping layer.
- **One thread pool per subsystem.** `BimanualFranka` keeps a `_camera_pool`;
  `MultiRobotWrapper` keeps a `_pool`. Don't share or replace these without
  thinking about teardown order.
- **`l_` / `r_` prefixes everywhere** for bimanual action/observation keys.
  Leader configs use the per-arm `*LeaderFields` dataclass (a plain dataclass,
  not a `TeleoperatorConfig` subclass) to keep draccus from recursing through
  the choice registry when building the bimanual CLI.
- **No emojis, no extra docs.** Existing files are sparse and direct — match
  that style.
- **Data outside the repo.** Datasets, eval rollouts, and trained policies
  live under `~/franka_data/`. The only thing the repo tracks is code +
  reference frames + config.

## Tests

`tests/` holds full-stack equivalence tests against robosuite's **actual**
`OperationalSpaceController` class (mujoco and pylibfranka are stubbed; the real
controller and the real `_ArmSession._compute_tau` run). No pytest needed:

```
python tests/test_osc_stack.py         # send_action → goal → tau vs robosuite
python tests/test_spacemouse_action.py # stick deflection → tau vs the sim policy
python tests/test_grippers.py          # WSG MOVE coalescing + fault recovery (loopback fake)
```

Run these after touching `osc_torque_controller.py`, `bimanual_franka.py`,
`pylibfranka_control.py`, `safety.py`, `wsg.py` or the SpaceMouse. They cover the
gain remap, the delta envelope, the goal-orientation hold rule, both
`uncouple_pos_ori` settings, the nullspace reference, the torque clamp/rate
limiter, and that every hardware-bridging knob is a no-op at its default.

**A tuning knob must be pinned in the parity harness or it silently disables the
suite.** `make_robot` builds the robot from `SingleArmFrankaConfig`, so any field
it does not override is inherited from the rig and fed to *both* sides of the
comparison — the test then asserts the rig agrees with itself and passes. This
happened: `ee_rotation_fudge=0.35` turned 8 tests red with the control stack
untouched. `_SIM_PARITY_KNOBS` lists what is pinned, and
`test_every_hardware_knob_is_pinned` fails on any config field not classified as
pinned, session-side or inert — **including one newly added to a robot config**,
which is how it should be read when it starts failing after a merge.

Coverage worth knowing about:
- **Per axis**, both signs and several magnitudes: translation matches to ~1e-14,
  rotation to ~1e-8 (robosuite's `quat2mat` rounds through float32).
- **Trajectories over all 3^6 = 729 combinations** of translation/rotation axis
  signs, stepped with the arm state evolving between steps. Single-step tests
  cannot catch a `goal_ori` divergence, because the disagreement only compounds
  once it is carried across steps.
- **Device axis → base axis** for the SpaceMouse, asserted rather than assumed:
  `x→lin Y-  y→lin X+  z→lin Z+  roll→ang X+  pitch→ang Y+  yaw→ang Z-`.

`orientation_error` is robosuite's sin-based `0.5*sum(rc_i x rd_i)`, so a
t-radian error reports `sin(t)` — a 4% shortfall at the 0.5 rad envelope bound.
That is pinned by a test; do not "fix" it into a true axis-angle error or the
controller stops matching the policies' training dynamics.

`ActionSafetyScreen` is bypassed in the parity tests (robosuite has no
equivalent) and tested separately — if a parity test starts failing on a
downward-z action, suspect the worktable brake before the controller.

## When something breaks

The README's "Common errors" section covers the operator-facing failure
modes: UDP timeouts (check NUC SSH logs), rough-collision faults (Franka UI),
arms in a dangerous pose (switch to Program mode and guide-by-hand from the
EE buttons), unresponsive grippers (open question).

For code-level debugging:

- Bad/laggy camera frames → check the drain loop is recycling buffers
  (`logger.debug` in `_fetch_frame` reports how many it drained per call).
- Stale teleop input → SpaceMouse: check the HID drain loop; GELLO async:
  check the read thread is alive.
- Arm limp or not tracking → the server holds the current pose if no goal
  arrives for `torque.loop.stale_goal_timeout_s` (0.5 s). Check the policy loop
  rate and `~/pylibfranka_server.log` on the NUC.
- `communication_constraints_violation` → the RT loop missed its 1 kHz deadline.
  Check the server is running under `chrt -f 80` and that nothing else is
  saturating the cores in `torque.loop.rt_cpu_candidates`.
- A gain change had no effect → you did not re-run `deploy_nuc_server.sh`. The
  NUC runs its own copy of `control.yaml`'s `torque:` block.
- `RuntimeError: Cannot resolve the torque block` on the NUC → the deploy did
  not write `nuc_control_config.py`. Re-run the deploy script; it fails the
  import check rather than letting the arm run on a stale gain.
- Point cloud / EE geometry in the wrong place → check the frame convention
  before touching math: `robot_base_in_world` maps base → world directly.
  `python -m franka_config dump world` shows what the stack actually sees.
- `ModuleNotFoundError: franka_config` → run
  `uv pip install --no-deps -e ~/franka_ws/franka_config -C editable_mode=compat`,
  or set `FRANKA_CONFIG_DIR`.
- Gripper closes then refuses to open → a MOVE that ends with the axis blocked
  (i.e. any grasp) faults the WSG, which then rejects the next MOVE. `move()` is
  fire-and-forget, so the reader thread is the only place that error is visible;
  it logs at warning level and the sender emits `STOP()` ahead of the next
  MOVE to clear the axis. Grep the run log for `[WSG] ERR`.
- Gripper FAST STOP latched after a crash → re-`connect()` clears it via
  `ack_fast_stop()` in `WSG.__init__`.
