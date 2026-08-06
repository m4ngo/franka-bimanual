# CLAUDE.md

Real-world control stack for a **bimanual Franka FR3** setup at TRI. The
workspace is a collection of LeRobot plugin packages plus shell scripts that
drive the standard `lerobot-teleoperate` / `lerobot-record` / `lerobot-replay`
/ `lerobot-train` CLIs against this hardware.

See `README.md` for hardware-specific bring-up procedures (FCI enablement,
control-box power, NUC `start_control.sh`, calibration). This file is the
codebase-level orientation.

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

Each NUC runs `./start_control.sh` which exposes the FR3 over RPyC; the
workstation connects directly via `rpyc.classic.connect` (one connection per
arm) and bypasses `net_franky.franky`'s singleton (IP,PORT) limitation.

## Central configuration — read this first

**`config/*.yaml` is the single source of truth for every environment constant**:
world origin, per-arm base poses, arm IPs/ports, camera intrinsics/extrinsics,
control gains, safety limits, fudge factors, and the control rate. Nothing
outside `config/` should hardcode any of them. See
[config/README.md](config/README.md) for the file-by-file breakdown.

Read it through the `franka_config` package (never by parsing YAML yourself):

```python
import franka_config as fc
fc.control_fps()                    # 20 — one rate for teleop/record/rollout/sysid
fc.arm("right").robot_ip
fc.robot_base_in_world("left")      # Pose: p_world = R @ p_base + t
fc.camera("cam_2").calibration.cam_in_world
fc.control("gains.ee_pd.kp")
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
editable_mode=compat` (`franka_config` first) plus the non-PyPI deps (`net_franky`, FRAMOS-built
`pyrealsense2` from `~/librealsense2/wrappers/python/`, `PyGObject<3.52`,
`pyspacemouse`, `dynamixel-sdk`).

## Robot stack — `lerobot_robot_bimanual_franka`

The follower is composed of three subsystems, each isolated in its own module:

- [bimanual_franka.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/bimanual_franka.py)
  — the `Robot` subclass. Holds a `MultiRobotWrapper`, two `WSG` grippers, six
  cameras, and an `ActionSafetyScreen`. `get_observation()` reads cameras in a
  thread pool while batch-querying both arms' kinematic state in parallel;
  it caches that snapshot so the immediate `send_action()` skips a redundant
  RPyC round-trip.
- [franka_process.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/franka_process.py)
  — the RPyC bridge. **Read the docstring before editing.** A big `_SERVER_HELPERS`
  string is `exec`'d on the NUC at connect time so each control-loop op is one
  round-trip per arm. Two important wire-format constraints encoded there:
  - **Tuples, not lists.** brine encodes immutable values; lists cross as
    netrefs that cost a round-trip per element and spam `AttributeError` when
    numpy probes `__array__`.
  - **Read `cb_robot.state` directly under a `with` block.** The
    `get_last_callback_data` accessor leaks `state_mutex` on `AttributeError`;
    the helper recovers any lock left dangling by a crashed prior session.
  - Recoverable Franka errors (`UDP receive: Timeout`,
    `communication_constrains_violation`, `current mode ("Reflex")`,
    `type of motion cannot change`) trigger `recover_from_errors()` on the
    server side and a warning rather than an exception. Edit `_RECOVERABLE_ERRORS`
    if you add another known-recoverable string.
  - The 6×7 EE Jacobian is cached and recomputed only when joint angles drift
    past `_JACOBIAN_CACHE_Q_THRESHOLD` (0.5 rad L∞). This matters when adding
    code that needs fresh Jacobians.
- [rig_config.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/rig_config.py)
  — the bridge from `config/rig.yaml` profiles to concrete camera configs and
  per-arm connection fields. Keeps `franka_config` free of LeRobot imports.
- [safety.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/safety.py)
  — `ActionSafetyScreen` applies pre-dispatch shaping to either joint-velocity
  or EE-twist commands (all thresholds from `config/control.yaml`):
  - **Worktable brake**: caps downward velocity by `sqrt(2·MAX_DECEL·clearance)`
    so the EE can stop before reaching `WORKTABLE_HEIGHT + DISTANCE_MIN`.
    Heights and vertical velocities are evaluated in **world frame** — the
    screen is constructed with each arm's `robot_base_in_world`, so one table
    plane covers arms whose bases differ in height or yaw. The EE is a
    **sphere**, not a point (`worktable_brake.ee_sphere`, per-arm overridable
    in `arms.yaml`): its centre is a TOOL-frame offset that rotates with the
    gripper, clearance is measured from the sphere's bottom, and the centre's
    velocity carries the `ω × r` lever term — so a tilted or rotating gripper
    can't graze the table while the TCP still reads clear. EE mode corrects the
    twist along world-up only (horizontal and angular parts untouched); joint
    mode uniformly scales the whole vector to preserve direction.
  - L2-norm clamps on joint velocity and EE linear/angular velocity
    (`limits.*` in `config/control.yaml`).
  - Bimanual arm-repel is **not yet implemented** (noted in the module
    docstring).
- [wsg.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/wsg.py)
  — Schunk WSG (GCL protocol) gripper driver. One socket, two threads
  (reader + sender) coordinated by a `Condition`. Designed so `move()` only
  hits the wire when the target actually changes (coalesced by
  `_TARGET_CHANGE_THRESH_MM`, capped at `_MIN_MOVE_INTERVAL_S`) — this keeps
  motion plans from carpet-bombing the gripper, which is what makes it feel
  laggy. Blocking ops (`HOME`/`GRIP`/`RELEASE`) use a token-matched
  `_Waiter` queue. `bye()` is called on `close()` to avoid latching FAST STOP.

### Robot action schema

`BimanualFranka` has two modes selected by `use_ee_pos`:

- **Joint mode** (`JOINT_POS`): action keys `l_joint_1…l_joint_7,
  l_gripper, r_joint_1…r_joint_7, r_gripper`. Internally converted to
  joint-velocity commands with `gains.joint_pd` from `config/control.yaml`.
- **EE mode** (`EE_POS` / `EE_DELTA`): action keys `l_{x,y,z,qx,qy,qz,qw,gripper}`
  and `r_*`. Internally converted to Cartesian twists with `gains.ee_pd`.
  Quaternion error uses a Hamilton-product formulation with hemisphere
  selection (`q_err[3] < 0` → negate).

Gripper values are normalised to `[0, 1]` against
`gripper.wsg.true_max_mm` (110.0 mm).

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
  hardware-specific defaults.
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

Both camera modules guard the heavy import with `try/except` in `__init__.py`
so config-only consumers can import the configs without pulling in Aravis or
librealsense2.

## Scripts

Bash wrappers around the LeRobot CLIs — these encode the IP/port/USB
constants for this exact rig. All scripts assume the venv is active.

| Script | What it does | Mode |
|---|---|---|
| `teleop.sh` | Bimanual GELLO joint-mode teleop | `use_ee_pos=false` |
| `gello_ee_teleop.sh` | Bimanual GELLO EE-mode teleop (FR3 FK on leader) | `use_ee_pos=true` |
| `spacemouse_teleop.sh` | Bimanual SpaceMouse EE-mode teleop | `use_ee_pos=true` |
| `record_data.sh <repo_id> <n_eps> <task> <out_dir> <resume>` | Record GELLO joint teleop dataset → HuggingFace | joint |
| `ee_record_data.sh <repo_id> <n_eps> <task> <out_dir> <resume>` | Record GELLO EE teleop dataset | EE |
| `replay.sh <repo_id> <episode>` | Replay one episode of a recorded dataset | joint |
| `train.sh <repo_id> <policy_repo> <bs> <steps> <policy_type> <resume> <config>` | Train a policy with wandb logging, upload to HF | — |
| `rollout_policy.sh <repo_id> <n_eps> <policy_repo> <out_dir>` | Roll out a policy in EE mode and log trajectories | EE |
| `openpi_client_franka.py` | Single-arm (right) OpenPI inference client; sends DROID-style joint-velocity observations to a remote websocket policy | joint |
| `local_module_check.sh` | Editable-install + uninstall recipe for all five packages | — |

Device paths (GELLO USB ports, SpaceMouse hidraw nodes) and motion scales come
from `config/teleop.yaml`. **Which** device a single-arm rig uses is a separate
setting — `teleop_device` on the rig profile in `config/rig.yaml` — because the
leader the operator holds is independent of which follower arm it drives. Both
SpaceMice enumerate as identical HID devices, so grabbing the wrong one
connects cleanly and then does nothing.

`scripts/old/` holds pre-LeRobot prototypes; don't depend on them.

## Conventions worth following

- **Never hardcode an environment constant.** IPs, ports, gains, limits,
  transforms, rates, device paths → `config/*.yaml`, read via `franka_config`.
- **Tuples over lists at the RPyC boundary.** See `franka_process.py`.
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
- "Robot already connected on this (ip, port)" — net_franky's singleton
  caching; the direct-RPyC path in `franka_process.py` is the workaround,
  not the cause. Don't reintroduce `franky.Robot(ip)` calls on the workstation.
- Point cloud / EE geometry in the wrong place → check the frame convention
  before touching math: `robot_base_in_world` maps base → world directly.
  `python -m franka_config dump world` shows what the stack actually sees.
- `ModuleNotFoundError: franka_config` → run
  `uv pip install --no-deps -e ~/franka_ws/franka_config -C editable_mode=compat`,
  or set `FRANKA_CONFIG_DIR`.
- Gripper FAST STOP latched after a crash → re-`connect()` clears it via
  `ack_fast_stop()` in `WSG.__init__`.
