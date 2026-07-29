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
  (`192.168.201.10`, gripper `192.168.2.20`, RPyC port `18812`).
- **`luigi@192.168.3.11`** NUC controls the **left** arm
  (`192.168.200.2`, gripper `192.168.2.21`, RPyC port `18813`).
- Six GigE cameras (4× Basler ARV, 2× FRAMOS D415e) — IP/serial map in
  `BimanualFrankaConfig` and the README.

Each NUC runs `./run_server.sh`, which launches `pylibfranka_server.py` pinned
to **CPUs 0-1**. The server is only an RPyC ↔ shared-memory proxy; it spawns one
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
- The loop's core is chosen at spawn from `/proc/interrupts`. Both the arm's UDP
  and the workstation's RPyC ride one NIC here whose MSI-X queues sit on most of
  2-7; steering the IRQs away instead would need root.
- The loop writes before it computes. See `pylibfranka_control.py`'s docstring —
  the robot grades response latency, not tick duration.

## Package layout

The repo is **five editable LeRobot plugin packages** in a flat layout plus a
`scripts/` directory:

```
franka_ws/
├── lerobot_robot_bimanual_franka/    # follower: two FR3 arms + WSG grippers + 6 cameras
├── lerobot_teleoperator_gello/       # leader: joint-mode and EE-mode GELLOs
├── lerobot_teleoperator_spacemouse/  # leader: 3Dconnexion SpaceMice (EE only)
├── lerobot_camera_arv/               # Aravis GigE cameras (Basler BFS)
├── lerobot_camera_framos/            # FRAMOS D415e via FRAMOS librealsense2
├── scripts/                          # bash wrappers around lerobot-* CLIs
├── config/                           # calibration JSONs land here
└── frames/                           # per-camera reference snapshots
```

Each package self-registers with LeRobot's config registries via
`@RobotConfig.register_subclass`, `@TeleoperatorConfig.register_subclass`, and
`@CameraConfig.register_subclass` decorators — that is how the
`--robot.type=bimanual_franka` / `--teleop.type=bimanual_gello` strings on the
CLI resolve to classes here.

Setup commands live in [scripts/local_module_check.sh](scripts/local_module_check.sh)
— it installs the five packages with `uv pip install --no-deps -e ... -C
editable_mode=compat` plus the non-PyPI deps (`net_franky`, FRAMOS-built
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
  stays numpy-only with no package-relative imports. Verify changes with
  `scripts/check_osc_parity.py`, which diffs it against robosuite's real modules.
- [safety.py](lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/safety.py)
  — `ActionSafetyScreen` applies pre-dispatch shaping to goal poses (torque
  paths) or velocity commands (the remaining velocity callers):
  - **Worktable brake**: caps downward motion by `sqrt(2·MAX_DECEL·clearance)`
    so the EE can stop before reaching `WORKTABLE_HEIGHT + DISTANCE_MIN`.
    `shape_goal` / `shape_joint_goal` express that envelope as a bound on the
    downward position error (the impedance loop settles at `v = (kp/kd)·error`)
    plus a hard floor the goal can never cross; `shape_ee` / `shape_joint` keep
    the velocity-domain form.
  - L2-norm clamps on joint velocity (2.0 rad/s) and EE linear/angular
    velocity (1.0 m/s, 2.0 rad/s). These bound the *velocity-domain* callers
    only; the torque paths are bounded by the NUC's speed guard instead.
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

`BimanualFranka` has three modes selected by `control_mode`. All land on torque;
`send_action` is robosuite's `set_goal` half, the NUC runs `run_controller`.

- **`JOINT_POS`**: keys `l_joint_1…l_joint_7, l_gripper, r_joint_1…`. Dispatched
  as a joint-impedance goal at `JOINT_IMPEDANCE_KP`.
- **`EE_POS`**: keys `l_{x,y,z,qx,qy,qz,qw,gripper}` and `r_*`, an absolute pose
  used directly as the OSC goal.
- **`EE_DELTA`**: same keys as a delta. `goal = ee_pos + delta`, rebuilt from the
  *current* pose every policy step (never accumulated onto the previous goal),
  delta clipped to osc_pose.json's ±0.05 m / ±0.5 rad envelope.

The `kp`/`kd` action entries use the sim's exponential remap, matching
`multi-fast/utils/envs/libero.py`: `kp = 150·10^a_kp` clipped to [0, 1500],
`damping_ratio = 1·10^a_kd` clipped to [0, 10], `kd = 2√kp·ratio`.

Gripper values are normalised to `[0, 1]` against
`WSG.GRIPPER_TRUE_MAX_MM = 110.0` mm.

Camera frames are 224×224 RGB by default and exposed in the observation under
`cam_1` … `cam_6`. Camera failures degrade to `blank_frame()` (last known
image) rather than raising.

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
| `teleop.sh` | Bimanual GELLO joint-mode teleop | `JOINT_POS` |
| `gello_ee_teleop.sh` | Bimanual GELLO EE-mode teleop (FR3 FK on leader) | `EE_POS` |
| `spacemouse_teleop.sh` | Bimanual SpaceMouse EE-mode teleop | `EE_POS` |
| `record_data.sh <repo_id> <n_eps> <task> <out_dir> <resume>` | Record GELLO joint teleop dataset → HuggingFace | joint |
| `ee_record_data.sh <repo_id> <n_eps> <task> <out_dir> <resume>` | Record GELLO EE teleop dataset | EE |
| `replay.sh <repo_id> <episode>` | Replay one episode of a recorded dataset | joint |
| `train.sh <repo_id> <policy_repo> <bs> <steps> <policy_type> <resume> <config>` | Train a policy with wandb logging, upload to HF | — |
| `rollout_policy.sh <repo_id> <n_eps> <policy_repo> <out_dir>` | Roll out a policy in EE mode and log trajectories | EE |
| `openpi_client_franka.py` | Single-arm (right) OpenPI inference client; sends DROID-style joint-velocity observations to a remote websocket policy | joint |
| `deploy_nuc_server.sh <mario\|luigi>` | Copy the torque server + controller to a NUC and restart it under `chrt -f 80` | — |
| `check_osc_parity.py` | Diff `osc_torque_controller` against robosuite's real `osc.py` / `control_utils.py` | — |
| `check_osc_e2e.py` | Same, but through the whole `send_action` → server path | — |
| `check_osc_axes.py` | Move the arm one OSC axis at a time; reports commanded-vs-measured | EE |
| `check_spacemouse.py` | Print raw SpaceMouse channels and the base-frame delta they become | — |
| `measure_joint_friction.py` | Per-joint Coulomb/viscous friction from constant-velocity torque; sets `friction_kc`'s constants | joint |
| `local_module_check.sh` | Editable-install + uninstall recipe for all five packages | — |

USB ports: **left GELLO `/dev/ttyUSB1`, right GELLO `/dev/ttyUSB0`**. SpaceMice
default to `/dev/hidraw2` / `/dev/hidraw3` (the script overrides the config
defaults of `/dev/hidraw4` / `/dev/hidraw5`).

`scripts/old/` holds pre-LeRobot prototypes; don't depend on them.

## Conventions worth following

- **Tuples over lists at the RPyC boundary.** See `franka_process.py`.
- **The control law runs on the NUC; the goal is set here.** Anything that must
  respond faster than a policy period (damping, Coriolis, torque limiting) belongs
  in `pylibfranka_control.py`; anything at policy rate (goal composition, safety
  shaping, gains) belongs in `bimanual_franka.py`. Re-deploy the NUC side after
  touching it — the workstation copy is not what runs.
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
  reference frames.

## Tests

`tests/` holds full-stack equivalence tests against robosuite's **actual**
`OperationalSpaceController` class (mujoco and pylibfranka are stubbed; the real
controller and the real `_ArmSession._compute_tau` run). No pytest needed:

```
python tests/test_osc_stack.py        # send_action → goal → tau vs robosuite
python tests/test_spacemouse_action.py # stick deflection → tau vs the sim policy
```

Run both after touching `osc_torque_controller.py`, `bimanual_franka.py`,
`pylibfranka_control.py` or the SpaceMouse. They cover the gain remap, the delta
envelope, the goal-orientation hold rule, both `uncouple_pos_ori` settings, the
nullspace reference, the torque clamp/rate limiter, and that every
hardware-bridging knob is a no-op at its default.

Coverage worth knowing about:
- **Per axis**, both signs and several magnitudes: translation matches to ~1e-14,
  rotation to ~1e-8 (robosuite's `quat2mat` rounds through float32).
- **Trajectories over all 3^6 = 729 combinations** of translation/rotation axis
  signs, stepped with the arm state evolving between steps. Single-step tests
  cannot catch a `goal_ori` divergence, because the disagreement only compounds
  once it is carried across steps.
- **Device axis -> base axis** for the SpaceMouse, asserted rather than assumed:
  `x->lin Y-  y->lin X+  z->lin Z+  roll->ang Y-  pitch->ang X+  yaw->ang Z+`.

`orientation_error` is robosuite's sin-based `0.5*sum(rc_i x rd_i)`, so a
t-radian error reports `sin(t)` -- a 4% shortfall at the 0.5 rad envelope bound.
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
  arrives for `_STALE_GOAL_TIMEOUT_S` (0.5 s). Check the policy loop rate and
  `~/pylibfranka_server.log` on the NUC.
- `communication_constrains_violation` → the RT loop missed its 1 kHz deadline.
  Check the server is running under `chrt -f 80` and that nothing else is
  saturating cores 2-7.
- Gripper FAST STOP latched after a crash → re-`connect()` clears it via
  `ack_fast_stop()` in `WSG.__init__`.
