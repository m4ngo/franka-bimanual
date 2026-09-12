# Right-arm rig: session handoff

Session of 2026-09-05 (approx 19:50-21:45). Written for a fresh Claude session on
a different account. Transient: delete once the work below is committed.

Goal that started the session: the right FR3 was blinking red and could not be
commanded. That turned out to be three unrelated problems stacked on each other,
only one of which was the LED.

## Current hardware state (as of 21:45, 2026-09-05)

    RIGHT (mario, 192.168.201.10)   RobotMode.Idle  errors []  m_load 0.0   READY
    LEFT  (luigi, 192.168.200.2)    RobotMode.Other errors ["cartesian_reflex"]  m_load 0.0

Torque server on mario is UP: 18812 (torque) and 18822 (gripper) both listening,
deployed 20:35. The teleop attempt at 21:34 failed only because the E-stop was
engaged at that moment; it has since been released.

Nothing has been driven yet. The first successful teleop run has NOT happened.

## What was actually wrong (three separate things)

1. **Joints locked in Desk.** `robot_mode` read `Other` while the left arm read
   `Idle`. FCI was already enabled the whole time (port 1337 open, libfranka
   connected and read state fine) -- enabling FCI and unlocking brakes are
   independent, which is why "enable FCI" was never the fix.

2. **pylibfranka could not import on mario.** `ImportError: libfranka.so.0.18:
   cannot open shared object file`. The extension module's RUNPATH pointed at
   `~/libfranka/build/lib.linux-x86_64-cpython-312/pylibfranka`, which stopped
   existing on Aug 28 when someone reused `~/libfranka/build/` for a plain CMake
   build of libfranka 0.18.1 (the old tree was moved to `build-0.18.0/`). The
   correct 0.18.0 `libfranka.so.0.18` was sitting next to the extension in
   site-packages the whole time, but RUNPATH had no `$ORIGIN`.

3. **E-stop engaged** at launch time -- reported as
   `Move command rejected: command not possible in the current mode ("User stopped")`.

Mode -> meaning, since all three look similar from the arm:

    UserStopped   E-stop / external activation device engaged   release the button
    Other         brakes locked (or a latched reflex)           Unlock in Desk
    Idle          ready to arm                                  go

`current_errors` stays empty for the first two. It only reports motion/control
errors, so no safety-layer state ever shows up there.

## Repo changes (uncommitted, NOT yet committed by request)

Nine files. Everything else dirty in the tree belongs to other work
(`ee_goals.py`, `ee_kinematics.py`, `lerobot_source.py`,
`prepare_baseline_datasets.py`, `real_reach*`, `check_reach*`) -- do not sweep
those into the same commit.

    M config/rig.yaml                                          +12
    M lerobot_robot_bimanual_franka/.../__init__.py            exports
    M lerobot_robot_bimanual_franka/.../config_single_arm_franka.py   1 line
    M scripts/lerobot_record_homed_single_arm.py               --rig
    M scripts/single_arm_record_data_homed.sh                  $9 rig positional
    M scripts/single_arm_teleop.sh                             --rig banner
    M scripts/teleop_single_arm.py                             --rig
    ?? lerobot_robot_bimanual_franka/.../config_single_arm_right.py   new
    ?? lerobot_robot_bimanual_franka/.../single_arm_right.py          new

Proposed commit message:

    Add single_arm_right rig profile for the physical right FR3

    Every single-arm profile mapped the r_ key prefix to the LEFT arm, so no
    tooling could drive the right FR3 alone. Adds the profile, a config/robot
    subclass pair, and a --rig selector on both drivers and wrappers.
    Default is unchanged.

## The problem the profile solves

`config/rig.yaml` had no profile driving the physical right arm alone:

    bimanual_franka     l: left,  r: right     (needs both arms healthy)
    single_arm_franka   r: left
    openpi_single_arm   r: left

Both single-arm profiles map the `r_` key prefix to the LEFT FR3, and
`teleop_single_arm.py` / `lerobot_record_homed_single_arm.py` hardcoded
`_PROFILE = "single_arm_franka"`. So every `single_arm_*.sh` script drove the
left arm. This is deliberate per CLAUDE.md (keeps old `r_*` datasets readable)
but it means the key prefix is never the physical arm.

New profile:

    single_arm_right:
      arms: {r: right}
      teleop_device: right
      cameras: [cam_3, cam_4, cam_2]
      depth_cameras: [cam_2]
      depth_center_arm: r
      control_mode: EE_DELTA

`cam_3`/`cam_4` are the right-arm wrist cameras and `cam_2` the world scene, per
`config/cameras.yaml` role/mount metadata. The left profile uses `cam_1`/`cam_5`,
which are the left-arm wrists -- the profiles are internally coherent.

Verified resolution:

    single_arm_franka  -> arm=left   robot=192.168.200.2   gport=18823  cams=(cam_1, cam_5, cam_2)
    single_arm_right   -> arm=right  robot=192.168.201.10  gport=18822  cams=(cam_3, cam_4, cam_2)

Note `gport`: the right WSG server is 18822, the left is 18823. That is a
different process from the torque server and must resolve through the profile.

## Design choices and why

- **Subclassed rather than refactored.** `SingleArmRightConfig` extends
  `SingleArmFrankaConfig` and overrides only the profile-derived fields.
  A factory that parameterises the base class would be less duplicative, but
  `SingleArmFrankaConfig` is what `tests/test_osc_stack.py`'s `make_robot` builds
  from, and CLAUDE.md is explicit that touching it is how thirteen parity tests
  go red with the controller untouched. Subclassing leaves it byte-identical.
  The subclass adds NO new dataclass fields, so
  `test_every_hardware_knob_is_pinned` (which enumerates
  `dataclasses.fields(SingleArmFrankaConfig)`) is unaffected.

- **Additive only.** `single_arm_franka` and `openpi_single_arm` are untouched
  and re-verified to still resolve to the left arm. Changing the existing
  profile's mapping was rejected: it would silently repoint every existing
  script and dataset workflow.

- **`--rig` defaults to `rig.yaml`'s `default_single_arm_profile`**, now
  `single_arm_right`. It was `single_arm_franka` in every driver while
  `single_arm_teleop.sh` printed `single_arm_right` in its banner without passing
  it, so the banner named one arm and the other was dialled.

- **Robot class naming is load-bearing.** LeRobot's
  `make_device_from_device_class` strips "Config" off the config class name and
  searches `<parent module>`, `<parent>.<lowercased>`, and `config_x -> x`. So
  `SingleArmRightConfig` in `config_single_arm_right.py` requires a
  `SingleArmRight` class in `single_arm_right.py`. Verified the dispatch resolves.

- One incidental fix: `config_single_arm_franka.py`'s `__post_init__` error
  message hardcoded the string "single_arm_franka", which would misreport for the
  subclass. Now uses `self.rig_profile`.

## Machine changes OUTSIDE the repo (invisible to git)

**mario (`192.168.3.10`), patchelf on the pylibfranka extension.** This is the
one with no record on the machine itself beyond a backup file.

    patchelf --set-rpath '$ORIGIN:/opt/openrobots/lib' \
      ~/pylibfranka_env/lib/python3.12/site-packages/pylibfranka/_pylibfranka.cpython-312-x86_64-linux-gnu.so

    backup: _pylibfranka.cpython-312-x86_64-linux-gnu.so.bak-20260905

This restores mario to exactly what luigi already had (luigi's RUNPATH was
already `$ORIGIN:/opt/openrobots/lib` and imported fine). Both NUCs run
pylibfranka 0.18.0 with a byte-identical bundled `libfranka.so.0.18`
(md5 `0e52466024666a02fa96dabb62042dac`).

**This reverts silently if anyone reinstalls or upgrades the pylibfranka wheel
on mario.** `deploy_nuc_server.sh` does NOT undo it (it copies server sources,
not the venv). If `libfranka.so.0.18: cannot open shared object file` comes back,
this is why.

patchelf itself is not installed on mario; it was run from an unpacked PyPI
wheel in `/tmp/pe_x` (may be gone). `pip download patchelf` works there, and
luigi has `/usr/bin/patchelf`.

Also written this session: a memory entry `venv-location.md` noting the venv is
at `~/franka_ws/.venv`, not the `~/.venv` that CLAUDE.md and several scripts
claim. `source ~/.venv/bin/activate` fails silently.

## Coexistence with qirico

A separate stack (`~/qirico/panda_control/`, user `qirico`) drives the same right
arm, as recently as Sep 3. Checked and confirmed non-interfering:

- Their binaries link `~/libfranka/build/libfranka.so.0.18` (their 0.18.1 build);
  ours uses the venv-bundled 0.18.0. The patchelf change touches only our venv.
  Verified their `read_current_pose` still runs after the change.
- `franka_config` is not installed on mario at all, and there is no `rig.yaml`
  anywhere on that machine. Their config is `~/qirico/panda_control/config/robot.yaml`.
  The rig profile work cannot reach them.
- **FCI is single-client.** While our server holds the control token their daemon
  cannot connect, and vice versa. This is the only real conflict. User confirmed
  they are not currently running it.

`deploy_nuc_server.sh` overwrites our own files in mario's home
(`pylibfranka_*.py`, `osc_torque_controller.py`, `franka_jacobian.py`,
`torque_config.py`, `nuc_control_config.py`, `run_server.sh`) and pkills ports
18812/18822. It does not touch `~/qirico/` or `~/libfranka/`.

## Findings worth keeping

- **Payload is unmodelled on both arms.** The right arm had `m_load = 1.8` with
  COM `[0, 0, 0.047242]`, set by qirico's daemon for the Schunk WSG that is
  physically mounted. The power cycle reset it to 0.0. Our stack NEVER sets the
  load -- `pylibfranka_server.py`'s `exposed_set_load` deliberately raises ("not
  routed to the control process"), and there is no payload entry in `config/*.yaml`.
  So 0.0 is likely the plant the existing gains and friction trims were identified
  against, and setting 1.8 now would be a plant change requiring re-tuning.
  Left at 0.0 deliberately. Flagged as relevant to the sysid work: a permanently
  unmodelled ~1.8 kg payload is a systematic gravity error the tuning is absorbing.

- **Left arm reflex, probably the same cause.** Left latched `cartesian_reflex`
  on the translational Z axis (`cartesian_collision [0,0,2,0,0,0]`) reading
  18.86 N with `m_load = 0.0`. User confirmed nothing is touching the EEF.
  18.86 N is about 1.92 kg -- close to a WSG-class gripper. `arms.yaml` configures
  the left as `franka_hand`, but its own comment notes scripts historically drove
  a WSG at 192.168.2.21 there, and the README lists a SCHUNK. Worth checking what
  is physically on that flange; if it is a WSG, unmodelled mass inflates the
  external-force estimate the reflex threshold compares against, which explains a
  trip from a modest bump. Clears safely with `automatic_error_recovery()` (no
  motion). Deliberately left latched.

- **README is stale.** Its "Single-arm (right arm only) ... mario NUC
  192.168.201.10" section contradicts `config/rig.yaml`, where every single-arm
  profile pointed at the left arm. `rig.yaml` is authoritative. The new
  `single_arm_right` profile is what that README section actually describes,
  including the `cam_3`/`cam_4`/`cam_2` camera set.

- **mario has runaway kworkers.** `kworker/0:*+pm` threads eating ~11% CPU each on
  CPU 0, load average ~5.8, uptime 28 days. `run_server.sh` pins the RPyC and
  gripper servers to cores 0-1. Within the placement rules, but new since the
  last run, and worth watching for `communication_constraints_violation`. The RT
  loop itself correctly picked CPU 7 on the last start.

## Pending

1. **Run teleop.** Never successfully executed. Right arm is `Idle` and the
   server is up, so it should work now.
2. **Commit the nine files** (deliberately not done).
3. **Left arm** still latched in `cartesian_reflex`; decide on the gripper mass
   question first.
4. **Decide on the payload** question above -- affects sysid.
5. **Make the NUC patchelf durable**, or at least recorded on the machine. It
   silently reverts on a pylibfranka reinstall.
6. **Fix the stale README** single-arm section.
7. `spacemouse_teleop.sh` (bimanual EE_POS) still never calls
   `BimanualSpaceMouse.seed_from_robot` -- pre-existing, noted in CLAUDE.md,
   untouched this session.

## Resume commands

Preflight (cheap; camera connect takes seconds before it reaches the arm):

    ssh mario@192.168.3.10 'source ~/pylibfranka_env/bin/activate && python -c "
    import pylibfranka
    s = pylibfranka.Robot(\"192.168.201.10\").read_once()
    print(s.robot_mode, s.current_errors)"'

Want `RobotMode.Idle []`. Then:

    ./scripts/single_arm_teleop.sh spacemouse_delta

`--rig=single_arm_right` is now the default and is passed through, so the banner
and the `driving <rig>` log line must agree. `--rig=single_arm_franka` opts back
into the LEFT arm.

Other modes: `gello_ee`, `gello`, `spacemouse_ee` -- same `--rig` flag.
Recording: `single_arm_record_data_homed.sh ... <mode> <depth> single_arm_right`
(rig is the 9th positional).

Re-deploy after any `torque:` or NUC-source change:

    ./scripts/deploy_nuc_server.sh mario

Desk (needs sudo, must be port 443 -- Desk 301-redirects to `https://localhost/desk/`
and drops any other port):

    ./scripts/open_fci.sh right     # then browse https://localhost

## Verification state

    tests/test_osc_stack.py         56/56 passed
    tests/test_spacemouse_action.py 15/15 passed
    tests/test_grippers.py           8/8 passed

Verified: profile resolution both rigs, plugin dispatch to `SingleArmRight`,
left profile unchanged, pylibfranka imports clean on mario without
`LD_LIBRARY_PATH`, qirico's binary still runs, deploy landed, server starts and
pins the RT loop to CPU 7.

NOT verified: any actual arm motion. Nothing has been commanded through the new
profile.
