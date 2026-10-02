# Running a trained policy on the arm

The converters in this directory turn one recording into the three formats the
three methods train on. This is the other end: running each trained policy on
the real FR3 and writing down success and time, so the comparison the directory
exists for can actually be scored.

All three methods -- SAIL, B-Spline and our own multi-fast residual runner --
share the operator protocol, the episode log and the output tree, so their runs
are directly comparable. Only the two baselines need a separate policy process.

## Why there are two processes

Each upstream project needs its own environment (`.venv-sail`, `.venv-bspline`,
from `scripts/setup_baseline_envs.sh`), and they conflict with each other and
with the workspace venv that owns `lerobot`, `franka_config` and the RPyC link
to the arm. So the policy never runs in the same process as the robot:

```
 baseline venv (.venv-sail / -bspline)  workspace venv (~/franka_ws/.venv)
 -------------------------------------  ---------------------------------
 policy_server.py                       rollout.py
   loads the checkpoint         ZMQ       owns the arm, the cameras,
   answers "what action?"   <-------->    the clock and the recording
                                            |
                                            v
                                        send_action -> NUC -> 500 Hz OSC
```

The arm side stays in our venv on purpose: that is where the safety screen, the
rig profiles and the action schema live, and none of it should be reimplemented
against a foreign checkout.

## Running it

Each script starts both halves. `--rig` picks the profile; **check the banner** —
if it says `single_arm_franka` when you meant `single_arm_right`, the flag did
not take and the other arm is about to move.

```bash
./scripts/sail_rollout.sh --start-server \
    --ckpt ~/franka_data/policies/pickup-bowl/sail/<ts>/models/model_epoch_1000.pth \
    --rig=single_arm_right --num-episodes 10

./scripts/bspline_rollout.sh --start-server \
    --ckpt ~/franka_data/policies/pickup-bowl/bspline/<ts>/checkpoints/latest.ckpt \
    --rig=single_arm_right --speed-up-times 1.0 --num-episodes 10

python residual_wrapper/run_residual.py \
    --base-policy ~/franka_data/policies/pickup-bowl/multifast/pretrained_model \
    --residual-policy ~/franka_data/policies/pickup-bowl/multifast/best.pt \
    --rig=single_arm_right --num-episodes 10
```

`sail_rollout.sh` passes SAIL's own `base_cfg_weight_1.json` guide config, as
upstream's README evaluates; `--guide-config ""` runs unguided.

None of them takes an output path. Each works out which dataset its policy was
trained on and files itself under that task automatically -- see **Where a run
goes** below. Pass `--train-dataset <repo-id>` when a checkpoint predates the
stamp and the run cannot work it out.

Each wrapper reaps the policy server's whole process group on exit, including a
Ctrl-C. That matters: an orphaned server keeps the port — the next run would
then handshake with the stale one and silently evaluate the previous checkpoint
under the new one's recorded sha256. The interpreter comes from
`baselines/interpreters.py` (`$SAIL_PYTHON` / `$BSPLINE_PYTHON` override it).

To run the halves separately, start the server yourself and drop
`--start-server`:

```bash
.venv-sail/bin/python    baselines/sail_bridge/policy_server.py    --ckpt-path <CKPT> --port 5556
.venv-bspline/bin/python baselines/bspline_bridge/policy_server.py --ckpt-path <CKPT> --port 5555
```

Then, before the arm is involved at all:

```bash
python scripts/check_policy_server.py sail --port 5556      # or bspline --port 5555
```

prints the checkpoint's handshake and checks that one inference comes back in
the shape the rollout expects. It is the only check that runs the real server
against the real checkpoint; the offline harness fakes both.

The B-Spline server subclasses upstream's own `policy_server_bspline.py` to add
one request (`meta`) and changes nothing else — upstream replies `{}` to a key it
does not know, so it cannot tell the client the image sizes or the spline degree,
and both are load-bearing. The SAIL server does three things upstream's simulator
env did for the policy: frame-stacks observations, processes images to CHW float,
and asks for the whole action sequence (item 11 below). Neither submodule is
edited.

During an episode: **right arrow** ends it as a success, **left arrow** as a
failure, a timeout counts as a failure, and Ctrl-C aborts the run. That verdict
is the measurement — there is no automatic success detector on this rig.

## Where a run goes

Every run is one self-contained directory, grouped by the dataset the policy was
**trained** on rather than by method. That grouping is the experiment: one task's
demonstrations, every method that learned from them, side by side.

```
~/franka_data/outputs/
  HuskyMango/pickup-bowl/
    20260912_143000-sail/
      manifest.json      everything known about the run
      episodes.jsonl     one line per episode, appended as it finishes
      dataset/           the LeRobotDataset recorded during the run
      videos/            one time-aligned mp4 per camera (--save-videos)
      force_profiles.npz the end-effector force at every goal sent (see End-effector force)
      chunks.npz         every plan the policy returned, as base-frame poses, at the step it arrived
      episode_000.html   per-episode 3D page: arm, realized EE path, current plan, dispatched goals
                         (written at run end; re-render with `python -m baselines.rollout_viz <run dir>`)
    20260912_151500-bspline/
    20260912_160200-multifast/
```

Nothing is written into the repo. `--outputs-root` moves the tree, `--no-record`
skips the LeRobotDataset (the manifest and episode log are always written), and
two runs of one method in the same second are refused rather than merged.

### How a run knows its task

The converters stamp the dataset id onto the HDF5 they write
(`--source-repo-id`, defaulting to the dataset argument), the policy servers read
it back off the checkpoint's training file and report it in the `meta`
handshake, and the rollout files itself under it. multi-fast reads
`train_config.json` next to its base-policy checkpoint instead.

`--train-dataset` overrides all of that, and is required when none of it
resolves. When the flag and the checkpoint disagree the flag wins and the
manifest records both with `agrees: false` -- that disagreement is usually a
checkpoint pointed at the wrong directory, and silently filing a run under the
wrong task is the failure this exists to prevent.

## What the manifest holds

Enough to tell, months later, exactly what ran:

| Section | What it answers |
|---|---|
| `run` | id, method, `sweep` (the sweep id, when the run came from one -- null for a standalone rollout), status (`completed` / `interrupted` / `failed`), start, duration |
| `train_dataset` | which task, how it was resolved, whether the flag and checkpoint agreed, and the recording's own episode/frame counts |
| `policy` | checkpoint path, sha256, size and mtime; the server's full `meta`; the control mode and **why** it was chosen |
| `parameters` | every resolved knob -- rates, horizons, speed-up, guidance, tolerances |
| `environment` | rig profile and the **physical** arm behind the `r_` prefix, its IPs and ports, cameras and their resolutions, the `torque:` and `tuning:` blocks in force, the safety floor, the git commit and whether the tree was dirty, host, user, python, argv |
| `outputs` | the recorded dataset's repo id, path, episodes and frames; the videos |
| `summary` | episodes, successes, success rate, and time-to-success (mean/median/min/max) over successes only; the end-effector force `ee_force_n` ([below](#end-effector-force)) |

`episodes.jsonl` carries one line per attempt: the verdict and how it was
reached, wall time, goals dispatched, frames recorded, inferences, slow steps,
guided inferences, achieved vs nominal rate, send-gap mean and max, worst lead,
the end-effector force (`ee_force_n`), whether homing converged, and any abort
reason. It is appended as each episode
finishes, so a run interrupted at the robot still describes everything that did.

The recorded LeRobotDataset carries the same `observation.state` / `action`
features a `run_residual.py` recording does, so all three methods' rollouts are
readable by one set of tools.

## End-effector force

Every real rollout records the force at the end effector, the same way
multi-fast's `eval_fast.py` records it in sim: one sample per control step,
summarised per episode as the mean, median, p95 and max of |F|. The sim side is
in [LIBERO_SIM.md](LIBERO_SIM.md#wrist-force); both use
[force_log.py](force_log.py).

**Where it comes from.** The FR3 has no wrist force sensor. It has a torque
sensor in each of its seven joints, and libfranka turns those into an estimate
of the external wrench at the end effector, `O_F_ext_hat_K`: force (N) then
torque (Nm), in the base frame. Both NUCs' pylibfranka expose it (checked
2026-10-01, alongside `K_F_ext_hat_K` and `tau_ext_hat_filtered`). The control
child publishes it with every state
([pylibfranka_control.py:445](../lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/pylibfranka_control.py#L445)) into a
free slot of the state block
([pylibfranka_shm.py:106](../lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/pylibfranka_shm.py#L106)). The server appends
it to the state bundle, and the workstation unpacks it
([`_unpack`](../lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/franka_process.py#L53)) and exposes it per arm as
[`BimanualFranka.last_ee_wrench`](../lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/bimanual_franka.py#L592).

**When it is sampled.** Once per goal sent, right after `send_action`, whose
state read is the one that goal was anchored on: the
`Dispatcher.send` for SAIL and B-Spline
([rollout_common.py:350](rollout_common.py#L350)), and `_run_episode` for
`run_residual.py` ([run_residual.py:516](../residual_wrapper/run_residual.py#L516))
and `run_residual_openpi.py`. The reach scripts (`real_reach_rollout.py` and
`run_residual.py` with a FAST `.zip`) read it in `RealReach.step`
([real_reach.py:211](../lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/real_reach.py#L211)), with the end-of-period pose, and
put it in `info["ee_force"]` like the sim env does.

**Where it is written.**

| run | per step | per episode |
|---|---|---|
| SAIL, B-Spline, `run_residual.py` (best.pt) | `force_profiles.npz` in the run directory: `ee_force_003`, `ee_torque_003` and `time_003` (seconds since the episode started) for episode 3 | `ee_force_n` in `episodes.jsonl`; the run's average in the manifest's `summary.ee_force_n` |
| reach (`real_reach_rollout.py`, FAST `.zip`) | `ee_force` and `ee_torque` datasets in `episodes.hdf5` ([EPISODE_HDF5.md](../EPISODE_HDF5.md)) | attrs `ee_force_mean_n`, `_median_n`, `_p95_n`, `_max_n` |
| `run_residual_openpi.py` | `force_profiles.npz` in `--viz-dir`, else `<output-dir>_force_profiles.npz` beside the dataset | printed only |

Each episode's end-of-episode line prints `|F| mean/p95/max`, and
`rollout_summary.py` shows an `|F| N avg/p95/max` column for every run that
recorded it.

**It is not the same quantity as the sim's.** robosuite's sensor sits between
the arm and the gripper, so it carries the gripper's own weight and reads about
5 N at rest. libfranka's is the *external* wrench: its model subtracts the end
effector load configured in Desk, so at rest it should read near 0 N plus
estimation error (friction and cable torque reach `tau_ext`; see
`sysid/identify_payload.py`). Compare contact forces between the two, not the
resting level. A wrong end-effector mass in Desk shows up as a constant offset
here. Two more libfranka conventions: positive means the robot pushing on the
environment, which |F| ignores, and the estimate is zeroed near a singularity.
The resting level on this rig has not been measured yet.

**It needs a redeploy.** Until the NUC runs the new server and control child,
the client gets the old 49-float state bundle, the wrench is `None`, and the run
logs `EE force not recorded` once and carries on without it. Redeploy both
NUCs:

```bash
./scripts/deploy_nuc_server.sh luigi
./scripts/deploy_nuc_server.sh mario
```

To read the newest real run's first episode:

```bash
python - <<'EOF'
import numpy as np
from pathlib import Path
real = [p for p in (Path.home() / "franka_data" / "outputs").rglob("force_profiles.npz")
        if "time_000" in np.load(p).files]
npz = max(real, key=lambda p: p.stat().st_mtime)
z = np.load(npz)
f = np.linalg.norm(z["ee_force_000"], axis=1)          # |F| per goal sent, N
t = z["time_000"]                                      # seconds since the start
print(npz.parent.name, f"{len(f)} samples, peak {f.max():.1f} N at {t[f.argmax()]:.2f} s")
EOF
```

## Reading the comparison

```bash
python scripts/rollout_summary.py HuskyMango/pickup-bowl
```

```
HuskyMango/pickup-bowl
  run                        method      eps   ok   rate  median s   mean s  status       notes
  ---------------------------------------------------------------------------------------------
  20260912_143000-sail       sail          4    3    75%      9.10     9.23  completed    100Hz precision eag
  20260912_151500-bspline    bspline       4    3    75%      7.40     7.37  completed    100Hz 2.0x
  20260912_160200-multifast  multifast     4    4   100%      6.10     6.00  completed    20Hz
```

Time is over **successes only**: a failure's duration is the timeout and says
nothing about how fast a method is.

The positional argument is any path prefix under the output root, so the same
command narrows to a suite, a task or one `<timestamp>-<method>` run directory.
`--all` sweeps every task and `--json` emits the manifests instead of the table.

`--sweep <id>` is the other axis: it selects the rollouts of one sweep wherever
they landed and adds a row per backend pooling the whole thing, which is the
comparison a multi-task sweep exists to produce.

```
bsp_2x -- pooled over tasks
  method     tasks   eps    ok   rate  mean median s  notes
  ---------------------------------------------------------
  bspline        8   160   149  93.1%           4.91  2.0x
```

`--sweep latest` picks the newest sweep on disk. Only rollouts started through a
sweep runner carry an id; one launched by hand gets none unless you pass
`--sweep-id` yourself.

## The two speed mechanisms

Both baselines claim to finish faster than the demonstrations, and both need the
goal-push rate decoupled from the camera rate to show it. Observations and
inference run at `obs_fps` (20, the rate everything else on this rig uses); OSC
goals go out at `--exec-fps`. The NUC's torque law runs at 500 Hz regardless, so
this costs nothing there.

**SAIL** modulates speed per step. The last action column is a precision label;
if any label inside a window around the current step is set, that step is
dispatched at `--slow-fps` instead of `--exec-fps`. It also runs a receding
horizon — the previous prediction keeps executing *while* the new one is
inferred (at least `inf_delay` rows, more if inference takes longer), and the
new one is entered at the row matching how many went out since its observation
— and error-adaptive guidance, which conditions the next prediction on the tail
of the current plan but only while the arm is tracking it. All three are on by
default when the checkpoint supports them; `--no-precision` and `--no-eag` turn
the last two off.

Inference on the arm takes real time (about 85 ms per request), longer than
the 4 rows a 16-row chunk has spare at 100 Hz. Rather than stop when the old
plan runs out, the loop applies the paper's latency bound (Sec. 4.4): every row
is sent at most `(horizon - execute_n) / (2 * latency)` Hz, using the last
request's round trip. That keeps the arm moving through inference, at the cost
of a lower top speed when inference is slow: at 100 ms the cap is 40 rows/s.

Two things about the targets shape that loop. Row 0 of every chunk is the pose
the arm was *observed* at — SAIL's controller-invariant targets are the reached
poses, and upstream's converter labels frame t with frame t's own pose — so a
chunk entered at row 0 first tells the arm to stay put; upstream never does
(its fixed `inf_delay` skips the first rows) and neither does this loop once it
has a plan. And upstream evaluates under `osc_pose_SAIL.json`, kp 600, where
the arm reaches each target and EAG's 2 cm bound is met; at the law's default
150 the arm trails a 20 Hz plan by ~2 cm and guidance is dropped on nearly every
inference. On the arm SAIL runs at kp 300 and damping ratio 0.5
(`baselines.sail.osc_kp`, `osc_damping_ratio`), sent through the kp and kd
action channels; the manifest records what ran. The sim rollout has its own
setting, `baselines.sail.sim_osc` (LIBERO_SIM.md, "SAIL's controller").

**B-Spline** predicts spline parameters and evaluates them at wall-clock `t`, so
`--speed-up-times 2.0` is a change of variable and nothing else. `t` advances at
`speed_up_times · origin_time_scale` per second.

A sped-up plan needs the goal led forward. The OSC trails a goal moving at
velocity `v` by `(kd/kp)·v`, so at `speed_up_times` s the arm lags s times
further behind than it did in the demonstrations. Every replan starts from the
observed pose, so that extra lag came back as a backward jump in the goal on
up to 0.9 of replans at 4x. Upstream avoids this by speeding its servo up with
the plan (`set_ik_dt_scale`). Here the gains stay as they are; instead each
sample is moved `(kd/kp)·(1 - 1/s)` seconds further along the plan's own
velocity (`rollout_common.damping_lag`, `policy_math.lead_goal`), which puts the
lag back to the demonstrations' 1x lag. At 1x the lead is zero.

The B-Spline server runs inference the way upstream deploys it: 10 DDIM steps
(`--num-inference-steps`), the whole denoising loop replayed as one CUDA graph
(upstream's `CudaGraphDDIMSampler`), and one warm-up at startup instead of one
inside every episode.

## Things that fail silently

Read this list before debugging a rollout that "just tracks badly". Each is
covered by a check in scripts/check_baseline_rollout_offline.py.

1. **The observation pose is `O_T_EE`.** The converters build every training
   observation from `ee_kinematics.eef_poses_from_qpos`, and the rollout uses the
   same function. `residual_wrapper`'s `current_ee_pose` looks like the obvious
   helper and applies a *different* correction (grip-site position, hand-body
   orientation) for sim-trained students; using it here puts both baselines 45
   degrees off distribution.
2. **`origin_time_scale` is the RECORDING rate, not upstream's 10.0.** The
   spline's knots count frames and `t` advances in knot-index units per second.
   Upstream's default is their own 10 Hz; against our 20 Hz data every plan plays
   at **0.5x** — twice the wall clock — which reads as a sluggish controller
   rather than a misconfiguration. It defaults to `control_fps()` here.
3. **SAIL's control mode comes from the checkpoint.** `train.action_keys` of
   `actions` means deltas (EE_DELTA); `absolute_actions*` means poses (EE_POS).
   The training template and the shipped guide template disagree about which SAIL
   uses, so nothing but the checkpoint can settle it. `--control-mode` overrides.
4. **Ours takes an EE_POS base and dispatches EE_DELTA, like the reach path.**
   `run_residual.py` reads the checkpoint's own unnormaliser stats and refuses a
   base whose training actions were per-step deltas. The base's absolute poses
   are taken relative to the pose it planned from, the residual is summed in
   that normalised space (FAST's `clip(base + residual, ±chunk length)`), and
   every resulting target is executed as the one-step delta from the pose
   measured at that step -- the two stages of multi-fast's ActionChunkWrapper
   in target mode, shared with `reach_residual.py`. The manifest records
   `base_action_space: EE_POS` next to `control_mode: EE_DELTA` for that reason.
5. **The precision label must be stripped before dispatch, and only if present.**
   It is the last column. Strip it when it is not there and the gripper channel
   is amputated instead; leave it when it is and the label is commanded as a
   gripper position. The server derives this and refuses to claim a label the
   action width cannot hold.
6. **`EE_POS` has no delta envelope, and that is faithful to osc.py.** A sped-up
   plan leads the arm on purpose. `baselines.exec.max_lead_m` therefore **aborts
   the episode**; it is not a clamp, because clamping would be a third limit
   layer and would eat exactly the lead the method needs. If it fires, lower
   `--speed-up-times` or `--exec-fps` — do not raise the bound to get past it.
7. **`obs_stride` is 1 here, not upstream's 20.** Theirs is calibrated for a
   10 Hz data / 200 Hz control split; 20 would buffer 40 observations before the
   first plan.
8. **Camera frames are resized to what the checkpoint declares.** SAIL's HDF5
   carries full-resolution frames while its config expects 84x84, and nothing
   else in the stack would notice the mismatch. Both servers report their shapes
   in the `meta` handshake.
9. **A run is filed by its TRAINING dataset, not its rollout.** The directory
   name comes from the demonstrations the policy learned from, so every method
   trained on one task lands together. A checkpoint converted before the stamp
   existed cannot say which that was, and the run stops and asks for
   `--train-dataset` rather than guessing.
10. **The two single-arm rigs expose different cameras** — `single_arm_franka` has
   cam_1/cam_5/cam_2, `single_arm_right` has cam_3/cam_4/cam_2 — so a checkpoint
   trained on one names keys the other does not have. The rollout refuses that
   before homing rather than letting it surface as a `KeyError` inside the policy
   server mid-episode. `--allow-missing-cameras` sends blank frames instead and
   accepts that the policy is off-distribution.
11. **A recorded dataset is labelled at the DISPATCH rate**, one frame per goal,
   not at `obs_fps`. For SAIL that is the nominal fast rate; the real per-step
   rate varies, and `slow_steps` says how often it dropped.
12. **SAIL's policy wants what its simulator env used to hand it.** With
   `train.frame_stack: 2` (the template) every observation key must arrive as a
   `[T, ...]` stack, seeded with copies of the first frame and then fed one frame
   per *step* — the client sends the frame taken one dispatch before each
   observation alongside it, so the stack is not one inference apart; images must already
   be CHW float in [0, 1]; and `get_action` returns ONE action and then serves
   an internal queue unless asked for the sequence. The server does all three
   (`sail_bridge/policy_server.py`), and `check_policy_server.py` is where a
   regression shows up as a shape error rather than as a policy that "does
   nothing".

Every knob above lives in `config/policy.yaml` under `baselines:` and nowhere
else. `python -m franka_config get policy.baselines.exec.fast_fps` prints what
the stack will actually use.

## Checking it without the robot

`scripts/check_baseline_rollout_offline.py` runs both loops end to end against a
fake arm and fake policy servers — no hardware, no baseline venv:

```bash
python scripts/check_baseline_rollout_offline.py
python scripts/check_baseline_rollout_offline.py --only sail
```

The fake arm holds a real joint configuration and responds to a goal with a
damped-Jacobian step toward it using this repo's own `zero_jacobian`, so the
`q -> FK pose` relationship the rollout reads back is self-consistent — which is
what makes the lead monitor and the tracking-error check testable at all. The
servers are real ZMQ peers answering with synthetic chunks and splines, so the
client, the pickling and the handshake are the real ones.

It covers the ported helpers against upstream's own source, the control-mode
resolution for all three action keys, the receding-horizon index bookkeeping, the
precision strip, the rate switching, the spline's wall-clock timing at three
`(speed_up, origin_time_scale)` combinations, the abort-not-clamp behaviour, and
the recorded dataset's schema and fps.

Its `[servers]` section also drives the real servers' request handling under a
stubbed robomimic: frame stacking, image processing, key filtering,
`return_action_sequence`, and the checkpoint-config reads.

What it cannot tell you: whether the arm tracks, and whether a real checkpoint's
actions are sane — `check_policy_server.py` covers the second.

## First run on hardware

Operator on the e-stop. Preflight per `RIGHT_ARM_RIG_HANDOFF.md` (want
`RobotMode.Idle []`) and `scripts/check_policy_server.py` against the server,
then walk up:

```bash
# 1. handshake and homing only, no motion
./scripts/sail_rollout.sh --start-server --ckpt <CKPT> --rig=single_arm_right \
    --dry-run --num-episodes 1

# 2. one episode with the speed features off
./scripts/sail_rollout.sh ... --exec-fps 20 --no-precision --no-eag \
    --num-episodes 1 --episode-time-s 30

# 3. raise --exec-fps, then re-enable precision, then EAG -- one at a time
# 4. B-spline at 1.0x before anything faster
./scripts/bspline_rollout.sh ... --speed-up-times 1.0 --num-episodes 1
```

Watch `send-gap max` in the loop log — a spike there is how long the OSC loop sat
on one goal, and it is the visible hitch — and
`control_command_success_rate` on the NUC. Nothing here changes `torque:`, so no
`deploy_nuc_server.sh` is needed.
