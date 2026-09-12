# Running a trained baseline on the arm

The converters in this directory turn one recording into the three formats the
three methods train on. This is the other end: running each trained policy on
the real FR3 and writing down success and time, so the comparison the directory
exists for can actually be scored.

## Why there are two processes

Each upstream project needs its own conda environment, and they conflict with
each other and with the workspace venv that owns `lerobot`, `franka_config` and
the RPyC link to the arm. So the policy never runs in the same process as the
robot:

```
 conda env (SAIL / robodiff)            workspace venv (~/franka_ws/.venv)
 ---------------------------            ---------------------------------
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
    --ckpt ~/franka_data/sail/model_epoch_300.pth \
    --guide-config baselines/sail/robomimic/SAIL/guide_template/base_cfg_weight_1.json \
    --rig=single_arm_right --num-episodes 10 \
    --repo-id you/sail-eval --output-dir ~/franka_data/sail-eval

./scripts/bspline_rollout.sh --start-server \
    --ckpt ~/franka_data/bsp/latest.ckpt \
    --rig=single_arm_right --speed-up-times 1.0 --num-episodes 10 \
    --repo-id you/bsp-eval --output-dir ~/franka_data/bsp-eval
```

Each wrapper reaps the policy server's whole process group on exit, including a
Ctrl-C. That matters: `conda run` spawns python as a child, and an orphaned
server keeps the port — the next run would then handshake with the stale one and
silently evaluate the previous checkpoint under the new one's recorded sha256.

To run the halves separately, start the server yourself and drop
`--start-server`:

```bash
conda activate SAIL
python baselines/sail_bridge/policy_server.py --ckpt-path <CKPT> --port 5556

conda activate robodiff
python baselines/bspline_bridge/policy_server.py --ckpt-path <CKPT> --port 5555
```

The B-Spline server subclasses upstream's own `policy_server_bspline.py` to add
one request (`meta`) and changes nothing else — upstream replies `{}` to a key it
does not know, so it cannot tell the client the image sizes or the spline degree,
and both are load-bearing. Neither submodule is edited.

During an episode: **right arrow** ends it as a success, **left arrow** as a
failure, a timeout counts as a failure, and Ctrl-C aborts the run. That verdict
is the measurement — there is no automatic success detector on this rig.

## What comes out

A `metrics.json` next to the dataset (or under `~/franka_data/baseline_eval/`),
rewritten after every episode so an interrupted run keeps what finished:

```json
{"policy": "sail", "ckpt_sha256": "...", "rig": "single_arm_right",
 "control_mode": "EE_POS", "exec_fps": 100, "obs_fps": 20,
 "summary": {"episodes": 10, "successes": 7, "success_rate": 0.7,
             "mean_time_to_success_s": 8.4},
 "episodes": [{"episode": 0, "success": true, "wall_time_s": 8.42, "steps": 842,
               "inferences": 53, "slow_steps": 120, "guided_inferences": 41,
               "aborted": null, "max_lead_m": 0.031}]}
```

Plus, with `--repo-id`, a LeRobotDataset carrying the same `observation.state` /
`action` features a `run_residual.py` recording does, so all three methods'
rollouts are readable by one set of tools.

## The two speed mechanisms

Both baselines claim to finish faster than the demonstrations, and both need the
goal-push rate decoupled from the camera rate to show it. Observations and
inference run at `obs_fps` (20, the rate everything else on this rig uses); OSC
goals go out at `--exec-fps`. The NUC's torque law runs at 500 Hz regardless, so
this costs nothing there.

**SAIL** modulates speed per step. The last action column is a precision label;
if any label inside a window around the current step is set, that step is
dispatched at `--slow-fps` instead of `--exec-fps`. It also runs a receding
horizon — `inf_delay` steps of the previous prediction, then `execute_n_actions`
of the new one — and error-adaptive guidance, which conditions the next
prediction on the tail of the current plan but only while the arm is tracking it.
All three are on by default when the checkpoint supports them; `--no-precision`
and `--no-eag` turn the last two off.

**B-Spline** predicts spline parameters and evaluates them at wall-clock `t`, so
`--speed-up-times 2.0` is a change of variable and nothing else. `t` advances at
`speed_up_times · origin_time_scale` per second.

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
4. **The precision label must be stripped before dispatch, and only if present.**
   It is the last column. Strip it when it is not there and the gripper channel
   is amputated instead; leave it when it is and the label is commanded as a
   gripper position. The server derives this and refuses to claim a label the
   action width cannot hold.
5. **`EE_POS` has no delta envelope, and that is faithful to osc.py.** A sped-up
   plan leads the arm on purpose. `baselines.exec.max_lead_m` therefore **aborts
   the episode**; it is not a clamp, because clamping would be a third limit
   layer and would eat exactly the lead the method needs. If it fires, lower
   `--speed-up-times` or `--exec-fps` — do not raise the bound to get past it.
6. **`obs_stride` is 1 here, not upstream's 20.** Theirs is calibrated for a
   10 Hz data / 200 Hz control split; 20 would buffer 40 observations before the
   first plan.
7. **Camera frames are resized to what the checkpoint declares.** SAIL's HDF5
   carries full-resolution frames while its config expects 84x84, and nothing
   else in the stack would notice the mismatch. Both servers report their shapes
   in the `meta` handshake.
8. **The two single-arm rigs expose different cameras** — `single_arm_franka` has
   cam_1/cam_5/cam_2, `single_arm_right` has cam_3/cam_4/cam_2 — so a checkpoint
   trained on one names keys the other does not have. The rollout refuses that
   before homing rather than letting it surface as a `KeyError` inside the policy
   server mid-episode. `--allow-missing-cameras` sends blank frames instead and
   accepts that the policy is off-distribution.
9. **A recorded dataset is labelled at the DISPATCH rate**, one frame per goal,
   not at `obs_fps`. For SAIL that is the nominal fast rate; the real per-step
   rate varies, and `slow_steps` says how often it dropped.

Every knob above lives in `config/policy.yaml` under `baselines:` and nowhere
else. `python -m franka_config get policy.baselines.exec.fast_fps` prints what
the stack will actually use.

## Checking it without the robot

`scripts/check_baseline_rollout_offline.py` runs both loops end to end against a
fake arm and fake policy servers — no hardware, no conda env:

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

What it cannot tell you: whether the arm tracks, and whether a real checkpoint's
actions are sane.

## First run on hardware

Operator on the e-stop. Preflight per `RIGHT_ARM_RIG_HANDOFF.md` (want
`RobotMode.Idle []`), then walk up:

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
