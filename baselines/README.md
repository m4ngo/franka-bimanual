# Baselines: SAIL and B-Spline Policy

We want to show our method finishes tasks faster than other methods, without
dropping the success rate. To do that fairly, all three methods have to learn
from the *same* demonstrations and be tested on the *same* task.

This directory is the glue that makes that possible. You record once, and it
turns that one recording into the three different file formats the three
methods each expect.

## The idea in one picture

```
   one teleop recording (EE_POS)
              |
   scripts/prepare_baseline_datasets.py
              |
    +---------+---------+
    |         |         |
 sysid.hdf5  sail.hdf5  bspline.hdf5
    |         |         |
 our method  SAIL     B-Spline
    |         |         |
    +---------+---------+
              |
    run each on the robot, record
    success + time per episode
```

## How to run it

Record one dataset the normal way (`scripts/ee_record_data.sh`), then:

```bash
python scripts/prepare_baseline_datasets.py \
    --source-repo-id my-recording \
    --out-dir ~/franka_data/baseline_prep/my-recording
```

That writes `sysid.hdf5`, `sail.hdf5` and `bspline.hdf5` into the output
directory. It only reads files, never touches the robot, so it is safe to run
any time after recording.

Useful flags:

| Flag | Why you'd use it |
|---|---|
| `--episodes 0,1,2` | Convert only a few episodes, for a quick check |
| `--no-images` | Skip camera frames. Much faster; use when checking shapes |
| `--skip sail bspline` | Only build the artifacts you need right now |
| `--bspline-image-size 84x84` | Shrink camera frames for B-Spline training |

You can also run any one converter on its own:

```bash
python -m baselines.sail_bridge.dataset my-recording --out sail.hdf5
python -m baselines.bspline_bridge.dataset my-recording --out bspline.hdf5
```

## Which files matter

Read them in this order:

1. **`scripts/prepare_baseline_datasets.py`** — the top level. Short. It just
   calls the three converters in turn. Start here.
2. **`baselines/common.py`** — the shared work: open the recording, walk it
   episode by episode, work out where the arm actually was and where it was
   told to go, attach camera frames, write the file.
3. **`baselines/sail_bridge/dataset.py`** and
   **`baselines/bspline_bridge/dataset.py`** — one short function each. They
   only decide which numbers get which names in the output file. All the
   shared machinery lives in `common.py`.

Two helpers were pulled out of the sysid converter so all three converters can
share them:

- `lerobot_robot_bimanual_franka/ee_kinematics.py` — turns joint angles into a
  gripper position and orientation.
- `lerobot_robot_bimanual_franka/lerobot_source.py` — opens a recording and
  checks it is the kind we expect.

The two upstream projects are git submodules (`sail/`, `bspline_policy/`). We
don't edit them. Each needs its own conda environment, and they conflict with
each other and with ours, so training and rollout run by calling out to those
environments rather than importing them.

For a map of either upstream repo — its entry points, useful scripts, and how
it is put together — read [SAIL.md](SAIL.md) and
[BSPLINE_POLICY.md](BSPLINE_POLICY.md).

## Two numbers, and why they're different

Every converter works out two things for each moment in the recording:

- **where the arm actually was** ("reached") — computed from the recorded joint
  angles.
- **where the arm was told to go** ("commanded") — the recorded target.

These are never the same, because a real arm lags behind its target. That gap
is the whole point: a policy learns to predict the target from what it sees, so
the observation must be the *actual* position and the training target must be
the *commanded* one. Swapping them gives the policy its own answer as input,
and it learns nothing.

## What each output file contains

`sail.hdf5`, one entry per episode:

| Name | What it holds |
|---|---|
| `obs/robot0_eef_pos`, `obs/robot0_eef_quat` | where the gripper actually was |
| `obs/robot0_joint_pos`, `obs/robot0_gripper_qpos` | recorded joint angles and grip |
| `obs/<cam>_image` | camera frames |
| `absolute_actions` | the reached position, as position + rotation + grip |
| `commanded_absolute_actions` | the commanded position, same layout |
| `actions` | the step as a *change* rather than a position |

`bspline.hdf5` is simpler, matching what that project's own converter writes:
`obs/arm_pos`, `obs/arm_quat`, `obs/gripper_pos`, `obs/<cam>_image`, and
`actions`.

`sysid.hdf5` is unchanged from before; it is what our own method already used.

## Notes for whoever picks this up next

- **The recording must be in EE_POS.** The converters check this and refuse
  otherwise. Our controller and our simulation work in absolute positions, so
  that is the honest format to record in. SAIL is the only one that wants
  changes rather than positions, so we compute those from the recording.
- **SAIL normally replays demonstrations through a simulator** to work out
  where the arm ended up. We skip that: a real recording already knows where
  the arm was, and a real measurement beats a simulated one. So SAIL's
  `add_all_actions.py` is not used. Its later steps
  (`save_awe_waypoint_concurrent.py`, `label_awe_trajectory_precision.py`) run
  against our file unchanged.
- **robomimic requires two pieces of bookkeeping** to open a file at all: a
  sample count on each episode, and an environment description on the file. We
  write both. Without them, training stops before it starts.
- The first few frames of every episode are dropped, because each episode
  begins with the arm settling into its start position and that movement is not
  part of the demonstration.

## Running a trained policy on the arm

That half is built: see [ROLLOUT.md](ROLLOUT.md).

```bash
./scripts/sail_rollout.sh    --start-server --ckpt <CKPT> --rig=single_arm_right
./scripts/bspline_rollout.sh --start-server --ckpt <CKPT> --rig=single_arm_right
```

Each starts a policy server in the upstream conda env and a rollout client in
our venv, because the environments conflict. Success is marked by the operator
(right arrow = success, left = failure), and each run writes a `metrics.json`
with success and time per episode plus, optionally, a LeRobotDataset.

`python scripts/check_baseline_rollout_offline.py` exercises both loops against
a fake arm and fake policy servers, so the plumbing can be checked without
hardware or either conda env.

## Not built yet

Training wrappers. Both upstream training paths still run by hand in their own
conda environments, as [SAIL.md](SAIL.md) and
[BSPLINE_POLICY.md](BSPLINE_POLICY.md) describe.
