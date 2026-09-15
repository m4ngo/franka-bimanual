# Baselines: SAIL and B-Spline Policy

We want to show our method finishes tasks faster than other methods without
dropping the success rate. For that to be fair, every method has to learn from
the *same* demonstrations and be tested on the *same* task, on the same arm.

This directory is the glue: one recording goes in, and out come trained SAIL
and B-Spline policies, their rollouts on the FR3, and a table comparing them
with ours. Follow the steps in order.

```
   record once (EE_POS)
        |
   1. convert      scripts/prepare_baseline_datasets.py
        |                 -> sysid.hdf5   sail.hdf5   bspline.hdf5
   2. train        python -m baselines.sail_bridge.train     sail.hdf5
                   python -m baselines.bspline_bridge.train  bspline.hdf5
        |                 -> ~/franka_data/policies/<dataset>/{sail,bspline}/<run>/
   3. roll out     ./scripts/sail_rollout.sh      --start-server --ckpt ...
                   ./scripts/bspline_rollout.sh   --start-server --ckpt ...
                   python residual_wrapper/run_residual.py ...   (ours)
        |                 -> ~/franka_data/outputs/<dataset>/<timestamp>-<method>/
   4. compare      python scripts/rollout_summary.py <dataset>
```

Everything below assumes the workspace venv is active (`~/franka_ws/.venv`).

## 0. One-time setup

The two upstream projects are git submodules (`sail/`, `bspline_policy/`) and
each needs its own interpreter -- they conflict with each other and with the
workspace venv. Build both once:

```bash
./scripts/setup_baseline_envs.sh          # or: sail | bspline
```

That creates `.venv-sail` and `.venv-bspline` next to `.venv` and checks each
can see the GPU. It is *not* the upstream conda recipe: those pin a torch with
no kernels for the RTX 5090, so this uses a current torch and otherwise follows
their package lists. Every script that needs one of these interpreters finds it
through `baselines/interpreters.py`; `$SAIL_PYTHON` / `$BSPLINE_PYTHON`
override it.

If the submodules are empty: `git submodule update --init baselines/sail
baselines/bspline_policy`.

## 1. Record and convert

Record the demonstrations in **EE_POS** with the normal single-arm tooling --
`scripts/single_arm_record_data_homed.sh` with the `spacemouse_ee` or
`gello_ee` mode. The converters refuse anything else: both baselines train on
absolute poses, and a delta recording cannot be turned into one after the fact.

```bash
python scripts/prepare_baseline_datasets.py \
    --source-repo-id pickup-bowl \
    --out-dir ~/franka_data/baseline_prep/pickup-bowl
```

This reads the recording (never the arm) and writes `sysid.hdf5`, `sail.hdf5`
and `bspline.hdf5`. The dataset id is stamped onto each file so everything
downstream -- checkpoints, servers, rollouts -- knows which task it belongs to
without you retyping it.

| Flag | Why you'd use it |
|---|---|
| `--image-size 84x84` | Shrink camera frames in both files. The file's size is what the policies train and roll out at; the default keeps the recording's resolution |
| `--episodes 0,1,2` | Convert only a few episodes, for a quick check |
| `--no-images` | Skip camera frames. Fast; for checking shapes only |
| `--skip sysid` | Only build what you need right now |

## 2. Train

Each trainer runs upstream's own training in its own venv, with a config
generated from the file (camera names, image size, dataset path, output
directory). Nothing is edited inside the submodules.

```bash
python -m baselines.sail_bridge.train    ~/franka_data/baseline_prep/pickup-bowl/sail.hdf5
python -m baselines.bspline_bridge.train ~/franka_data/baseline_prep/pickup-bowl/bspline.hdf5
```

SAIL's trainer first runs its two labelling passes on the file (AWE waypoints,
then precision labels; skipped when already present, `--relabel` redoes them)
and prints the fraction of steps labelled precise per demo -- if that is near
0% or 100%, tune `--err-threshold` before spending a training run.

Checkpoints land under `~/franka_data/policies/<dataset>/`:

```
policies/pickup-bowl/
  sail/<timestamp>/models/model_epoch_N.pth
  bspline/<timestamp>/checkpoints/latest.ckpt
```

Both trainers print the checkpoint path and the rollout command when they
finish. Upstream's defaults are long (SAIL 1000 epochs, B-Spline 601); to make
sure the pipeline runs before committing the time:

```bash
python -m baselines.sail_bridge.train    <sail.hdf5>    --epochs 2 --epoch-every-n-steps 10 --save-every 1
python -m baselines.bspline_bridge.train <bspline.hdf5> --epochs 2 --checkpoint-every 1
```

`--dry-run` on either prints the generated config and the exact command without
running anything.

## 3. Roll out

Each rollout is two processes: the policy server in its venv, the arm in ours.
The wrapper starts both and stops the server when it exits:

```bash
./scripts/sail_rollout.sh --start-server --rig=single_arm_right --num-episodes 10 \
    --ckpt ~/franka_data/policies/pickup-bowl/sail/<ts>/models/model_epoch_1000.pth \
    --guide-config baselines/sail/robomimic/SAIL/guide_template/base_cfg_weight_1.json

./scripts/bspline_rollout.sh --start-server --rig=single_arm_right --num-episodes 10 \
    --ckpt ~/franka_data/policies/pickup-bowl/bspline/<ts>/checkpoints/latest.ckpt \
    --speed-up-times 1.0
```

Before the first rollout of any new checkpoint, prove the server answers
sanely without touching the arm -- start it yourself, then:

```bash
.venv-sail/bin/python baselines/sail_bridge/policy_server.py --ckpt-path <pth> --port 5556 &
python scripts/check_policy_server.py sail --port 5556
```

(`bspline` likewise on 5555.) It prints the checkpoint's handshake and checks
one inference comes back with the right shape. Then read
[ROLLOUT.md](ROLLOUT.md) for the walk-up on hardware: `--dry-run` first, then
one slow episode, then the speed features one at a time.

During an episode **right arrow = success, left arrow = failure**, timeout is a
failure, Ctrl-C aborts the run. That verdict is the measurement.

Every run writes one directory under `~/franka_data/outputs/`, grouped by the
dataset the policy was **trained** on, so all three methods for one task sit
side by side:

```
outputs/pickup-bowl/
  20260912_143000-sail/        manifest.json  episodes.jsonl  dataset/  videos/
  20260912_151500-bspline/
  20260912_160200-multifast/
```

The manifest records everything about the run (checkpoint and its hash, every
parameter, the rig, the git state); `episodes.jsonl` has one line per episode.
The run works out which task it belongs to from the checkpoint; pass
`--train-dataset` only when a checkpoint predates the stamp.

## 4. Compare

```bash
python scripts/rollout_summary.py pickup-bowl
```

```
  run                        method      eps   ok   rate  median s   mean s  status       notes
  20260912_143000-sail       sail          4    3    75%      9.10     9.23  completed    100Hz precision eag
  20260912_151500-bspline    bspline       4    3    75%      7.40     7.37  completed    100Hz 2.0x
  20260912_160200-multifast  multifast     4    4   100%      6.10     6.00  completed    20Hz
```

Time-to-success is over successes only. `--all` sweeps every task, `--json`
dumps the manifests.

## Checking without the robot

- `python scripts/check_baseline_rollout_offline.py` runs both rollout loops
  against a fake arm and fake policy servers, plus the servers' own request
  handling under a stubbed robomimic. No hardware, no baseline venv.
- `python scripts/check_policy_server.py <sail|bspline>` checks a *real* server
  with a *real* checkpoint (above).
- The whole chain was last exercised end to end on `pipeline-test-9-12`
  (3 episodes): convert, label, 2-epoch training of both, both servers passing
  the preflight.

## Where the files are

| What | Where |
|---|---|
| Recordings | `~/franka_data/<dataset>/` |
| Converted HDF5s | `~/franka_data/baseline_prep/<dataset>/` |
| Trained policies | `~/franka_data/policies/<dataset>/<method>/<run>/` |
| Rollouts | `~/franka_data/outputs/<dataset>/<timestamp>-<method>/` |
| Baseline venvs | `~/franka_ws/.venv-sail`, `~/franka_ws/.venv-bspline` |
| The knobs | `config/policy.yaml`, `baselines:` block |

Nothing is written into the repo.

## Details, for whoever changes this next

**Which files matter.** `common.py` does the shared conversion work (open the
recording, walk it episode by episode, work out where the arm was and where it
was told to go, attach frames, write the file); `sail_bridge/dataset.py` and
`bspline_bridge/dataset.py` only decide which numbers get which names.
`sail_bridge/train.py` and `bspline_bridge/train.py` generate the upstream
configs and call upstream's trainers; `*/policy_server.py` serve a checkpoint
over ZMQ; `*/rollout.py` drive the arm; `rollout_common.py` and
`run_record.py` are what the two rollouts (and `run_residual.py`) share.

**Two numbers per step, and why they differ.** Every converter records where the
arm actually *was* (from the joint angles, via `ee_kinematics`) and where it was
*told to go* (the recorded target). A real arm lags its target, so the
observation is the reached pose and B-Spline's training target is the
commanded one; giving a policy its own answer as input teaches it nothing.
SAIL is the exception by its own design: it trains on the reached poses
(`absolute_actions`, as upstream's replay pass produces) with the precision
label appended, and `commanded_absolute_actions` is written alongside for
`--action-key` to pick instead.

**What each file holds.** `sail.hdf5`: `obs/robot0_eef_{pos,quat}`,
`obs/robot0_joint_pos`, `obs/robot0_gripper_qpos`, `obs/<cam>_image`,
`actions` (deltas), `absolute_actions`, `commanded_absolute_actions`, and after
labelling `waypoints_dp`, `precisions`, `absolute_actions_with_precision`.
`bspline.hdf5`: `obs/arm_pos`, `obs/arm_quat`, `obs/gripper_pos`,
`obs/<cam>_image`, `actions` (commanded pose, 7-dim; the loader makes it
10-dim rot6d). Both carry `source_repo_id` on the root. robomimic also needs a
`num_samples` attribute per demo and `env_args` on `data`; both are written,
with env type 6 (real) so training never tries to build a simulator.

**SAIL's replay pass is skipped.** Upstream's `add_all_actions.py` replays each
demo in a simulator to discover where the arm ended up; a real recording
already knows. Its later passes run on our file unchanged, from
`sail_bridge/train.py`.

**The first few frames of every episode are dropped** -- the arm settling into
its start pose is not part of the demonstration.

Maps of the two upstream repos: [SAIL.md](SAIL.md), [BSPLINE_POLICY.md](BSPLINE_POLICY.md).
Everything about running on the arm: [ROLLOUT.md](ROLLOUT.md).
