# SAIL and B-Spline on LIBERO

The sim half of the three-way comparison we run on the FR3. Same env, same init
states, same clock for all three:

- **SAIL**, **B-Spline** -- trained here from LIBERO demos.
- **pi05** -- multi-fast's base policy, pretrained. Not multi-fast: no FAST
  residual exists for these tasks, so it files under its own name.

`baselines/README.md` is the hardware side; `baselines/ROLLOUT.md` covers run
directories.

## Setup, once

```bash
source ~/franka_ws/.venv/bin/activate
./scripts/setup_baseline_envs.sh          # builds .venv-sail, .venv-bspline
```

Idempotent. `multi-fast/.venv` must already exist -- it is the only interpreter
with robosuite, libero and openpi.

## Run everything

```bash
# 1. LIBERO demos -> sail.hdf5 + bspline.hdf5 per task  (pi05 needs none of this)
python -m baselines.libero_bridge.dataset ~/libero_data/libero_90 \
    --out-dir ~/franka_data/baseline_prep/libero_90

# 2. one policy per task per baseline
python -m baselines.libero_bridge.train ~/franka_data/baseline_prep/libero_90 \
    --tasks-file baselines/libero_bridge/teacher_tasks.txt --steps 100000

# 3. roll all three out; prints a pooled table at the end
python -m baselines.libero_bridge.evaluate ~/franka_data/baseline_prep/libero_90 \
    --tasks-file baselines/libero_bridge/teacher_tasks.txt --num-episodes 20 \
    --save-video
```

On one 5090: conversion ~15 s/task. Training dominates -- SAIL 107-252 min/task
(median ~190), B-Spline 92-100 min. Rollout is cheap, median 1.4 min per
(task, method) for 20 episodes, so the full three-way sweep is ~35 min.

## Tasks are numbered

A task is named by its index in its suite, in LIBERO's own order
([`task_names`](libero_bridge/sim_env.py#L98)): task 2 of libero_10 is
`task_2`. That names its prep directory (`baseline_prep/libero_10/task_2/`), the
id stamped on its files (`libero_10/task_2`), and so where its policies and
rollouts go. Every task flag takes the index, the tag or the full LIBERO name:
`--tasks 0 2 9`, `--task task_2`.

The converter reads the suite from the source path -- the nearest directory named
after one, so `libero_10/libero_10_relabeled/` is libero_10 -- or from `--suite`.
An index means a different task in every suite, so it refuses to write over a
`task_<i>/` stamped with another suite, and `evaluate` takes the suite from the
prep directory's files rather than a default.

libero_90 was converted before this, so its directories, policies and rollouts
keep the full task names. Indices find them too
([`task_dirs`](libero_bridge/train.py#L126)); `teacher_tasks.txt` is indices.

## One task

`evaluate` is a loop over `scripts/libero_rollout.sh`. Call that directly for a
single run; [`checkpoint_for`](libero_bridge/train.py#L59) finds the checkpoint.

```bash
PREP=~/franka_data/baseline_prep/libero_90
TASK=44                                     # KITCHEN_SCENE9_turn_on_the_stove
CKPT=$(python -c "
from pathlib import Path
from baselines.libero_bridge.train import checkpoint_for, task_dirs
d, = task_dirs(Path('$PREP'), ['$TASK'])
print(checkpoint_for(d / 'bspline.hdf5', 'bspline'))")

./scripts/libero_rollout.sh --backend bspline --start-server \
    --ckpt "$CKPT" --task "$TASK" --num-episodes 20 --save-video

./scripts/libero_rollout.sh --backend pi05 --task "$TASK" --num-episodes 20
```

`--suite` defaults to libero_90; pass it for any other suite. `pi05` takes no
`--ckpt` and no `--start-server`, and files under `<suite>/task_<i>`; for
libero_90's name-tagged tasks add `--train-dataset libero_90/<name>` to land
beside the baselines (`evaluate` does this itself). On disk the checkpoints are
`<task>/bspline/<run>/checkpoints/latest.ckpt` and
`<task>/sail/<run>/models/model_epoch_N.pth` under
`~/franka_data/policies/<suite>/`.

Unconsumed flags pass through to `baselines.libero_bridge.rollout`. For its
`--help`, run it under `multi-fast/.venv/bin/python` -- nothing else can import
it.

## Training on Tillicum

Locally the sweep trains one policy at a time: ~6 h per libero_10 task at 200k
steps. On Tillicum every task and backend is its own Slurm job on its own GPU,
all running at once, so a sweep takes about as long as its slowest training.
The trainers are the same; only the launch differs. The files are in
`baselines/tillicum/` ([Files](#files)).

Once per image. On the workstation (~15 min cold, 5 GB pushed):

```bash
cd ~/franka_ws
TAG=$(git describe --always --dirty)
docker build -f baselines/tillicum/Dockerfile -t atsai06/franka-baselines:$TAG .
docker tag atsai06/franka-baselines:$TAG atsai06/franka-baselines:latest
docker push atsai06/franka-baselines:$TAG && docker push atsai06/franka-baselines:latest
rsync -a --mkpath baselines/tillicum/ atsai06@tillicum.hyak.uw.edu:/gpfs/scrubbed/atsai06/tillicum/
```

Then on Tillicum, on a compute node (not the login node):

```bash
ssh atsai06@tillicum.hyak.uw.edu
salloc --qos=debug --gpus=1 --cpus-per-task=8 --mem=64G --time=00:30:00
# layer cache and build scratch off the small home quota; the image unpacks to 15 GB
export APPTAINER_CACHEDIR=/gpfs/scrubbed/$USER/apptainer_cache APPTAINER_TMPDIR=/gpfs/scrubbed/$USER/tmp
mkdir -p /gpfs/scrubbed/$USER/containers $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
apptainer pull --force /gpfs/scrubbed/$USER/containers/franka-baselines.sif docker://atsai06/franka-baselines:latest
exit
```

Once per sweep. Send the converted tasks up (HDF5s only, 8.2 GB for libero_10).
`--info=progress2` shows overall progress; without it a large rsync prints
nothing until it is done. `--partial` keeps a half-sent file, so running the
same command again picks up where it stopped:

```bash
rsync -a --mkpath --partial --info=progress2 --include='task_*/' --include='*.hdf5' --exclude='*' \
    ~/franka_data/baseline_prep/libero_10/ \
    atsai06@tillicum.hyak.uw.edu:/gpfs/scrubbed/atsai06/franka_home/franka_data/baseline_prep/libero_10/
```

On the Tillicum login node, dry run, then submit. `3:bspline` is one backend of
a task; task_3's SAIL finished locally.

```bash
bash /gpfs/scrubbed/$USER/tillicum/submit_libero_train.sh libero_10 --tasks "3:bspline 4 5 6 7 8 9" --steps 200000
bash /gpfs/scrubbed/$USER/tillicum/submit_libero_train.sh libero_10 --tasks "3:bspline 4 5 6 7 8 9" --steps 200000 --apply

squeue -u $USER                                          # R running, PD waiting for a GPU
bash /gpfs/scrubbed/$USER/tillicum/progress.sh libero_10  # epoch, rate and time left per unit
sacct -X --name=libero_10 -S now-2days --format=JobID%18,State,Elapsed,NodeList
tail -f /gpfs/scrubbed/$USER/franka_home/franka_data/baseline_prep/libero_10/train_logs/task_4.sail.log
```

Back on the workstation, fetch the results, then [evaluate](#run-everything) as usual:

```bash
rsync -a --partial --info=progress2 --exclude='epoch=*.ckpt' \
    atsai06@tillicum.hyak.uw.edu:/gpfs/scrubbed/atsai06/franka_home/franka_data/policies/libero_10/ \
    ~/franka_data/policies/libero_10/
rsync -a --include='task_*/' --include='sail.hdf5' --include='train_logs/***' --include='slurm_logs/***' \
    --exclude='*' \
    atsai06@tillicum.hyak.uw.edu:/gpfs/scrubbed/atsai06/franka_home/franka_data/baseline_prep/libero_10/ \
    ~/franka_data/baseline_prep/libero_10/
```

The first rsync leaves out B-Spline's top-k `epoch=*.ckpt` (~14 of its 17 GB per
task); rollout reads `latest.ckpt`. The second brings back the logs and
`sail.hdf5`, which SAIL's AWE pass labelled on the cluster.

Resubmitting the same command skips finished units and resumes unfinished ones.
`-- --retrain` retrains; anything after `--` goes to train.py.

Before a first real sweep, run a smoke run into a throwaway home: 1000 steps
per backend, both at once, ~3 min. It checks the image, the mounts and GPU
access on Tillicum without touching the real data:

```bash
rm -rf /gpfs/scrubbed/$USER/franka_smoke
mkdir -p /gpfs/scrubbed/$USER/franka_smoke/franka_data/baseline_prep/libero_10
cp -r /gpfs/scrubbed/$USER/franka_home/franka_data/baseline_prep/libero_10/task_3 \
    /gpfs/scrubbed/$USER/franka_smoke/franka_data/baseline_prep/libero_10/
CLUSTER_HOME=/gpfs/scrubbed/$USER/franka_smoke SBATCH_TIMELIMIT=00:10:00 \
    bash /gpfs/scrubbed/$USER/tillicum/submit_libero_train.sh libero_10 --tasks 3 --steps 1000 --apply \
    -- --extra --checkpoint-every 1
```

It passes when both `slurm_logs/libero_10_*.out` end in `trained 1`, and the
last `it/s` in `train_logs/task_3.sail.log` is ~18. Three things keep it short:

- task_3's `sail.hdf5` was labelled on the workstation, so SAIL skips its AWE
  pass, which takes ~27 min on Tillicum for an unlabelled task.
- `--checkpoint-every 1` keeps B-Spline at 1000 steps. At its default of 100
  epochs it rounds a budget up to the next checkpoint, which is ~16k steps here.
- The normal queue runs both units at once; the debug queue ran them one after
  the other.

How it works:

- **One unit per array element.**
  [train_libero.slurm](tillicum/train_libero.slurm#L59) runs the unchanged
  `libero_bridge.train --tasks <i> --backend <b> --resume` inside the image, so
  skipping, markers and per-unit logs are the local ones. There is one training
  per GPU because Tillicum allows at most 8 CPUs per GPU and B-Spline's
  dataloader alone uses 8 workers.
- **The container sees the cluster data at the workstation's path.**
  `--home $CLUSTER_HOME:/home/franka`
  ([`WORKSTATION_HOME`](tillicum/submit_libero_train.sh#L23)) puts
  `/gpfs/scrubbed/$USER/franka_home/franka_data` at `~/franka_data`. SAIL's
  config, B-Spline's task yaml and `.trained.json` record absolute paths, and
  the policy servers read the training HDF5's stamp through them. So what the
  cluster records is valid here once synced back, and rollouts file under the
  right task.
- **SAIL reads from RAM.** robomimic reads image windows out of the HDF5 for
  every sample, and GPFS is slow at small random reads. The job copies
  `sail.hdf5` to `/dev/shm`, bind-mounts it over its own path, and copies it
  back on exit if AWE labelled it
  ([cleanup](tillicum/train_libero.slurm#L41)). B-Spline trains from an
  in-memory copy it builds itself.
- **`--resume`** continues a unit that has checkpoints but no `.trained.json`
  ([`_RESUME_FROM`](libero_bridge/train.py#L48)). SAIL starts a new run dir from
  the newest `model_epoch_N.pth`, with a fresh optimizer. B-Spline continues its
  own run dir, optimizer included, to the epoch count it started with. A
  preempted job requeues (`--requeue`) and loses at most one checkpoint
  interval: 100 epochs for both.
- **Same environments as here.** The Dockerfile runs `setup_baseline_envs.sh`,
  pinned by `UV_CONSTRAINT` to versions frozen from `.venv-sail` /
  `.venv-bspline`, so the checkpoints load in the workstation's venvs. B-Spline
  checkpoints are dill pickles, which break across version changes. The
  Dockerfile's header has the command that regenerates the constraint files.

Measured locally, running the real `train_libero.slurm` against the image
through docker as a non-root user on a read-only root filesystem:

- libero_10 task_4, 1000 steps per backend.
- SAIL: 18.4 it/s from `/dev/shm`, against 16.6 in the local sweep. Its AWE pass
  took 14.8 min with B-Spline sharing the CPU.
- Each backend was killed after its first checkpoint, resumed, and finished.
- Both checkpoints then loaded in the workstation venvs and filed their rollouts
  under `libero_10/task_4`.
- The smoke run above, on one 5090 shared by both units: 138 s.

Measured on Tillicum, one H200 per unit:

- SAIL trains at ~17.7 it/s, the same as the 5090. Data loading is 7% of an
  epoch, so staging in `/dev/shm` keeps it GPU-bound.
- AWE labelling of an unlabelled task took ~27 min on the job's 8 CPUs. The
  real sweep pays that once per SAIL unit, well inside the 12 h limit.

| symptom | cause |
|---|---|
| `no .../franka-baselines.sif; apptainer pull it first` | the pull step, or `SIF=` points elsewhere |
| `missing .../task_<i>/<backend>.hdf5` | the rsync up has not run, or `--tasks` names a task not converted |
| libero_90: `nothing to train`, or `missing .../task_9/sail.hdf5` | its prep dirs carry task names, not `task_<i>`; the cluster path needs indexed dirs, so reconvert |
| units sit in `PD` with `QOSMaxGRESPerUser` | the per-user GPU cap; the rest start as GPUs free up |
| a unit log says `already trained to N steps` | its `.trained.json` exists on the cluster; `-- --retrain` |
| `training did not produce a checkpoint (exit 0, failed=False)` | SAIL's budget ended between checkpoints, which an image from before the last-epoch save could not handle; pull the image again |
| `getpwuid(): uid not found` | the image run under docker `--user` with no passwd entry; apptainer adds one |

## Speed

Each method has its own knob; they are not interchangeable.

| method | knob | default |
|---|---|---|
| SAIL | `--fast-fps` / `--slow-fps`, playback rate of action rows | 100 / 20 Hz (`baselines.exec`) |
| B-Spline | `--speed-up-times`, plan playback multiplier | 1.0 |
| pi05 | none, one action per env step | -- |

`fast_fps` does nothing for B-Spline in sim -- on the arm it is a goal dispatch
rate, here the plan is sampled on the env clock.

SAIL also needs a controller that keeps up with its playback, so it rolls out
under its own gains ([SAIL's controller](#sails-controller)). `--osc-kp` and
`--osc-damping-ratio` override them for one run, and take B-Spline too. LIBERO's
5 cm per-step limit is the same for every method and has no flag.

`--extra` forwards everything after it to each rollout
([evaluate.py:80](libero_bridge/evaluate.py#L80)). One `--sweep-id` per setting
keeps the pooled rows separate:

```bash
PREP=~/franka_data/baseline_prep/libero_90
TASKS=baselines/libero_bridge/teacher_tasks.txt

for s in 1 2 4 8; do
    python -m baselines.libero_bridge.evaluate "$PREP" \
        --tasks-file "$TASKS" --backend bspline --num-episodes 20 --save-video \
        --sweep-id "bsp_${s}x" --log-dir "$PREP/rollout_logs_bsp_${s}x" \
        --extra --speed-up-times "$s"
done
```

`--dry-run` prints the commands instead of running them. Videos cost ~145 KB per
episode.

## Read results

Runs live under `~/franka_data/outputs/`:

```
libero_10/                                 the suite (sim runs)
  task_2/                                  the task the policy trained on
    <timestamp>-bspline/                   one run
      manifest.json                        config, provenance, git state, ckpt sha256
      episodes.jsonl                       one line per episode
      force_profiles.npz                   per-step wrist force and torque (see Wrist force)
      videos/                              with --save-video
libero_90/KITCHEN_SCENE9_turn_on_the_stove/  libero_90: named, not numbered
HuskyMango/                                hardware runs, same shape
```

`rollout_summary.py`'s positional argument is any path prefix under that root.
A numbered task's header also prints its instruction.

```bash
python scripts/rollout_summary.py --all                     # everything
python scripts/rollout_summary.py libero_90                 # one suite
python scripts/rollout_summary.py libero_10/task_2          # one task
python scripts/rollout_summary.py \
    libero_90/KITCHEN_SCENE9_turn_on_the_stove/20260926_163435-bspline
```

`--sweep <id>` picks one `evaluate` invocation's rollouts wherever they landed and
adds a row per backend and suite pooling the sweep, plus an `all` row when a
backend ran more than one suite
([`render_sweep`](../scripts/rollout_summary.py#L89)). `--sweep latest` takes the
newest. Every run recorded since 2026-10-01 shows the OSC it ran in its notes,
e.g. `[kp 300, damping 0.5, step limit 5 cm]`
([`osc_text`](run_record.py#L480)); the step limit is the largest position step
one env step may command. Only rollouts started through `evaluate` carry an id unless you pass
`--sweep-id` yourself. `--json` emits manifests instead of the table.

Success rate is the fraction where LIBERO's own goal predicate fired.
Time-to-success is over successes only -- a failure's duration is just the
timeout.

For a PNG sheet instead of a table -- stat tiles, outcome per task, every episode
as a dot, episode-length histogram, the run's parameters straight from the
manifest, OSC gains and step limit included, and the wrist force
([below](#wrist-force)) -- use
[rollout_report.py](../scripts/rollout_report.py). It takes any number of sweep
ids and draws one sheet per method and suite in each, so a sweep that ran all
three gives `<sweep>.sail.png`, `<sweep>.bspline.png` and `<sweep>.pi05.png`,
and one that ran both suites gives `<sweep>.libero_10.png` and
`<sweep>.libero_90.png`. With more than one sheet in a suite it adds
`comparison.png` (`comparison.<suite>.png` when there are several suites).
Every sheet is 16:9 (3200 x 1800 px). Panels that list tasks sit side by side and
share one column of task names. On the comparison sheet each method has one
colour in every bar chart (pi05 green, SAIL amber, B-Spline indigo), and several
sweeps of one method are shades of that colour, lightest first in speed-up order
([`column_colors`](../scripts/rollout_report.py#L720)):

```bash
python scripts/rollout_report.py --sweep sail_100_20 bsp_1x bsp_2x bsp_4x bsp_8x
./scripts/collect_rollout_report.sh ~/franka_ws/libero_report sail_100_20 bsp_2x
./scripts/collect_rollout_report.sh ~/franka_ws/libero_report_sail_gains sail_kp300_z05
```

The collect script copies the sheets, each sweep's `rollout_summary.py --sweep`
table as `<sweep>.summary.txt`, and every rollout video into one folder, split by
outcome as `videos/<sweep>/<task>/{success,timeout}/`, with an `index.json`
giving each run's rates, video counts and OSC. That folder is data, so it
is gitignored -- regenerate it rather than committing it.

### Wrist force

Every sim rollout since 2026-10-01 records the force and torque at the gripper's
wrist sensor after each env step. It is the same reading multi-fast's
`eval_fast.py` takes
([libero.py:379](../multi-fast/utils/envs/libero.py#L379)), read by
[`SimTask.wrench`](libero_bridge/sim_env.py#L227) and recorded by
[`Stepper._advance`](libero_bridge/rollout.py#L193). It lands in three places:

| where | what |
|---|---|
| `force_profiles.npz` in the run directory | `ee_force_003` and `ee_torque_003` are episode 3's (steps, 3) vectors in N and Nm; `control_freq` converts steps to seconds. 3-10 KB per episode, rewritten after each one |
| `episodes.jsonl`, `ee_force_n` | that episode's mean, median, p95 and max of \|F\| ([`force_stats`](force_log.py#L22)) |
| `manifest.json`, `summary.ee_force_n` | the run: each of those averaged over episodes, which is what `eval_fast.py` prints ([eval_fast.py:504](../multi-fast/eval_fast.py#L504)), plus `peak`, the largest single reading ([`force_summary`](run_record.py#L389)) |

The numbers compare directly with eval_fast's. It masks every step after
success; here the episode already ends at success, so nothing needs masking. At
rest the sensor reads about 5 N, which is the gripper's own weight. A mean near
5 N therefore means the arm barely touched anything; contact shows in p95 and
max.

Where it shows up:

- The rollout log: `|F| mean / p95 / max` on every episode line, and the run's
  averages after the success rate.
- `rollout_summary.py`: an `|F| N avg/p95/max` column, per run and pooled per
  sweep.
- The sweep sheet: a seventh tile (mean \|F\|, with p95 and max), each episode's
  p95 as a dot per task (the third per-task panel in the top row), a histogram of
  each episode's peak, and every episode's \|F\| profile overlaid on a log axis
  ([`force_overlay`](../scripts/rollout_report.py#L438)). Colour is the task, in
  sheet order. Shade is the episode, first lightest. Tasks 9-16 reuse the eight
  colours dashed, and any beyond that are grey
  ([`task_styles`](../scripts/rollout_report.py#L425)). A key beside the overlay
  lists each task's p95 and peak.
- The comparison sheet: a per-task p95 grid and mean/p95/max bars per sweep.
- `collect_rollout_report.sh`'s `index.json`: `ee_force_n` per run.

The real rollouts record the same statistics in the same files, from the arm's
own force estimate ([ROLLOUT.md](ROLLOUT.md#end-effector-force)).

Runs recorded before 2026-10-01 have no profiles. Their tile reads "not
recorded", the column shows `-`, and the extra rows are left out.

To pull one episode's profile out of the newest run:

```bash
python - <<'EOF'
import numpy as np
from pathlib import Path
npz = max(Path.home().glob("franka_data/outputs/libero_*/*/*/force_profiles.npz"),
          key=lambda p: p.stat().st_mtime)
z = np.load(npz)
f = z["ee_force_000"]                                    # (steps, 3), N
t = np.arange(1, len(f) + 1) / float(z["control_freq"])  # simulated seconds
print(npz.parent.name, f.shape, f"peak {np.linalg.norm(f, axis=1).max():.1f} N")
EOF
```

## The plant is stock

No plant or gripper overrides. B-Spline and pi05 also run the stock controller;
SAIL runs its own ([next section](#sails-controller)). Check this first if a
number looks wrong.

multi-fast's [plant_overrides.py](../multi-fast/utils/envs/plant_overrides.py)
fits MuJoCo to the real FR3, which has no place here: the demos were recorded
under the shipped Panda model, so a fitted plant measures the gap between plants
rather than between policies.
[regenerate_libero_dataset.py:122](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L122)
recorded them with a bare `OffScreenRenderEnv`, and
[sim_env.py:184](libero_bridge/sim_env.py#L184) builds the rollout env the same
way instead of through `make_libero_env`, which would re-apply the overrides on
every reset.

Read off the live env: `dof_armature` `5/(i+1)` per joint, `dof_frictionloss`
0.1, `dof_damping` 0.1 (0.01 wrist), joint torque limits 80 Nm (12 Nm on the
last two). The stock controller: kp 150, damping ratio 1.0, `uncouple_pos_ori`
true, `output_max` 0.05 m / 0.5 rad, `control_freq` 20 Hz.

## SAIL's controller

SAIL rolls out under its own OSC gains,
[`baselines.sail.sim_osc`](../config/policy.yaml#L88): kp 300 (stock 150) and
damping ratio 0.5 (stock 1.0). Everything else is stock, including LIBERO's 5 cm
limit on how far one env step may move the position target, which every method
runs under. B-Spline and pi05 keep the stock gains
([`baselines.bspline.sim_osc`](../config/policy.yaml#L117) is null). Each run's
manifest records what the controller actually ran as `environment.osc`, and
`rollout_summary.py` and the report sheets print it as
`kp 300, damping 0.5, step limit 5 cm`.

**Why SAIL gets its own.** SAIL trains on the poses the arm *reached*
(`absolute_actions`, see [Where the data comes from](#where-the-data-comes-from)),
not on the goals the teleoperator sent. A reached pose does not depend on the
controller that produced it. So the method swaps the soft teleoperation
controller for a stiff one at rollout and plays the targets faster than they were
recorded (SAIL paper, arXiv 2506.11948, Sec. 4.2). The paper's settings:

| source | kp | damping ratio | step limit | torque limits |
|---|---|---|---|---|
| paper, sim (Table J.4) | 1000-3000 per task | 0.5-1.0 | none (absolute poses) | removed |
| upstream code, `osc_pose_SAIL.json` | 600 | 1.0 | none | removed |
| paper, real Franka (Table J.5) | 2x position, 1.6x rotation | 1.0 | -- | -- |
| here | 300 | 0.5 | 5 cm (stock) | stock |

Without the stiff controller, the paper's ablation loses most of SAIL's speed
and some of its success (Table K.6, column "-HG": on Lift, success 1.00 to 0.67
and speed over the demos 3.98x to 0.71x).

**Why the stock gains held SAIL back.** Each env step's position target is at
most 5 cm from the measured pose. While the arm chases a target further than
that, the commanded acceleration `kp·0.05 - kd·v` reaches zero at
`v = sqrt(kp)·0.05 / (2·damping ratio)`: 0.31 m/s at the stock gains. The demos
already move at up to 0.24-0.34 m/s (per task, from `absolute_actions` at 20 Hz),
and SAIL's 5x playback asks for 0.7 m/s at the median and 1.25 m/s at the 95th
percentile. kp 300 alone raises the cap to 0.43 m/s; kp 300 at damping 0.5
raises it to 0.87 m/s, which is why halving the damping matters as much as
doubling kp.

**Measured without a policy.** Replay each demo's reached poses on SAIL's own
schedule -- precision-labelled rows at 1x, the rest at 5x -- and check whether
the task still succeeds. This is the paper's App. G.1 test. 12 tasks (the 8
teachers and libero_10 tasks 0-3), 10 demos each:

```bash
for t in 9 29 44 46 57 61 67 72; do
    multi-fast/.venv/bin/python scripts/check_libero_sim_rollout.py --task $t --demos 10 \
        --targets reached --osc-kp 300 --osc-damping-ratio 0.5
done
for t in 0 1 2 3; do
    multi-fast/.venv/bin/python scripts/check_libero_sim_rollout.py --suite libero_10 --task $t \
        --source-dir ~/libero_data/libero_10/libero_10_relabeled --demos 10 \
        --targets reached --osc-kp 300 --osc-damping-ratio 0.5
done
```

| kp | damping ratio | demos succeeded | mean steps to success | substeps with a joint at its torque limit |
|---|---|---|---|---|
| 150 | 1.0 (stock) | 22.5% | 96 | 2.6% |
| 300 | 1.0 | 50.8% | 91 | 13.5% |
| **300** | **0.5** | **56.7%** | **80** | 27.4% |
| 600 | 1.0 | 65.0% | 79 | 26.6% |
| 1000 | 0.75 | 53.3% | 81 | 56.3% |
| 3000 | 0.5 | 37.5% | 74 | 80.5% |

The paper removed the torque limits; here the plant stays stock, which is why
kp 1000 and up do worse. At demo speed (`--fast-fps 20`, every row at 1x) the
same replay succeeds on 88.3% of demos stock and 94.2% at kp 300 and damping
0.5, so the gains only matter once SAIL speeds up. kp 600 replays best; kp 300
is what the policies were measured at, and the 2x the paper used on its real
Franka.

**Measured with the policies.** 11 tasks (the 8 teachers and libero_10 tasks
0-2), 20 episodes each, all on 2026-10-01, all with the 5 cm limit:

| kp | damping ratio | sweep | libero_90 | libero_10 | all 11 tasks |
|---|---|---|---|---|---|
| 150 | 1.0 (stock) | `gains_stock` | 85.6%, 5.35 s | 71.7%, 13.30 s | 81.8%, 7.52 s |
| 300 | 1.0 | `gains_kp300` | 86.9%, 4.56 s | 61.7%, 10.82 s | 80.0%, 6.27 s |
| **300** | **0.5** | `sail_kp300_z05` | **86.2%, 3.94 s** | **66.7%, 9.48 s** | **80.9%, 5.45 s** |

Each cell is success rate and mean median time to success. Success does not
move: the spread is smaller than the run-to-run noise (three reruns of the stock
setting on the teachers scored 86.2%, 88.8% and 91.9%; libero_10 has only 60
episodes per row). Time falls 28%, and kp 300 at damping 0.5 was faster than
stock on all 11 tasks and faster than damping 1.0 on 10. In closed loop the
policy re-plans from wherever the arm is, so a slow arm costs time rather than
success. The replay above cannot re-plan, which is why it shows the tracking so
plainly.

Lifting the 5 cm limit to 1 m, as upstream's absolute pose targets would, was
tried the same day (sweeps `gains_*_clip`) and dropped: the limit is part of
LIBERO's action space, every method runs under it, and with it kept the same
gains were no slower (5.45 s against 5.82 s).

Reproduce a row. This one is stock; with no `--osc-*` flags SAIL runs its
default. Use a new sweep id, or the pooled row mixes in the runs above:

```bash
SWEEP=sail_stock_rerun
OSC=(--osc-kp 150 --osc-damping-ratio 1)
python -m baselines.libero_bridge.evaluate ~/franka_data/baseline_prep/libero_90 \
    --tasks-file baselines/libero_bridge/teacher_tasks.txt --backend sail --num-episodes 20 \
    --sweep-id $SWEEP --extra "${OSC[@]}"
python -m baselines.libero_bridge.evaluate ~/franka_data/baseline_prep/libero_10 \
    --tasks 0 1 2 --backend sail --num-episodes 20 --sweep-id $SWEEP --extra "${OSC[@]}"
python scripts/rollout_summary.py --sweep $SWEEP
```

To run several settings at once, give each its own `--port` after `--extra`, and
start them a few seconds apart: two runs of one task and method that start in
the same second refuse to share a run directory.

**What still limits SAIL's speed: the precision labels.** Rows near a gripper
event or a dense cluster of waypoints play at `slow_fps`, 20 Hz, which is demo
speed. Over the 12 tasks that is 74% of env time even with perfect tracking, so
SAIL can at best run about 2x faster than the demos. In the `sail_kp300_z05`
sweep, 70% of env steps went to slow rows. The paper sets the slow speed per
task, from 1x to 5x (Table J.4, "slowdown c"); `--slow-fps` is the knob here.

**B-Spline keeps the stock gains.** B-Spline trains on the goals the
teleoperation controller was sent (`goal_pos`), which lead the reached pose by
29 mm at the median. Those goals belong to the controller that was chasing
them. A stiffer one reaches them sooner than the demos did and overshoots (the
paper's App. G.1 makes the same point). Measured:

- Replay of the commanded goals at demo speed (`check_libero_sim_rollout.py`
  without `--targets`, the 12 tasks above): 98.3% of 120 demos stock, 95.0% at
  kp 300 with damping 1.0 or 0.5.
- The policy at `--speed-up-times 2` on the 11 tasks above: stock 89.1% at
  6.38 s (`bsp_2x` and the 2026-09-30 libero_10 runs), kp 300 85.5% at 5.54 s
  (`bsp2x_kp300`).

kp 300 trades 3.6 points of success for 13% of time, the trade
`--speed-up-times` already makes. The `--osc-*` flags still take B-Spline for
anyone who wants that trade.

pi05 refuses the `--osc-*` flags: its normalised deltas were learned under the
stock controller.

## The bridge

Three processes, three interpreters:

| process | interpreter | holds |
|---|---|---|
| policy server | `.venv-sail` / `.venv-bspline` | checkpoint, torch, upstream repo |
| rollout + env | `multi-fast/.venv` | robosuite, libero, mujoco, pi0.5 |
| launcher | `.venv` | `franka_config`, for ports and constants |

SAIL and B-Spline conflict with each other and with lerobot -- diffusers 0.21.4
vs 0.11.1, robomimic 0.3.1 vs 0.2.0 -- and neither has libero. So the policy is a
separate process and only a socket crosses between them. That is why the sim work
needed no new policy code: both servers are the unchanged hardware ones
([sail](sail_bridge/policy_server.py), [bspline](bspline_bridge/policy_server.py)),
which do not care whether an observation came from an arm or from mujoco.
[`python_for`](interpreters.py#L32) picks the interpreter: `$SAIL_PYTHON` /
`$BSPLINE_PYTHON`, else the venv, else `conda run`.

**[libero_rollout.sh](../scripts/libero_rollout.sh)**, in order: splits arguments
(`--guide-config` and `--port` to the server, rest to the rollout,
[:32](../scripts/libero_rollout.sh#L32)); refuses if the port is already held,
because a leftover server would answer the handshake as if it were ours
([:97](../scripts/libero_rollout.sh#L97)); starts the server in its own
interpreter ([:107](../scripts/libero_rollout.sh#L107)); waits on the handshake
rather than sleeping, since checkpoint load time varies
([:113](../scripts/libero_rollout.sh#L113)); runs the rollout under
`multi-fast/.venv` ([:121](../scripts/libero_rollout.sh#L121)); on exit kills the
server's process group and waits for the port to free
([`cleanup`](../scripts/libero_rollout.sh#L71)). The group, not the pid, because
`conda run` spawns python as a child. It never `exec`s the rollout, which would
drop the trap and orphan the server.

**Wire protocol** ([zmq_client.py](zmq_client.py), one REQ socket, pickle):

```
{"meta": True}   -> capability dict
{"reset": True}  -> {}
{"obs": ...}     -> SAIL      {"chunk": (N, act_dim)}
                    B-Spline  {"bspline": ndarray, "bspline_meta": {...}}
```

[`PolicyClient.meta`](zmq_client.py#L80) lists every field. The rollout takes
`obs_key_shapes` to size frames, `act_dim` and `precision_column` to parse rows,
`n_obs_steps` for the frame stack, and checks `backend` and `train_dataset`
against what it was told to run. `REQ_RELAXED` + `REQ_CORRELATE`
([:54](zmq_client.py#L54)) stop one timed-out request from poisoning the socket
for every later send. Every use is behind a lock
([`PolicyClient`](zmq_client.py#L43)) because a REQ socket is not thread-safe and
B-Spline's planner requests from a worker thread. Both servers reply
`{"error": ...}` instead of dying
([`SAILPolicyServer.run`](sail_bridge/policy_server.py#L276)).

**B-Spline's server needed wrapping.** Upstream's `policy_server_bspline.py` is a
submodule file we do not edit, and it replies `{}` to anything but `reset` and
`obs` -- including the handshake. So
[bspline_bridge/policy_server.py](bspline_bridge/policy_server.py) loads it by
path ([`_load_upstream`](bspline_bridge/policy_server.py#L43)) and subclasses its
server to add the `meta` branch
([`Server`](bspline_bridge/policy_server.py#L166)), built from the checkpoint's
config. Start upstream's directly and the handshake returns `{}`.

**Shared helpers.** `rollout.py` runs under `multi-fast/.venv`, which has no
lerobot, so it cannot import `rollout_common.py`. What both loops need lives in
[policy_math.py](policy_math.py) and
[spline_plan.py](bspline_bridge/spline_plan.py)
([`decode_action`](bspline_bridge/spline_plan.py#L45)); `rollout_common`
re-exports it, so the hardware entrypoints are untouched.

## Where the data comes from

Not raw LIBERO, but what
[regenerate_libero_dataset.py](../multi-fast/scripts/libero/regenerate_libero_dataset.py)
writes -- every demo replayed, with three changes:

1. No-ops dropped ([:176](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L176))
   and failed replays discarded ([:237](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L237)),
   so the baselines see exactly the frames multi-fast sees.
2. The OSC controller's absolute goal recorded as `goal_pos` / `goal_ori`
   ([:223](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L223)). This
   is why conversion is a re-key, not a replay -- no simulator needed.
3. Ten settle steps before recording
   ([:144](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L144)); the
   rollout repeats them (`sim_env.SETTLE_STEPS`).

**Reached vs commanded.** `obs/ee_*` is where the arm was, `goal_*` where it was
told to go. SAIL trains on reached poses (upstream's design), B-Spline on
commanded. They are in different frames -- `goal_pos` and `obs/ee_pos` are the
grip site, `obs/ee_ori` is the wrist body -- so the regeneration script converts
with a per-demo tool transform
([:156](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L156)) and
sign-canonicalises against the pre-action quaternion
([:224](../multi-fast/scripts/libero/regenerate_libero_dataset.py#L224)). The
rollout undoes exactly that.

[dataset.py](libero_bridge/dataset.py) writes the two files
([`sail_episode`](libero_bridge/dataset.py#L113),
[`bspline_episode`](libero_bridge/dataset.py#L132)):

| key | SAIL | B-Spline | from |
|---|---|---|---|
| `obs/robot0_eef_pos` | yes | yes | `obs/ee_pos` |
| `obs/robot0_eef_quat` | yes | yes | `obs/ee_ori` |
| `obs/robot0_joint_pos` | yes | -- | `obs/joint_states` (AWE needs it) |
| `obs/robot0_gripper_qpos` | yes | yes | `obs/gripper_states` |
| `obs/agentview_image` | yes | yes | `obs/agentview_rgb`, flipped, resized |
| `obs/robot0_eye_in_hand_image` | yes | yes | same |
| `actions` | LIBERO's deltas | commanded pose | |
| `absolute_actions` | reached pose | -- | what SAIL trains on |
| `commanded_absolute_actions` | commanded pose | -- | for comparison |

Pose rows are `[pos(3), rotvec(3), gripper(1)]`; the gripper column is LIBERO's
own ±1 channel, unconverted. Frames are 84x84 (both upstreams' benchmark size),
`INTER_AREA` from 256, rotated 180° by
[`sim_env.policy_frame`](libero_bridge/sim_env.py#L56) -- which the converter
imports rather than copying, so training and rollout frames cannot drift.
Rotation vectors are copied, not re-derived
([`poses`](libero_bridge/dataset.py#L78)): scipy's `as_rotvec` folds the angle
into `[0, π]` and the gripper-down pose sits at π, so passing these through scipy
is what would introduce a discontinuity. Demos are renumbered from 0, since
LIBERO's indices have gaps and B-Spline's loader indexes `demo_{i}` by position.

## Training

[train.py](libero_bridge/train.py) is a loop, not a trainer -- each task's HDF5
goes to the unchanged `baselines.{sail,bspline}_bridge.train` as a subprocess, so
sim and hardware policies are trained by identical code.

- `--steps` goes to both, so they train equally long.
- Checkpoints land in `~/franka_data/policies/<suite>/task_<i>/<backend>/`, from
  the id the converter stamped on the HDF5
  ([`backend_dir`](libero_bridge/train.py#L52)).
- SAIL first runs its AWE waypoint and precision-labelling passes, ~5 min/task.
  They edit the HDF5 in place and skip if the keys exist.
- Completion is a marker, not a checkpoint
  ([`marker_for`](libero_bridge/train.py#L68)) -- both trainers checkpoint
  periodically, so "a checkpoint exists" would roll out a half-trained policy.
  `.trained.json` records the budget reached; `--allow-unfinished` overrides.
- `--resume` continues an unfinished training from its newest checkpoint
  instead of starting over (see Training on Tillicum).

[teacher_tasks.txt](libero_bridge/teacher_tasks.txt) lists eight libero_90 tasks
(9, 29, 44, 46, 57, 61, 67, 72) -- the ones multi-fast itself trains and evaluates
on. All 89 convertible tasks work, but at 100k steps that is ~12 days of GPU for
the two baselines.

## Rollout

**Action space.** The policies predict absolute poses; LIBERO takes normalised
deltas. robosuite's OSC under `control_delta=True` composes

```
goal_pos = measured_pos + scale_action(a[:3])        # ±0.05 m
goal_ori = R(scale_action(a[3:6])) @ measured_ori    # ±0.5 rad
```

so hitting a target means solving for `a` against the pose measured at that step.
[`SimTask.action`](libero_bridge/sim_env.py#L265) calls multi-fast's
[`target_slot_to_delta`](../multi-fast/utils/base_policy_utils.py#L553) rather than
reimplementing the inverse -- it is the exact inverse of the relabeler that wrote
the training targets, so executor and converter cannot drift. It also undoes the
site-to-body transform with
[`estimate_site_to_body`](../multi-fast/utils/base_policy_utils.py#L598),
re-estimated per episode in
[`SimTask.start`](libero_bridge/sim_env.py#L203). Skip that transform and nothing
errors; it just commands a goal rotated ~90°. The delta is recomputed every env
step against that step's measured pose
([`Stepper.send`](libero_bridge/rollout.py#L160)), matching `EE_POS` on the arm,
and out-of-range targets saturate like `scale_action` does. pi05 skips all of it
([`Stepper.send_raw`](libero_bridge/rollout.py#L183)).

**Clock.** The clock is the env's step counter; otherwise a plan's phase depends
on GPU speed and success rates do not reproduce. SAIL executes a fixed `inf_delay`
rows of the old plan before entering the new one, which is upstream's own latency
model. B-Spline's planner gets the sim clock and runs synchronously
([clock / synchronous](bspline_bridge/spline_plan.py#L93)) -- the only change to
that file. Episode time is simulated seconds, written to `wall_time_s` so
`rollout_summary.py` reads it unchanged, with real time beside it as
`clock_time_s`.

**SAIL** ([`sail_episode`](libero_bridge/rollout.py#L257),
[`sail_settings`](libero_bridge/rollout.py#L228)) implements upstream's three eval
mechanisms:

- *Receding horizon* -- enter a new chunk at the row matching what went out since
  its observation, never row 0, which is where the arm was already seen.
- *Precision speed modulation* -- the last action column is a label; a label near
  the current step drops it to `slow_fps`. `--no-precision` disables it.
- *Error-adaptive guidance* -- condition the next prediction on the current
  plan's tail, only while the arm tracks it. Needs a checkpoint trained with
  `future_action_condition` and a `--guide-config`; `--no-eag` disables it.

A SAIL row is a trajectory sample at the recording rate, so at `fast_fps` 100 Hz
against a 20 Hz env one row occupies a fifth of an env step and the trajectory
plays 5x faster than recorded -- the same ratio the arm runs, and the point of the
method. `Stepper.send` carries the fractional remainder; rounding rows up to whole
steps would cap every method at demo speed. The arm only keeps up under SAIL's
own controller ([SAIL's controller](#sails-controller)).

**B-Spline** ([`bspline_episode`](libero_bridge/rollout.py#L364),
[`bspline_planner_kwargs`](libero_bridge/rollout.py#L328)) is shorter because the
server returns spline parameters, not actions, so the loop samples the plan
itself. `origin_time_scale` is the env's control rate: knots count demo frames, so
`t` advances one knot per env step. Time alignment is off in sim -- upstream
stitches a new plan onto the old within a window bounded by time since its
observation, and with synchronous inference on a sim clock no sim time passes, so
the window is empty and the plan starts at `min_t`. That is correct at zero
latency; only the diagnostic was meaningless. `--time-align` restores it.

**pi05.** [`pi05.load`](libero_bridge/pi05.py#L63) builds the config block
multi-fast's own `load_base_policy` expects and returns its `Pi05BaseWrapper`, so
the 8-D state vector, the 224x224 resize-with-pad and the action unnormalisation
are multi-fast's code. This module owns only
[`observation`](libero_bridge/pi05.py#L103), which turns one LIBERO observation
into a batch-of-one with images rotated 180° and CHW float in `[0, 1]`. Four
differences from the baselines:

- **Renders at 96, not 256.** The baselines' training frames were rendered at 256
  and downscaled to 84, so their rollout renders at 256. pi0.5 is sensitive to
  it: on KITCHEN_SCENE9 the same policy scored 0/20 at 256 and 9/10 at 96.
  `--render-resolution` defaults per backend, reading multi-fast's own constant.
- **No policy server** -- openpi is already in `multi-fast/.venv`, so it loads
  in-process. `rollout.SERVED` ([:71](libero_bridge/rollout.py#L71)) is the list
  that needs one.
- **No action-space inverse** -- it emits LIBERO's own seven normalised deltas.
- **No receding horizon** -- [`pi05_episode`](libero_bridge/rollout.py#L397)
  replans every `chunk_size` (5) steps, matching `scripts/libero/eval_pi05.py`.

`load` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before the first JAX import,
or JAX takes 90% of the card. The checkpoint is
`gs://openpi-assets/checkpoints/pi05_libero`, cached under `~/.cache/openpi`; the
task instruction comes from the LIBERO benchmark API via multi-fast's loader.

**One episode:** reset, `set_init_state` with init state *i* from the suite's own
50 ([`SimTask.start`](libero_bridge/sim_env.py#L203) -- LIBERO's standard eval
states, the ones `eval_pi05.py` uses); ten settle steps; estimate the tool
transform; then loop observe / infer / convert / step
([`Stepper`](libero_bridge/rollout.py#L99)) until the env's own `done` -- LIBERO
scores the task itself, which is the whole reason to run this in sim -- or the
400-step budget. Each episode appends to `episodes.jsonl` and rewrites the
manifest.

## Results

8 tasks x 20 episodes, stock plant, all on 2026-09-26 unless marked. These are
the sweeps in `libero_report/`; `rollout_summary.py --sweep <id>` reprints any
of them.

| method | sweep | success | mean median-time |
|---|---|---|---|
| SAIL (100/20 Hz), stock controller | `sail_100_20` | 86.2% | 5.29 s |
| SAIL (100/20 Hz), its own gains, 2026-10-01 | libero_90 rows of `sail_kp300_z05` | 86.2% | **3.94 s** |
| B-Spline 1x | `bsp_1x` | 88.8% | 6.58 s |
| B-Spline **2x** | `bsp_2x` | **93.1%** | 4.91 s |
| B-Spline 4x | `bsp_4x` | 89.4% | **4.70 s** |
| B-Spline 8x | `bsp_8x` | 78.8% | 6.26 s |
| pi05 | `pi05_base` | 94.4% | 6.27 s |

B-Spline's 2026-09-24 speed sweep matched these exactly. SAIL and pi05 move
between reruns: three full SAIL sweeps scored 86.2%, 88.8% and 91.9%, three
pi05 sweeps 93.8%, 94.4% and 95.6%.

With its own gains SAIL is the fastest method here, but still below B-Spline 2x
and pi05 on success. A same-day rerun with the stock gains scored 85.6% at
5.35 s (`gains_stock`), so the gains bought the time and cost no success.

**B-Spline peaks at 2x**, beating 1x on both axes. 4x is faster for ~4 points of
success. 8x is worse than 1x on both.

8x fails on replan rate, not the action envelope. Plans per env step over
successful episodes: 0.072 at 1x, 0.152 at 2x, 0.342 at 4x, **0.844 at 8x**.
(`scripts/rollout_report.py` pools all episodes instead, so its figures are
higher -- a timeout replans for the full 400 steps.) A plan's horizon in wall-time
shrinks as `1/speed`, so at 8x the spline is consumed in about one env step and
B-Spline becomes a single-step controller, losing the open-loop plan it depends
on. Successful episodes take 132 steps at 1x, 103 at 2x, 102 at 4x, 133 at 8x.

The obvious alternative -- LIBERO's ±0.05 m per-step clip being the ceiling --
was tested and rejected: 40-80% of steps exceed the clip at 8x, but the
correlation between a task's saturation fraction and its 2x-to-8x drop is -0.59,
i.e. backwards.

Per-task spreads at 20 episodes are about ±10 points. Read the pooled rows.

## Verified

| claim | how | result |
|---|---|---|
| the executor reproduces the demos | `check_libero_sim_rollout.py --task <T>` replays each demo's recorded targets through `SimTask.action` and the `Stepper` | 40/40 over the eight tasks (5 demos each, 2026-10-01) |
| the pose-to-delta inverse | same check, derived deltas vs recorded actions | mean 3e-4 normalised, max 3e-2 (~1.5 mm) |
| the env reproduces the recording | replay recorded actions, diff observations | 3e-14 on `robot0_eef_pos` |
| the plant is stock | read `dof_armature` etc. off the live env | values above |
| SAIL's gains follow its targets at its speed better than stock | `check_libero_sim_rollout.py --targets reached`, 12 tasks x 10 demos ([SAIL's controller](#sails-controller)) | 56.7% at SAIL's gains, 22.5% stock |
| both hardware loops still work | `check_baseline_rollout_offline.py` | 86/86 |
| the line numbers here | `python scripts/check_doc_refs.py` | 0 problems |

Run `check_libero_sim_rollout.py` after touching `sim_env.py` or `Stepper`. Needs
mujoco, no policy, no GPU, ~1 min for five demos.

## When a run fails

| symptom | cause |
|---|---|
| `server on port N is a 'bspline' server, not sail` | a previous run's server of the other backend holds the port |
| `the server on port N serves a policy trained on <other task>` | `--suite` or `--task` does not match `--ckpt`, or a previous run's server of the **same** backend holds the port. Before this guard existed a stale server scored 0% on every task but its own. `ss -ltnp \| grep 5556`, kill that pid |
| `<file> holds libero_10/task_2, not libero_90/task_2` | the converter's `--out-dir` is another suite's prep directory |
| `<prep dir> should hold one LIBERO suite's tasks` | two suites, or a conversion from before the numbering, share one prep directory |
| `port N is already held by pid P` | a previous run's server is still up; the launcher refuses before loading a checkpoint. `kill P` and retry |
| `port N answered without a backend: {}` | upstream's `policy_server_bspline.py` started directly; use `bspline_bridge/policy_server.py` |
| `PolicyTimeout: no reply from the policy server` | the server died. Its traceback is in the sweep's per-task log, not the rollout's |
| `no checkpoint yet` / `training has not finished` | the `.trained.json` gate; `--allow-unfinished` overrides |
| `ValidationError` naming `run_tags` | B-Spline's wandb tag exceeded 64 chars. `bspline_bridge/train.py:task_name` truncates with a hash, so the guard was bypassed |
| `ModuleNotFoundError: robosuite` | wrong interpreter; must be `multi-fast/.venv` |
| `no interpreter for the sail baseline` | `setup_baseline_envs.sh` has not run, or `$SAIL_PYTHON` is stale |
| the card fills when a pi05 rollout starts | JAX preallocation; check nothing imported JAX before `pi05.load` |
| 0% where the method should work | check `render_resolution` in the manifest first |

## Traps

- **`pi05` is a base policy, not multi-fast.** The table compares two complete
  methods against one method's base.
- **pi0.5 sees a different observation** -- 224x224 resize-with-pad and an 8-D
  state, against 84x84 crops. That is each method's own training distribution, so
  it is the right comparison, but a success-rate gap is not attributable to
  architecture alone.
- **`fast_fps` means different things per backend.** SAIL: playback rate.
  B-Spline: a dispatch rate on the arm, nothing in sim.
- **Init states are the suite's 50**, not the demos' start states. They overlap
  but differ, so a policy can start from a state no demo did. Standard LIBERO
  protocol; `--first-episode` shifts the window.
- **Keep `--render-resolution` at 256 for SAIL and B-Spline** unless you
  reconvert the training data. pi05 defaults to 96.
- **One policy per task.** Neither baseline is language-conditioned.
- **SAIL runs before 2026-10-01 used the stock controller.** Their manifests
  have no `environment.osc`. Do not pool them with later SAIL runs, nor with the
  2026-10-01 sweeps named `gains_*_clip`, which lifted the 5 cm step limit and
  were dropped; their summaries say `step limit 100 cm`.
- **An index means a different task in every suite.** `--task 2` is valid in all
  of them, so a by-hand rollout on the wrong `--suite` runs the wrong task
  rather than failing. SAIL and B-Spline catch it against the checkpoint; pi05
  has no checkpoint to check against.

## Files

| file | what |
|---|---|
| [libero_bridge/dataset.py](libero_bridge/dataset.py) | LIBERO demos -> `sail.hdf5` + `bspline.hdf5` |
| [libero_bridge/sim_env.py](libero_bridge/sim_env.py) | the stock env, the observation, the pose-to-delta inverse |
| [libero_bridge/rollout.py](libero_bridge/rollout.py) | the sim loop for all three backends |
| [libero_bridge/pi05.py](libero_bridge/pi05.py) | multi-fast's base policy as a backend |
| [libero_bridge/train.py](libero_bridge/train.py) | per-task training sweep |
| [libero_bridge/evaluate.py](libero_bridge/evaluate.py) | per-task rollout sweep, sweep ids |
| [tillicum/submit_libero_train.sh](tillicum/submit_libero_train.sh) | the training sweep on Tillicum, one Slurm array element per task and backend |
| [tillicum/train_libero.slurm](tillicum/train_libero.slurm) | one array element: stage, mount, train |
| [tillicum/progress.sh](tillicum/progress.sh) | each unit's epoch, rate and time left, read from its training log |
| [tillicum/Dockerfile](tillicum/Dockerfile) | the image, both envs pinned to this machine's |
| [zmq_client.py](zmq_client.py) | the client for both policy servers |
| [interpreters.py](interpreters.py) | which python runs each baseline |
| [policy_math.py](policy_math.py) | helpers shared by sim and hardware loops |
| [../scripts/libero_rollout.sh](../scripts/libero_rollout.sh) | starts the server and the env in their own venvs |
| [../scripts/setup_baseline_envs.sh](../scripts/setup_baseline_envs.sh) | builds the two baseline venvs |
| [../scripts/check_libero_sim_rollout.py](../scripts/check_libero_sim_rollout.py) | replays demos through the executor |
| [../scripts/rollout_summary.py](../scripts/rollout_summary.py) | the comparison table |
| [../scripts/rollout_report.py](../scripts/rollout_report.py) | PNG summary sheets, one per sweep |
| [../scripts/collect_rollout_report.sh](../scripts/collect_rollout_report.sh) | sheets + videos into one folder |

Unchanged and reused: both `*_bridge/train.py`, both `*_bridge/policy_server.py`,
`train_common.py`, `zmq_client.py`. `run_record.py` and `rollout_summary.py` only
gained sweep ids, the OSC each run recorded and, for the summary, the
instruction beside a numbered task.
