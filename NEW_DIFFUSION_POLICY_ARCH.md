# Train and evaluate SAIL, B-Spline and ZAP in sim (LIBERO)

The sim version of [TRAIN_AND_EVAL.md](TRAIN_AND_EVAL.md), for whole LIBERO
suites. As written it covers libero_10, libero_goal and libero_spatial: 30 tasks,
10 per suite.

```
re-record LIBERO demos on the sysid plant → convert to three training sets → send
to Tillicum → train SAIL, B-Spline and base dp on Tillicum → fetch → base dp's
stats → train ZAP on Tillicum → evaluate all four in sim → report with charts
```

**The same recipe as on the arm.** All three methods train on the same demos with
the same network, optimizer, learning-rate schedule, EMA (a running average of
the weights, which is what inference loads) and batch size, all from
`baselines/diffusion_recipe.py`. Only what each policy reads and predicts
differs. One thing is not the same: the step budget is 100k steps in sim and 200k
on the arm (`send_datasets.sh` sets both). To match the arm, use
`--steps 200000 --name recipe-200k` in step 4; training then takes twice as long.

**One environment for every method.** The `sysid_2026_09_16` plant (MuJoCo fitted
to the FR3), ZAP's gripper settings, and each task's 50 standard LIBERO start
states. The demos are re-recorded on this plant first (step 1), so no policy is
evaluated on a plant other than the one its demos came from.

**Names.** "Base dp" is our diffusion policy alone. "Base dp + ZAP" is base dp
with the ZAP residual on top (a point-cloud policy that corrects base dp's plan).
The code files ZAP runs as method `multifast` and base dp runs as `dp`; the
report's charts use the names above. A *task index* is a task's position in its
suite in LIBERO's own order, 0-9; everything for task 3 of libero_goal is filed
under `libero_goal_sysid_2026_09_16/task_3`.

How each step works, and the checks that the methods really see the same data
and environment, are in [baselines/LIBERO_SIM.md](baselines/LIBERO_SIM.md#the-matched-comparison).

## Status (2026-10-29)

| step | state |
|---|---|
| 1-5 on libero_90 task 9 (the pilot) | run for real. SAIL, B-Spline and base dp trained 100k steps on Tillicum on 2026-10-24 |
| 6 on libero_90 task 9 | run: base dp succeeds 80% (40 of 50); 20 reference trajectories written |
| 7, ZAP training on a dp base | **never run.** No teacher or student exists for any dp base yet. The commands follow the pilot's, untested on Tillicum |
| 8-9 | checked end to end on libero_90 task 9, using the old student 40cdf603 for 10 episodes: eval, import, summary table and every chart |
| 1-2 on libero_10, libero_goal, libero_spatial | checked on one libero_10 task in a scratch folder; nothing kept |

## What it costs

Measured numbers are marked; the rest are extrapolated from them.

| step | where | time | notes |
|---|---|---|---|
| 1. re-record | workstation | about 20 min, three suites at once | measured: libero_10 task 0, 36 demos in 117 s |
| 2. convert | workstation | about 5 min a suite | measured: 30 s for that task |
| 4. train | Tillicum | about 290 GPU-hours | measured on an H200, libero_90 task 9: SAIL 112 min, B-Spline 135 min, base dp 5 h 32 min |
| 6. base dp stats | workstation | about 1.5 h | measured on task 9: 1 min for the 50-episode eval, 80 s for the references |
| 7. ZAP | Tillicum | the long pole | one 24 h teacher per task, then collection, then one student per suite |
| 8. evaluate | workstation | about 3 h for SAIL and B-Spline, 2.5 h for the other two | measured: 10 episodes of base dp + ZAP in 41 s |

**Disk.** The workstation had 18 GB free on 2026-10-29, which is not enough.
Free space before step 1.

| what | where | size |
|---|---|---|
| the source demos (already there) | `~/libero_data/<suite>/` | libero_10 38 GB, goal 20 GB, spatial 20 GB |
| re-recorded demos (step 1) | `~/libero_data_sysid/<suite>/` | up to the same again, less the demos that fail: libero_10 task 0 came to 2.3 of its 3.9 GB. Only step 2 reads them |
| training sets (step 2) | `~/franka_data/baseline_prep/`, `~/.cache/huggingface/lerobot/` | about 15 GB: libero_10 task 0 was 0.5 GB + 0.1 GB |
| policies (step 5) | `~/franka_data/policies/` | 4.5 GB a task, 135 GB in all. Two thirds are checkpoints the evaluation never reads: SAIL's epoch 100 and base dp's 20k-80k steps |

## Set some constants

Run this block in every new terminal on the workstation before any step.

```bash
cd ~/franka_ws && source .venv/bin/activate
TUSER=[YOUR TILLICUM USERNAME GOES HERE]   # your Tillicum (UW NetID) username
export TILLICUM_USER=$TUSER
SUITES="libero_10 libero_goal libero_spatial"
PLANT=sysid_2026_09_16                     # multi-fast/cfg/plant/<name>.yaml
RUN=p0916                                  # prefix of this evaluation's sweep ids; pick a new one to evaluate again

# The task indices step 2 converted for a suite.
tasks_of() { ls -d ~/franka_data/baseline_prep/${1}_$PLANT/task_* | sed 's/.*task_//' | sort -n; }
# Base dp's newest checkpoint for <suite> <task index>.
base_of() { echo "$(ls -td ~/franka_data/policies/${1}_$PLANT/task_$2/diffusion/*/checkpoints/last | head -1)/pretrained_model"; }
# Serve one base dp checkpoint on port 5557 and return once it answers. dp_down stops it.
dp_up() {
    ~/franka_ws/.venv/bin/python ~/franka_ws/baselines/dp_bridge/policy_server.py --port 5557 \
        --ckpt-path "$1" > ~/franka_data/dp_server.log 2>&1 &
    DP_PID=$!
    ~/franka_ws/.venv/bin/python ~/franka_ws/scripts/_wait_policy_server.py dp 5557 300
}
dp_down() { kill $DP_PID; wait $DP_PID 2>/dev/null; }
```

Base dp needs its own process: multi-fast's Python environment cannot load it, so
multi-fast asks this server for base dp's plans. One server holds one task's
checkpoint, which is why steps 6 and 8 start and stop it for each task.

## Folder structure

```
~/libero_data/<suite>/                          the stock-plant demos (multi-fast's regenerate_libero_dataset.py)
~/libero_data_sysid/<suite>/                    the same demos re-recorded on the sysid plant, and <task>_demo.replay.json
~/franka_data/
  baseline_prep/<suite>_<plant>/task_<i>/       SAIL's and B-Spline's training data: sail.hdf5, bspline.hdf5
  baseline_prep/<suite>_<plant>/train_logs/     SAIL and B-Spline training logs, fetched from Tillicum
  policies/<suite>_<plant>/task_<i>/            trained policies, fetched from Tillicum
    diffusion/recipe-100k/                        base dp
    sail/<timestamp>/                             SAIL
    bspline/<timestamp>/                          B-Spline
  outputs/<suite>_<plant>/task_<i>/             one folder per evaluation run, <timestamp>-<method>:
                                                sail, bspline, dp (base dp) or multifast (base dp + ZAP)
  analysis/libero_<plant>_<run>_<date>/         the report: charts, tables, videos, and its .zip
~/.cache/huggingface/lerobot/<suite>_<plant>/task_<i>/   base dp's training data (a LeRobot dataset)
~/franka_ws/multi-fast/
  cfg/base_stats/libero/dp_recipe/<plant>/<suite>/task_<i>.yaml   base dp's success rate and episode length, plus _references.npz
  logs/distill/<student>/                       ZAP students, fetched from Tillicum

Tillicum, /gpfs/scrubbed/<user>/
  tillicum/      the job scripts             franka_src/   the code the diffusion jobs run
  containers/    the images                  franka_home/  mirrors ~ above (franka_data/, .cache/)
  multi-fast/    ZAP's code: logs/fast/ teachers, data/distill/ their rollouts, logs/distill/ students
```

## First time only

1. **Images on the workstation.** Rebuild and push the baselines image whenever
   `baselines/` has changed since its last push, and build the image that runs
   base dp inside a Slurm job once:
   ```bash
   cd ~/franka_ws
   TAG=$(git describe --always --dirty)
   docker build -f baselines/tillicum/Dockerfile -t atsai06/franka-baselines:$TAG .
   docker tag atsai06/franka-baselines:$TAG atsai06/franka-baselines:latest
   docker push atsai06/franka-baselines:$TAG && docker push atsai06/franka-baselines:latest
   docker build -f baselines/tillicum/Dockerfile.dp -t atsai06/lerobot-dp-server:v0.5.1 baselines/tillicum
   docker push atsai06/lerobot-dp-server:v0.5.1
   ```
2. **Pull them on a Tillicum compute node**, not the login node:
   ```bash
   ssh $TUSER@tillicum.hyak.uw.edu
   salloc --qos=debug --gpus=1 --cpus-per-task=8 --mem=64G --time=00:30:00
   export APPTAINER_CACHEDIR=/gpfs/scrubbed/$USER/apptainer_cache APPTAINER_TMPDIR=/gpfs/scrubbed/$USER/tmp
   C=/gpfs/scrubbed/$USER/containers; mkdir -p $C $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
   apptainer pull --force $C/franka-baselines.sif docker://atsai06/franka-baselines:latest
   [ -f $C/lerobot-train.sif ] || apptainer pull $C/lerobot-train.sif docker://atsai06/lerobot-act-train:v0.5.1
   [ -f $C/lerobot-dp-server.sif ] || apptainer pull $C/lerobot-dp-server.sif docker://atsai06/lerobot-dp-server:v0.5.1
   exit
   ```
3. **multi-fast on Tillicum under your own account**, for step 7. Follow
   [multi-fast/scripts/cluster/README.md](multi-fast/scripts/cluster/README.md)
   steps 2 and 3 (the container and your folders), with two differences.
   `bash baselines/tillicum/sync.sh multifast` (on the workstation) replaces its
   `git clone`. And step 4's `sed` goes over every job script:
   `sed -i "s#/gpfs/scrubbed/ecai0608#/gpfs/scrubbed/$USER#g" scripts/cluster/*_tillicum.slurm`.

## 1. Re-record the demos on the sysid plant (workstation)

```bash
mkdir -p ~/libero_data_sysid
cd ~/franka_ws/multi-fast
for S in $SUITES; do (
  for f in ~/libero_data/$S/${S}_relabeled/*_demo.hdf5; do
    out=~/libero_data_sysid/$S/$(basename "$f")
    [ -f "${out%.hdf5}.replay.json" ] && continue   # finished in an earlier run
    rm -f "$out"                                     # half-written by an interrupted run
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/libero/regenerate_on_plant.py \
        "$f" --suite $S --out-dir ~/libero_data_sysid/$S
  done > ~/libero_data_sysid/$S.log 2>&1 ) &
done; wait
grep -h ": kept" ~/libero_data_sysid/libero_*.log
cd ~/franka_ws
```

Each step of a demo commands the goal pose the original demo commanded, so the
arm follows the demonstrated path while the fitted friction slows it down. Only
demos that still succeed are kept; `<task>_demo.replay.json` beside each file
lists every demo's outcome. The three suites run at once, one task after another
within each; running the block again skips finished tasks.

Expect to lose more demos than on the pilot. libero_90 task 9 kept 40 of 45;
libero_10 task 0 kept 21 of 36. Holding the last goal 40 more steps
(`--hold-steps 40`) saved only 2 of those 15, so the failures are not the slower
arm running out of time at the end; the slower plant changes how the grasps and
contacts play out.

## 2. Convert (workstation)

```bash
for S in $SUITES; do
  python -m baselines.libero_bridge.dataset ~/libero_data_sysid/$S --out-dir ~/franka_data/baseline_prep/${S}_$PLANT
  for d in ~/franka_data/baseline_prep/${S}_$PLANT/task_*; do
    python -m baselines.sail_bridge.gripper_labels $d/sail.hdf5
  done
done
```

1. **Convert.** For every task, three training sets from the same demos and the
   same 84x84 frames: `sail.hdf5` and `bspline.hdf5` in
   `baseline_prep/<suite>_<plant>/task_<i>/`, and base dp's LeRobot dataset in
   `~/.cache/huggingface/lerobot/<suite>_<plant>/task_<i>`.
2. **Label SAIL's precise steps** from gripper events, as on the arm (19% of
   steps on libero_10 task 0). SAIL slows down on these steps. Without labels,
   SAIL runs its own labelling pass (AWE) on Tillicum instead, about 27 min a
   task.

Keep the LeRobot datasets after training: base dp's policy server reads which
environment its demos came from out of the dataset's `source.json`, and the
evaluation refuses a base whose environment it cannot confirm. The re-recorded
demos in `~/libero_data_sysid/` are not read again; delete the `.hdf5` files
there if you need the space (keep the `.replay.json` files).

## 3. Send to Tillicum (workstation)

```bash
bash baselines/tillicum/sync.sh code
for S in $SUITES; do
  for I in $(tasks_of $S); do
    bash baselines/tillicum/sync.sh sim ~/franka_data/baseline_prep/${S}_$PLANT/task_$I
    bash baselines/tillicum/sync.sh lerobot ${S}_$PLANT/task_$I
  done
done
```

Sends the code the diffusion jobs run, then each task's `sail.hdf5`,
`bspline.hdf5` and LeRobot dataset, about 15 GB in all. It asks for your
password and a Duo push; one login covers 30 minutes of transfers, so a long
send may ask again. Running the block again resumes and skips what arrived.

## 4. Train SAIL, B-Spline and base dp (Tillicum login node)

```bash
T=/gpfs/scrubbed/$USER/tillicum; PREP=/gpfs/scrubbed/$USER/franka_home/franka_data/baseline_prep
PLANT=sysid_2026_09_16
export WANDB_API_KEY=<your wandb key>
for S in libero_10 libero_goal libero_spatial; do
  bash $T/submit_libero_train.sh ${S}_$PLANT --steps 100000 --apply \
      -- --extra --wandb --wandb-project franka-recipe            # SAIL and B-Spline, every task
  for d in $PREP/${S}_$PLANT/task_*; do
    bash $T/submit_diffusion_train.sh ${S}_$PLANT/${d##*/} --steps 100000 --name recipe-100k --apply \
        -- --ee-state=as-recorded --wandb.project=franka-recipe    # base dp, one task
  done
done
```

Each training is one Slurm job on one GPU, 90 in all. They run as GPUs free
up: `squeue` shows `PD` with reason `QOSMaxGRESPerUser` for jobs waiting on
your GPU limit. `--ee-state=as-recorded` is the one difference from the arm's
commands: the sim state already is the end-effector pose, so the job must not
convert joint angles into one. Leave out `--apply` to see what would be
submitted.

To track the jobs:

```bash
squeue -u $USER
for S in libero_10 libero_goal libero_spatial; do
  bash $T/progress.sh ${S}_$PLANT                                   # SAIL and B-Spline: epoch, speed, time left
  for d in $PREP/${S}_$PLANT/task_*; do bash $T/progress.sh diffusion ${S}_$PLANT/${d##*/}; done   # base dp
done
```

Loss curves are in the wandb project `franka-recipe`. Tillicum stops a job after
12 hours; submitting the same block again resumes unfinished training and skips
finished training.

## 5. Fetch the policies (workstation)

Once `progress.sh` says `finished` for every job:

```bash
for S in $SUITES; do
  for I in $(tasks_of $S); do bash baselines/tillicum/sync.sh fetch ${S}_$PLANT/task_$I; done
done
```

Copies the policies into `~/franka_data/policies/<suite>_<plant>/task_<i>/` and
the training logs into `baseline_prep/<suite>_<plant>/`.

## 6. Base dp's stats (workstation)

ZAP's teacher (step 7) needs base dp's success rate and episode length, and 20
of its successful paths as references, for every task.

```bash
cd ~/franka_ws/multi-fast
for S in $SUITES; do for I in $(tasks_of $S); do
  export DP_BASE_CKPT=$(base_of $S $I)
  dp_up "$DP_BASE_CKPT" || { dp_down; continue; }
  UV_NO_SYNC=1 ARM_FILTER=dp_recipe TASKS=${S}_$I EXTRA_OVERRIDES="plant=$PLANT" \
      OUT_DIR=logs/action_space_sweep/dp_recipe_${S}_${I}_$PLANT bash scripts/libero/eval_action_space_sweep.sh "" 50 10
  .venv/bin/python scripts/libero/sweep_to_base_stats.py logs/action_space_sweep/dp_recipe_${S}_${I}_$PLANT
  [ -f cfg/base_stats/libero/dp_recipe/$PLANT/$S/task_$I.yaml ] && \
      UV_NO_SYNC=1 MUJOCO_GL=egl bash scripts/utils/precompute_references.sh ${S}_dp libero.task_id=$I plant=$PLANT
  dp_down
done; done
grep -H avg_success_rate cfg/base_stats/libero/dp_recipe/$PLANT/libero_*/task_*.yaml
cd ~/franka_ws
```

1. **Base eval.** 50 episodes of base dp alone.
2. **Stats.** Writes `cfg/base_stats/libero/dp_recipe/<plant>/<suite>/task_<i>.yaml`.
   A task where base dp never succeeds gets no file and no teacher; ZAP is still
   evaluated on it in step 8.
3. **References.** `task_<i>_references.npz` beside the stats.

## 7. Train ZAP (Tillicum)

ZAP trains in three stages: an RL teacher for each task (24 h), 500 rollouts of
each teacher, and then one point-cloud student per suite distilled from them.
Every job that rolls base dp out starts its own copy of base dp's server.

Send multi-fast's code and step 6's stats, from the workstation:

```bash
bash baselines/tillicum/sync.sh multifast
```

Everything else in this step runs on the Tillicum login node. First these, in
any new terminal:

```bash
cd /gpfs/scrubbed/$USER/multi-fast
PLANT=sysid_2026_09_16
export WANDB_API_KEY=<your wandb key>
export DP_SERVER_SH=/gpfs/scrubbed/$USER/tillicum/dp_server.sh
# Base dp's checkpoint for <suite> <task index>, as the jobs see it.
ckpt_of() { echo /home/franka/franka_data/policies/${1}_$PLANT/task_$2/diffusion/recipe-100k/checkpoints/last/pretrained_model; }
# A teacher's id, from its Slurm job id.
teacher_of() { cat logs/slurm_logs/train_fast_$1/seed1/wandb_run_id.txt; }
```

**7a. Teachers**, one per task that has base stats. Run this once; it appends
each job to `teachers_dp_recipe.txt`, which the next steps read.

```bash
for S in libero_10 libero_goal libero_spatial; do
  for f in cfg/base_stats/libero/dp_recipe/$PLANT/$S/task_*.yaml; do
    I=$(basename $f .yaml); I=${I#task_}
    export DP_BASE_CKPT=$(ckpt_of $S $I)
    JOB=$(sbatch --parsable --export=WANDB_API_KEY,DP_SERVER_SH,DP_BASE_CKPT \
        scripts/cluster/run_tillicum.slurm online ${S}_${I}_dp plant=$PLANT policy.cumulative_residual=True)
    echo "$S $I ${JOB%%;*}" | tee -a teachers_dp_recipe.txt
  done
done
```

The teachers use the settings of the pi0.5 teachers ZAP was built on (`77bb0d2a`):
the sysid plant, variable impedance (the residual also sets the controller's
stiffness and damping) and a cumulative residual.

**7b. Collect** 500 rollouts of each teacher, once every teacher has finished
(`squeue -u $USER` shows no `train_fast`):

```bash
while read -r S I JOB; do
  mkdir -p data/distill/dp_recipe_${S}_p0916
  export DP_BASE_CKPT=$(ckpt_of $S $I)
  sbatch --export=WANDB_API_KEY,DP_SERVER_SH,DP_BASE_CKPT scripts/cluster/collect_distill_tillicum.slurm \
      ${S}_dp libero.task_id=$I plant=$PLANT teacher.id=$(teacher_of $JOB) distill.teacher_set=dp_recipe_${S}_p0916
done < teachers_dp_recipe.txt
```

Each writes `data/distill/dp_recipe_<suite>_p0916/<suite>_<i>.hdf5` when it is done.

**7c. Students**, one per suite, once collection has finished. Each also trains on
the reach data the current ZAP student (40cdf603) used, from the teacher set
`8_tasks_target_p0916_reach` in ecai0608's multi-fast folder. If your account
cannot read it, get `reach.hdf5` and `8_tasks_target_p0916_reach.yaml` from its
owner and point `REACH` at them.

```bash
REACH=/gpfs/scrubbed/ecai0608/multi-fast
REACH_ID=$(awk '/^singletons:/{s=1} s && $1 == "reach:" {print $2; exit}' \
    $REACH/cfg/teacher_sets/8_tasks_target_p0916_reach.yaml | tr -d "'")
for S in libero_10 libero_goal libero_spatial; do
  SET=dp_recipe_${S}_p0916
  cp $REACH/data/distill/8_tasks_target_p0916_reach/reach.hdf5 data/distill/$SET/
  {
    echo "notes: |"
    echo "  ZAP on the shared-recipe diffusion base (dp_recipe), $PLANT plant, comparison"
    echo "  gripper, variable impedance, cumulative residual; reach as 8_tasks_target_p0916_reach."
    echo "singletons:"
    echo "  reach: '$REACH_ID'"
    echo "suites:"
    echo "  $S:"
    while read -r s I JOB; do
      [ "$s" = "$S" ] && [ -f data/distill/$SET/${S}_$I.hdf5 ] && echo "    $I: '$(teacher_of $JOB)'"
    done < teachers_dp_recipe.txt
  } > cfg/teacher_sets/$SET.yaml
  cat cfg/teacher_sets/$SET.yaml
  sbatch scripts/cluster/train_distill_tillicum.slurm online policy=pointnet_xa2 data.teacher_set=$SET
done
```

`policy=pointnet_xa2` is 40cdf603's student architecture. The teacher set lists
only the tasks whose collection finished; a task left out is held out from that
student, which is fair to evaluate but worth saying when you report it.

**7d. The students' ids**, once they finish:

```bash
for S in libero_10 libero_goal libero_spatial; do
  echo "$S $(grep -l "teacher_set: dp_recipe_${S}_p0916" logs/distill/*/run_cfg.yaml | xargs -r ls -t | head -1 | cut -d/ -f3)"
done
```

Then fetch each one on the workstation, with the three ids that printed:

```bash
for id in <the three ids>; do bash baselines/tillicum/sync.sh student $id; done
```

## 8. Evaluate all four in sim (workstation)

Every condition runs 50 episodes per task, one from each of LIBERO's 50 start
states.

**SAIL at 5x, and B-Spline at 1x, 2x and 4x.** SAIL plays its fast steps at
100 Hz with OSC kp 300 and damping ratio 0.5 (`config/policy.yaml`); B-Spline
runs at the plant's kp 150 and damping ratio 1. The four settings of a suite run
at once, each with its own policy-server port:

```bash
for S in $SUITES; do
  P=~/franka_data/baseline_prep/${S}_$PLANT
  python -m baselines.libero_bridge.evaluate $P --backend sail --num-episodes 50 --save-video --resume \
      --sweep-id ${RUN}_sail_5x --log-dir $P/rollout_logs_${RUN}_sail_5x \
      --extra --port 5560 > $P/eval_${RUN}_sail_5x.log 2>&1 &
  for x in 1 2 4; do
    python -m baselines.libero_bridge.evaluate $P --backend bspline --num-episodes 50 --save-video --resume \
        --sweep-id ${RUN}_bsp_${x}x --log-dir $P/rollout_logs_${RUN}_bsp_${x}x \
        --extra --port 556$x --speed-up-times $x > $P/eval_${RUN}_bsp_${x}x.log 2>&1 &
  done
  wait
done
```

`tail ~/franka_data/baseline_prep/*_$PLANT/eval_${RUN}_*.log` shows progress.
`--resume` makes running the block again skip what finished.

**Base dp + ZAP, and base dp alone.** Both run through multi-fast's own eval, so
the only difference between them is the residual. Each suite is evaluated with
the newest student fetched for it in step 7d:

```bash
declare -A STUDENT
for S in $SUITES; do
  STUDENT[$S]=$(grep -l "teacher_set: dp_recipe_${S}_p0916" multi-fast/logs/distill/*/run_cfg.yaml \
      | xargs -r ls -t | head -1 | cut -d/ -f4)
  echo "$S: ${STUDENT[$S]}"                      # each suite must print an id
done
cd ~/franka_ws/multi-fast
for S in $SUITES; do for I in $(tasks_of $S); do
  export DP_BASE_CKPT=$(base_of $S $I)
  dp_up "$DP_BASE_CKPT" || { dp_down; continue; }
  for base in False True; do
    UV_NO_SYNC=1 MUJOCO_GL=egl bash scripts/distill/eval.sh ${S}_dp libero.task_id=$I plant=$PLANT \
        libero.comparison_env=true n_eval=50 env.n_envs=10 eval_base=$base student.id=${STUDENT[$S]}
  done
  dp_down
  D=multi-fast/logs/distill/${STUDENT[$S]}/${S}_$I
  (cd ~/franka_ws
   python -m baselines.libero_bridge.import_eval_distill $D/comparison.json --sweep-id ${RUN}_zap
   python -m baselines.libero_bridge.import_eval_distill $D/comparison_base.json --sweep-id ${RUN}_dp)
done; done
cd ~/franka_ws
```

`libero.comparison_env=true` puts multi-fast's eval in the same environment and
start states as SAIL and B-Spline. The importer files each run beside theirs,
tagged with the sweep id the report reads. Add `save_video=True` to the eval for
a video of each env's first episode, under the student's folder.

## 9. Report

```bash
OUT=~/franka_data/analysis/libero_${PLANT}_${RUN}_$(date +%Y%m%d)
./scripts/collect_rollout_report.sh $OUT \
    ${RUN}_sail_5x ${RUN}_bsp_1x ${RUN}_bsp_2x ${RUN}_bsp_4x ${RUN}_dp ${RUN}_zap
for S in $SUITES; do python scripts/rollout_summary.py ${S}_$PLANT > $OUT/$S.runs.txt; done
(cd $(dirname $OUT) && zip -qr $(basename $OUT).zip $(basename $OUT))
echo "scp franka@deepblue:$OUT.zip ."
```

The last line prints the command that downloads the zip to your PC.

| in the folder | what it is |
|---|---|
| `comparison.<suite>.png` | the six conditions side by side, one sheet per suite: pooled success, time to success, replans per step, success per task, wrist force per task |
| `<sweep>.<suite>.png` | one sheet per condition and suite: summary tiles, outcome per task, every episode's duration and wrist force, episode-length and peak-force histograms, every episode's force over time, and the run's settings |
| `<sweep>.summary.txt` | the condition's table per task, pooled per suite and over all three |
| `<suite>.runs.txt` | every run of a suite, one row each, tagged with its environment |
| `index.json` | each run's success rate, times, force and video counts |
| `videos/<sweep>/<task>/{success,timeout}/` | SAIL's and B-Spline's episode videos |

Success is LIBERO's own check that the task is done. Time to success counts
successful episodes only, in simulated seconds. Wrist force is the gripper's
wrist sensor; about 5 N at rest is the gripper's own weight.

## When something fails

| symptom | cause |
|---|---|
| `No space left on device` | see [Disk](#what-it-costs); delete the re-recorded `.hdf5` files once step 2 is done |
| step 1 keeps few demos | the plant is slower than the stock one, and some demos no longer succeed; `<task>_demo.replay.json` lists each demo's outcome |
| step 2: `... exists; delete it to convert over it` | that task's LeRobot dataset was converted already. Delete `~/.cache/huggingface/lerobot/<suite>_<plant>/task_<i>` or add `--skip lerobot`, then re-run the gripper labels, since `sail.hdf5` was rewritten |
| step 4: `expected [7 joints, gripper] states, got width 9` | `--ee-state=as-recorded` is missing from the diffusion job |
| step 4, submit: `no .../franka_src/...` | step 3's `sync.sh code` has not run |
| `progress.sh`: `FAILED: ...` | it prints the log path; the troubleshooting tables are in [LIBERO_SIM.md](baselines/LIBERO_SIM.md#training-on-tillicum) |
| `no reply from the dp policy server`, or `dp_up` returns without `ready` | the server did not start; read `~/franka_data/dp_server.log` |
| `dp_server.log`: `Address already in use` | a server from an earlier loop is still up: `kill $DP_PID`, or `ss -ltnp \| grep 5557` for its pid |
| `lerobot_dp base: the server holds ..., not base_policy.checkpoint=...` | `DP_BASE_CKPT` and the server name different checkpoints |
| `lerobot_dp base: ... recorded in a different environment` | `plant=` is missing, the gripper differs from the demos', or the LeRobot dataset was deleted |
| step 6 writes no stats for a task | base dp never succeeded on it; it gets no teacher |
| step 6 after retraining base dp prints `[have]` and `have ...` | the earlier base dp's eval and stats are still there. Delete `multi-fast/logs/action_space_sweep/dp_recipe_<suite>_<i>_<plant>/` and add `--force` to `sweep_to_base_stats.py` |
| `No base stats at cfg/base_stats/libero/dp_recipe/...` | step 6 has not run for this task, or its stats did not reach Tillicum (`sync.sh multifast`) |
| `traj_deviation_lambda > 0 requires ... references` | step 6's `precompute_references.sh` |
| a Slurm job: `no .../lerobot-dp-server.sif` | first-time step 2 |
| `..._dp names no task id to find its dp checkpoint by` | `DP_BASE_CKPT` is not set in this shell |
| `n_eval=... must be a multiple of env.n_envs` | keep `n_eval=50 env.n_envs=10`, so that each start state runs exactly once |
| `env ...'s episode ... started at state ..., not ...: a reset outside the schedule` | something reset an eval env an extra time; the run's start states no longer match the baselines', so its numbers cannot be used |
| `the policy's demos were recorded in ..., but this rollout runs in ...` | the policy was trained on another plant's demos |
| `rollout_report.py`: `no runs tagged ...` | the eval was imported without `--sweep-id`. Delete that run folder under `~/franka_data/outputs/` and import again with it |
| a base dp or ZAP sheet says force `not recorded` | its `comparison.json` predates 2026-10-29, when eval_distill started writing the wrist force |
| `rollout_summary.py` ends with `NOT COMPARABLE` | some runs predate the shared environment; compare only rows from the same one |
