# Record, train, and evaluate: SAIL, B-Spline, and ZAP

Guide to running task evaluation loop.

```
record dataset → filter dataset + convert to baselines → send dataset to 
Tillicum → train on Tillicum → fetch trained policies from Tillicum → evaluate
policies on the arm → generate report in form of .zip file
```

Our method is called **ZAP**, which applies a residual policy to a base 
diffusion policy ("base dp"). We compare against two baselines: SAIL and
B-Spline. All three methods are trained on the same filtered dataset, with
nearly identical policy architecture (see `baselines/diffusion_recipe.py`).

## Set some constants

You can set the constants below in your terminal, since it makes the later
commands less verbose and easier to read. Also reduces risk of messing something
up.

```bash
cd ~/franka_ws && source .venv/bin/activate
TUSER=[YOUR TILLICUM USERNAME GOES HERE]      # your Tillicum (UW NetID) username
NAME=[TASK NAME GOES HERE]             # name of the new task
TASK="semantic task instruction goes here"              # task instruction
HOME_POSE=home_pose_raised_19cm      # home_poses/<name>.json, where every episode starts
REPO=HuskyMango/filter-$NAME         # the filtered recording all three methods train on
```

## Workstation folder structure

```
~/franka_data/
  <name>/                                  the raw recording (a LeRobot dataset)
  baseline_prep/real/task_filter-<name>/   SAIL's and B-Spline's training data: sail.hdf5, bspline.hdf5
  baseline_prep/real/train_logs/           SAIL and B-Spline training logs, fetched from Tillicum
  policies/HuskyMango/filter-<name>/       trained policies, fetched from Tillicum
    diffusion/recipe-200k/                   ZAP's base diffusion policy
    sail/<timestamp>/                        SAIL
    bspline/<timestamp>/                     B-Spline
  outputs/HuskyMango/filter-<name>/        one folder per evaluation run, <timestamp>-<method>:
                                           sail, bspline, or multifast (base dp, with or without ZAP)
  analysis/filter-<name>_real_eval_<date>/ the report and .zip
~/.cache/huggingface/lerobot/HuskyMango/filter-<name>/   the filtered recording
~/checkpoints/<id>/checkpoint/best.pt                    ZAP residuals (trained in sim, multi-fast)

Tillicum, /gpfs/scrubbed/<user>/
  tillicum/      the job scripts             franka_src/   the code the diffusion job runs
  containers/    the two training images     franka_home/  mirrors ~ above (franka_data/, .cache/)
```

## 1. Record

```bash
./scripts/single_arm_record_data_homed.sh HuskyMango/$NAME 30 "$TASK" ~/franka_data/$NAME false \
    $HOME_POSE spacemouse_ee false single_arm_right
```

Above script records 30 episodes with the SpaceMouse controlling the right arm.
Each episode starts from `$HOME_POSE`.
- **Right arrow** ends an episode.
- **Left arrow** re-records it.
- **Esc** stops.
- Between episodes, reset the scene and press **Enter** to initiate next episode.

To add to an interrupted recording, change `false` after the output folder to `true`.

## 2. Filter and convert

```bash
python scripts/filter_noop_actions.py --source-repo-id HuskyMango/$NAME --target-repo-id $REPO --dry-run
python scripts/filter_noop_actions.py --source-repo-id HuskyMango/$NAME --target-repo-id $REPO
python scripts/prepare_baseline_datasets.py --source-repo-id $REPO --skip sysid \
    --out-dir ~/franka_data/baseline_prep/real/task_filter-$NAME
python -m baselines.sail_bridge.gripper_labels ~/franka_data/baseline_prep/real/task_filter-$NAME/sail.hdf5
```

1. **Dry run.** Prints how many frames each episode keeps.
2. **Filter.** Creates a filtered version of the dataset; drops frames where
   the target pose didn't move.
3. **Convert.** Turns the filtered recording into HDF5 format specific for
   training SAIL and B-Spline.
4. **Label SAIL's precise steps.** Labels the specific steps in the dataset as
   'precise,' which is how SAIL learns when to move fast/slow.

## 3. Send to Tillicum

```bash
bash baselines/tillicum/send_datasets.sh $TUSER ~/franka_data/baseline_prep/real/task_filter-$NAME
```

Sends four things to `/gpfs/scrubbed/$TUSER`: `sail.hdf5`, `bspline.hdf5`,
the filtered recording, and the diffusion job's code. It will ask for password
and Duo push verification. If the dataset send is ever interrupted, running it
again should resume from wherever it stopped.

**First time only.** Pull the two docker images on a Tillicum compute node. Make
sure to pull again after rebuilding the baselines image, which is needed whenever
`baselines/` changes ([LIBERO_SIM.md](baselines/LIBERO_SIM.md#a-real-recording-all-three-methods-on-the-shared-recipe)).

```bash
ssh $TUSER@tillicum.hyak.uw.edu
salloc --qos=debug --gpus=1 --cpus-per-task=8 --mem=64G --time=00:30:00
export APPTAINER_CACHEDIR=/gpfs/scrubbed/$USER/apptainer_cache APPTAINER_TMPDIR=/gpfs/scrubbed/$USER/tmp
mkdir -p /gpfs/scrubbed/$USER/containers $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
apptainer pull --force /gpfs/scrubbed/$USER/containers/franka-baselines.sif docker://atsai06/franka-baselines:latest
apptainer pull --force /gpfs/scrubbed/$USER/containers/lerobot-train.sif docker://atsai06/lerobot-act-train:v0.5.1
exit
```

## 4. Train on Tillicum

Run these on the Tillicum login node (`ssh $TUSER@tillicum.hyak.uw.edu`), with
the same `NAME` as above:

```bash
NAME=[TASK NAME GOES HERE]; REPO=HuskyMango/filter-$NAME; T=/gpfs/scrubbed/$USER/tillicum
export WANDB_API_KEY=<your wandb key>
bash $T/submit_libero_train.sh real --tasks filter-$NAME --steps 200000 --apply \
    -- --extra --wandb --wandb-project franka-recipe        # SAIL and B-Spline, one GPU each
bash $T/submit_diffusion_train.sh $REPO --steps 200000 --name recipe-200k --apply \
    -- --wandb.project=franka-recipe                        # ZAP's base diffusion policy
```

If you just want to see what the commands will print without actually running
them, just remove `--apply` from the arguments.

To track the jobs:

```bash
squeue -u $USER                       # R: running, PD: waiting (the REASON column says why)
bash $T/progress.sh real              # SAIL and B-Spline: epoch, speed, time left
bash $T/progress.sh diffusion $REPO   # ZAP's base: step, speed, time left
```

You can check loss curves in the wandb project `franka-recipe`. Small issue 
where Tillicum will stop the jobs after 12 hours, so helpful to check occasionally.
Submitting the same command again resumes any unfinished training from its last
checkpoint and skips the finished ones.

## 5. Fetch the policies

Once `progress.sh` says `finished` for all three, run this on the workstation:

```bash
bash baselines/tillicum/fetch_policies.sh $TUSER $REPO
```

It copies the policies and their logs into `~/franka_data/policies/$REPO/` and
prints the newest checkpoint of each. It's helpful to note down what directory
each policy was copied to, since theres a lot of them and may get lost easily.

## 6. Evaluate on the arm

First choose the checkpoints. These lines take the newest of each. To evaluate a
different checkpoint, set that variable to its path.

```bash
P=~/franka_data/policies/$REPO
BASE=$(ls -td $P/diffusion/*/checkpoints/last | head -1)/pretrained_model   # ZAP's base diffusion policy
SAIL=$(ls -t $P/sail/*/models/model_epoch_*.pth | head -1)
BSP=$(ls -t $P/bspline/*/checkpoints/latest.ckpt | head -1)
ZAP=$(ls -t ~/checkpoints/*/checkpoint/best.pt | head -1)                  # the ZAP residual
printf '%s\n' "$BASE" "$SAIL" "$BSP" "$ZAP"
```

Each run below is 20 episodes of up to 30 s, all starting from `$HOME_POSE`.
During an episode:
- **Right arrow** marks it a success.
- **Left arrow** marks it a failure.
- **Up arrow** throws the attempt away and runs it again.
- **Ctrl-C** aborts the whole run.

**SAIL at 5x.** Fast steps run at 100 Hz and precise steps at 1x. It uses OSC kp
300 and damping 0.5, set in `config/policy.yaml`.

```bash
./scripts/sail_rollout.sh --start-server --ckpt "$SAIL" --speed 5 \
    --rig=single_arm_right --home-pose-name $HOME_POSE --num-episodes 20 --episode-time-s 30
```

**B-Spline at 1x.**

```bash
./scripts/bspline_rollout.sh --start-server --ckpt "$BSP" --speed 1 \
    --rig=single_arm_right --home-pose-name $HOME_POSE --num-episodes 20 --episode-time-s 30
```

**B-Spline at 2x.**

```bash
./scripts/bspline_rollout.sh --start-server --ckpt "$BSP" --speed 2 \
    --rig=single_arm_right --home-pose-name $HOME_POSE --num-episodes 20 --episode-time-s 30
```

**B-Spline at 4x.**

```bash
./scripts/bspline_rollout.sh --start-server --ckpt "$BSP" --speed 4 \
    --rig=single_arm_right --home-pose-name $HOME_POSE --num-episodes 20 --episode-time-s 30
```

**Base dp + ZAP.**

```bash
python residual_wrapper/run_residual.py --base-policy "$BASE" --residual-policy "$ZAP" \
    --rig=single_arm_right --home-pose-name $HOME_POSE --num-episodes 20 --episode-time-s 30
```

**Base dp alone.**

```bash
python residual_wrapper/run_residual.py --base-policy "$BASE" --no-residual \
    --rig=single_arm_right --home-pose-name $HOME_POSE --num-episodes 20 --episode-time-s 30
```

Each run writes its own folder, `~/franka_data/outputs/$REPO/<timestamp>-<method>/`.
A run that stops early does not resume, so start it again.

## 7. Report and download

```bash
python scripts/package_real_eval.py $REPO
```

The script does four things:
- Picks the newest run with at least 20 episodes for each of the six conditions.
- Prints which run it took for each, with its checkpoints.
- Writes the report folder `~/franka_data/analysis/filter-<name>_real_eval_<date>/`.
- Zips that folder.

| in the zip | what it is |
|---|---|
| `summary.txt` | success, time and checkpoint for each run |
| `charts/` | one sheet per run, and `comparison.png` across all six |
| `stats/` | `runs.json`, plus each run's `manifest.json` and `episodes.jsonl` |
| `viz/<run>/` | an interactive 3D page per episode: the arm, its path and the policy's plans (open in a browser) |
| `videos/<run>/` | each episode's camera clip, named by its outcome, plus `all_episodes.mp4` |

To skip older runs, add `--since 20261020`. To choose runs by hand, add
`--runs <run folder>="bsp 1x" ...`.

For bowl-plate's six 20-episode runs, the script took about a minute and the zip
was 424 MB. The last line it prints is the download command to run on your PC:

```bash
scp franka@deepblue:/home/franka/franka_data/analysis/filter-<name>_real_eval_<date>.zip .
```

## When something fails

| symptom | cause |
|---|---|
| `send_datasets.sh`: `no sail.hdf5 ...` | step 2's convert line has not run |
| submit: `no .../franka_src/...` | the code is out of date; run `send_datasets.sh` again |
| `squeue` shows `PD` for hours | read the REASON column: `QOSMaxGRESPerUser` is your GPU cap, so your other jobs have to finish first |
| `progress.sh`: `FAILED: ...` | it prints the log path; the troubleshooting tables are in [LIBERO_SIM.md](baselines/LIBERO_SIM.md#a-real-recording-all-three-methods-on-the-shared-recipe) |
| a SAIL or B-Spline rollout refuses the home pose | `$HOME_POSE` is not where the recording started; set it to the pose you recorded from |
| `package_real_eval.py`: `no ... run with at least 20 episodes` | that condition has no finished 20-episode run yet; the report covers the rest |
