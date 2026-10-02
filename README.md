Details procedures, common errors, and codebase structure for working with the TRI setup.
## Machines
workstation: franka@deepblue (on tailscale)
- username: franka
- password: weirdl@123

NUC (right arm controller)
- username: mario
- password: weirdl@123
- can be SSH from workstation through mario@192.168.3.10

NUC (left arm controller)
- username: luigi
- password: weirdl@123
- can be SSH from workstation through luigi@192.168.3.11

right franka
- controller IP at 192.168.201.10
- SCHUNK gripper IP at 192.168.2.20

left franka
- controller IP at 192.168.200.2
- SCHUNK gripper IP at 192.168.2.21

camera IPs
- "192.168.0.142": "BFS_23595723"
- "192.168.0.116": "FRAMOS_D71"
- "192.168.1.138": "BFS_23595719"
- "192.168.1.139": "BFS_23595720"
- "192.168.1.143": "BFS_23595724"
- "192.168.1.102": "FRAMOS_D63",

## Setup steps
These are steps that should be taken before running any teleop, recording, etc. Required for working with the arms

Enable Franka control interface (FCI)
- Turn on both Franka control boxes (black box beneath each arm)
- wait for frankas to start up
- From the workstation, open chrome and enter both franka IPs
- chrome may say it's 'unsafe' website, just ignore and continue
- unlock both Franka arms and make sure they are in 'execution' mode
- After arms are unlocked, go to the dropdown in the top right, click 'enable FCI'
- Once both frankas have the FCI enabled, they will have green lights indicated ready to run

Start controllers
- SSH into luigi and mario
- ensure mario can ping 192.168.201.10 and luigi can ping 192.168.200.2
- run `./start_control.sh`

If the arms ever error out, make sure to check
- That the control scripts on Mario and Luigi didn't crash. If they did, just re-run the scripts
- That the Franka UI didn't reach some 'unrecoverable' fault (very rare)

Once both loops are running and both arms are unlocked with FCI enabled, setup is ready to run

## Files/folders to note
Important folders on the workstation
- `~/franka_data` is where we store all the data from rollouts, datasets, etc. Please don't store those things in the git repo since they are big files
- `~/franka_ws` is the actual workspace where you run all the scripts you want
	- all the things titled `lerobot_*` are the lerobot packages/wrappers used to interface between the various teleops/robots/cameras and the lerobot scripts
	- `~/franka_ws/scripts` contains the scripts used to roll out polices, record, teleop, etc.

## Quick start with scripts
Before running any scripts, ensure you have the right environment activated: `source ~/.venv/bin/activate`

Teleop
- Before even running teleop, if its the first time running teleop after turning on the robots, you may need to calibrate the GELLOs
	- Sometimes the calibration is saved, but if the GELLOs got turned off last time or disconnected, they may become uncalibrated
	- Calibration can be done using the script in /scripts/old/calibrate_gello_teleop.py
    - The GELLOs home position looks like this: (note that the joints are all straight except the 90 degree angle in the elbow and the gripper is fully open) <img width="1344" height="1008" alt="IMG_7486" src="https://github.com/user-attachments/assets/156e1dd7-5d33-4302-948f-51686f9cdebe" />

- Ensure setup is ready. MAKE SURE THE GELLOs AREN'T COLLIDING. It can be helpful to prop them on a box in their 'default position' to ensure nothing goes wrong when teleop starts
- Run `./~/franka_ws/scripts/teleop.sh`
- Teleop should start automatically, control using the GELLOs
- end teleop with `ctrl+C` on the script

recording
- Run `./~/franka_ws/scripts/record_data.sh <repo_id> <number_of_episodes> <task_name> <output_dir>`
- `repo_id` is the huggingface dataset you're recording to. For example, I used `HuskyMango/test` when I was testing. This should be a repo that you have write access to
	- Also make sure you are logged into the right hugging face account using `hf login`
	- or logout if there is another huggingface user already logged in
- `output_dir` is recommended to be `~/franka_data/data/#` where `#` is any name that isn't taken yet in the folder. Record won't work if there is already an existing folder at the given `output_dir`
- The recording uses the GELLOs to teleop
- Once recording starts, Rerun Viewer will also open. Pressing the right arrow key while focused on Rerun Viewer ends an episode, and pressing it again starts the next episode
- Once all episodes are done, recording will automatically complete and uploaded to the given repository on HuggingFace

replay
- for replaying a specific episode from a huggingface dataset
- Run `./~/franka_ws/scripts/replay.sh <repo_id> <episode_number>

train (one policy)
- for training the diffusion base policy alone on a specific huggingface dataset
- run `./scripts/training/train_diffusion.sh <repo_id> <policy_repo_id> <batch_size> <steps> <resume>`
- You'll need a hugging face model which the trained model will be uploaded to. For example, I used `HuskyMango/test_act` when I was training a test ACT model
- once the script runs you sorta just let it rip

train (the whole comparison: diffusion base policy + SAIL + B-Spline)
- one yaml describes the run: the recording, a run name, wandb, and each trainer's knobs. Copy the template and edit it:
  - `cp pipelines/example.yaml pipelines/<name>.yaml`
  - set `dataset` (the LeRobot repo id, found under `~/franka_data` or the HF cache), `name`, the epochs/steps, and `diffusion.policy_repo_id` (the Hub model the base policy is pushed to, or `null` to skip the push)
  - a misspelt key is an error, not a silent default
- start it: `python scripts/train_pipeline.py start pipelines/<name>.yaml`
  - converts the recording into the sysid / SAIL / B-Spline HDF5s in the foreground (`scripts/prepare_baseline_datasets.py`), then launches all three trainings detached, so you can close the terminal
  - `parallel: false` in the yaml runs them one after another instead of all at once (three at once takes ~31 GB of the 32 GB GPU at the defaults, and the last one to reach the GPU can run out of memory -- see `retry` below)
  - nothing trains from scratch while an earlier run holds its work: for each stage, if an earlier run of the same recording under the same root ran the same training (`STAGE_IDENTITY` in the script lists the settings that count -- the `diffusion` knobs for the base policy; the `convert` section plus the stage's own knobs for SAIL and B-Spline), a finished one is linked in instead of trained again (`<stage>/` and `logs/<stage>.log` point at the earlier run; `status` shows `reused` with a `from` line), and otherwise the newest checkpoint of a stopped or failed one is copied in and the training resumes from it (`status` shows the `from` line and `resuming`). Diffusion resumes exactly (lerobot restores step, optimizer and schedule; it carries on the earlier wandb run, or runs without wandb if that run was deleted, since lerobot can only resume into the run it started). B-Spline resumes exactly (model, EMA, optimizer, epoch, step; `baselines/bspline_bridge/bspline_train.py` keeps the epoch budget and learning-rate schedule right). SAIL resumes the model and EMA at the epoch count; robomimic saves no optimizer state, so Adam re-warms over a few hundred steps at the same constant learning rate. `start --retrain <stage>` (repeatable, or `all`) trains from scratch regardless
  - `start --dry-run` prints the exact commands without running anything; `start --no-train` only converts
- watch it: `python scripts/train_pipeline.py status` (the newest run under `~/franka_data/pipeline`, or pass a run directory)
  - per stage: state, pid, elapsed, progress (`step 54800/200000 loss 0.012`, `epoch 31/501`, SAIL's precise fraction), newest checkpoint, wandb link, log path
  - a failed stage also shows its `error` (the last exception line of its log, e.g. `torch.OutOfMemoryError: CUDA out of memory ...`) and the `retry` command for it
  - `python scripts/train_pipeline.py summary` writes `summary.png` (the three loss curves over the same table) at any time; the runner writes it itself when the last training ends
  - `python scripts/train_pipeline.py stop` kills the runner and every training it started
- if a stage fails: `python scripts/train_pipeline.py retry <stage>` (newest run, or pass the run directory)
  - the stage's log moves to `logs/<stage>.attempt<N>.log` and it is launched again from the last checkpoint it left under the run (diffusion's `checkpoints/last`, B-Spline's `checkpoints/latest.ckpt`, SAIL's newest `model_epoch_N.pth`; SAIL's labelling of the HDF5 is kept too)
  - while other stages are still training the retry waits for them to end, since the three share the GPU; `--now` launches at once
  - a stage that ran out of GPU memory while the others were training is requeued this way once on its own, so a `parallel: true` run with one OOM finishes by itself
  - the runner takes the request within a few seconds; if it has exited (or predates `retry`), a new one is started and takes over once the old one is gone
- everything for one run lives in one directory:
  ```
  ~/franka_data/pipeline/<dataset>/<timestamp>-<name>/
    config.yaml  pipeline.json  links.md  summary.png
    datasets/{sysid,sail,bspline}.hdf5
    diffusion/checkpoints/last/pretrained_model     run_residual.py --base-policy
    bspline/<ts>/checkpoints/latest.ckpt            bspline_rollout.sh --ckpt
    sail/<ts>/models/model_epoch_N.pth              sail_rollout.sh --ckpt
    logs/{convert,diffusion,bspline,sail,pipeline}.log
  ```
- notes
  - every run converts its own copy of the HDF5s (~2.2 GB at 224x224) on purpose: SAIL's labelling writes into the file in place, so two runs with different `err_threshold`s must not share one
  - checkpoints are large (about 1 GB each); keep `save_freq` / `checkpoint_every` / `save_every` at the template's values unless you want dozens of them
  - wandb: `wandb.enable: false` turns it off for all three; the SAIL trainer needs a wandb entity and takes your `wandb login` default unless `wandb.entity` is set
  - rolling the results out is described under `baselines/ROLLOUT.md`; the short version is below

roll out (the three methods)
- SAIL: `./scripts/sail_rollout.sh --start-server --ckpt <run>/sail/<ts>/models/model_epoch_N.pth --guide-config baselines/sail/robomimic/SAIL/guide_template/base_cfg_weight_1.json --rig=single_arm_right --num-episodes 10`
- B-Spline: `./scripts/bspline_rollout.sh --start-server --ckpt <run>/bspline/<ts>/checkpoints/latest.ckpt --rig=single_arm_right --speed-up-times 1.0 --num-episodes 10`
- ours: `python residual_wrapper/run_residual.py --base-policy <run>/diffusion/checkpoints/last/pretrained_model --residual-policy best.pt --num-episodes 10` (`--no-residual` runs the base policy alone)
- before the first rollout of a new baseline checkpoint, start its server on its own and run `python scripts/check_policy_server.py sail|bspline` (a handshake plus one synthetic inference, no arm)
- each run files itself under `~/franka_data/outputs/<train-dataset>/<timestamp>-<method>/`; `python scripts/rollout_summary.py <train-dataset>` prints every method's success rate and time side by side

roll out
- for rolling out a policy on a specific huggingface model
- run `./~/franka_ws/scripts/rollout_policy.sh <repo_id> <number_of_episodes>  <policy_repo_id> <output_dir>`
- In this case, `<repo_id>` is the repo you want the trajectory to be uploaded to, which can be helpful for debugging or evaluating the policy
- `output_dir` is recommended to be `~/franka_data/policy/eval/#` where `#` is any name that isn't taken yet in the folder. Record won't work if there is already an existing folder at the given `output_dir`

### Single-arm (right arm only)

These scripts control only the right Franka arm (mario NUC, `192.168.201.10`). For single-arm work you only need the right control box powered, FCI enabled, and `./start_control.sh` running on mario — the left arm and luigi NUC can stay off.

Hardware used by the single-arm wrapper:
- Right GELLO: `/dev/ttyUSB0`
- Right SpaceMouse: `/dev/hidraw3`
- Cameras: right wrist (`cam_3_wrist`, `cam_4_wrist`) and workspace scene (`cam_2_scene`)

teleop
- One script, four leader/mode pairings: `~/franka_ws/scripts/single_arm_teleop.sh <mode>`
  - `spacemouse_delta` (default) — SpaceMouse, EE_DELTA
  - `spacemouse_ee` — SpaceMouse, EE_POS (target seeded from the arm's real pose on start)
  - `gello_ee` — GELLO, EE_POS via FR3 forward kinematics
  - `gello` — GELLO, JOINT_POS
- The control mode follows from the mode; there is no separate flag for it.
- End teleop with `ctrl+C`

recording (homed)
- Each episode starts by driving the arm to a saved home pose in `~/franka_ws/home_poses/`. Pose files are JSON with `r_q` (7 joint angles) and `gripper` (0=closed, 1=open); see `home_poses/home_pose.json` for the default.
- Homed recording, same four modes as teleop:
  - `~/franka_ws/scripts/single_arm_record_data_homed.sh <repo_id> <num_episodes> <task> <output_dir> <resume> <home_pose_name> [spacemouse_delta|spacemouse_ee|gello_ee|gello] [depth]`
  - Example: `~/franka_ws/scripts/single_arm_record_data_homed.sh HuskyMango/test_single 10 pick_block ~/franka_data/data/test_single false home_pose gello_ee true`
- `home_pose_name` is the filename stem (e.g. `home_pose` loads `home_poses/home_pose.json`)
- `resume` is `true` to append to an existing local dataset, `false` for a fresh run
- `depth` is `true` (default) to include depth point-cloud observations, or `false` for RGB-only
- During recording, right arrow ends the current episode early, left arrow re-records it, and Escape stops the session. Press Enter in the terminal between episodes when prompted.

rollout (homed)
- Rolls out a pretrained policy on the right arm; each episode homes the arm first.
- `~/franka_ws/scripts/single_arm_rollout_policy_homed.sh <repo_id> <num_episodes> <policy_repo_id> <output_dir> <home_pose_name> [control_mode] [depth]`
- Example: `~/franka_ws/scripts/single_arm_rollout_policy_homed.sh eval_test 5 HuskyMango/test_act ~/franka_data/policy/eval/test_single home_pose EE_POS true`
- `<repo_id>` should start with `eval_` (LeRobot dataset naming convention for eval rollouts)
- `control_mode` is `JOINT_POS`, `EE_POS` (default), or `EE_DELTA` and must match how the policy was trained

## Common errors
teleop
- Sometimes, teleop will die on its own because of 'UDP timeout'. This usually indicates an error that happened with the arms but wasn't sent to the teleop. Check the SSH for Mario and Luigi, where there is likely to be a more descriptive error mode
- teleop will also die if a rough collision occurs. This is less common, since the force and torque tolerances for the TRI setup have been set relatively high. If this occurs, check the Franka UI for instructions. If there is no issues, teleop can be run again as normal. If there are issues, Franka UI may require a manual recalibration
- If the teleop ends in a poor position, such as the arms being in a dangerous pose to each other (like wrapped around each other) you can manually move the arms back to a safe position using the Franka UI. With the arms unlocked, set them from 'Execution' mode to 'Program'. Then, you can lightly squeeze the black buttons on the end effector together and slowly guide the arm back to position
- gripper issues: still working on it. The grippers are very unresponsive at the moment...
