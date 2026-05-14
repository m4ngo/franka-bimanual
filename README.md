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
- gripper IP at 192.168.2.20

left franka
- controller IP at 192.168.200.2
- gripper IP at 192.168.2.21

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
- Before even running teleop, if its the first time running teleop after turning on the robots, calibrate the GELLOs
	- *not filled in yet*
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

train
- for training a policy on a specific huggingface dataset
- run `./~/franka_ws/scripts/train.sh <repo_id> <policy_repo_id> <batch_size> <steps>`
- You'll need a hugging face model which the trained model will be uploaded to. For example, I used `HuskyMango/test_act` when I was training a test ACT model
- once the script runs you sorta just let it rip

roll out
- for rolling out a policy on a specific huggingface model
- run `./~/franka_ws/scripts/rollout_policy.sh <repo_id> <number_of_episodes>  <policy_repo_id> <output_dir>`
- In this case, `<repo_id>` is the repo you want the trajectory to be uploaded to, which can be helpful for debugging or evaluating the policy
- `output_dir` is recommended to be `~/franka_data/policy/eval/#` where `#` is any name that isn't taken yet in the folder. Record won't work if there is already an existing folder at the given `output_dir`

## Common errors
teleop
- Sometimes, teleop will die on its own because of 'UDP timeout'. This usually indicates an error that happened with the arms but wasn't sent to the teleop. Check the SSH for Mario and Luigi, where there is likely to be a more descriptive error mode
- teleop will also die if a rough collision occurs. This is less common, since the force and torque tolerances for the TRI setup have been set relatively high. If this occurs, check the Franka UI for instructions. If there is no issues, teleop can be run again as normal. If there are issues, Franka UI may require a manual recalibration
- If the teleop ends in a poor position, such as the arms being in a dangerous pose to each other (like wrapped around each other) you can manually move the arms back to a safe position using the Franka UI. With the arms unlocked, set them from 'Execution' mode to 'Program'. Then, you can lightly squeeze the black buttons on the end effector together and slowly guide the arm back to position
- gripper issues: still working on it. The grippers are very unresponsive at the moment...