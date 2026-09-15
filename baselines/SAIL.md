# SAIL — repo overview

Upstream: <https://github.com/nadunRanawaka1/SAIL>, submodule at [sail/](sail/),
pinned at `1440c16` ("Initial public release"). We do not edit it. What we feed
it is described in [README.md](README.md); this file is a map of the repo itself.

## The idea

SAIL (*Speed Adaptation for Imitation Learning*) runs an ordinary visuomotor
policy faster than the demonstrations it was trained on, without losing success
rate. It does not retrain for speed. It adds four things around a stock
Diffusion Policy:

1. **Absolute actions.** Replay each demo to recover the pose the arm actually
   reached, and train on absolute goal poses rather than deltas — a delta means
   something different at a different control rate.
2. **Precision labels.** Extract waypoints with AWE, cluster them, and label
   each timestep precise or not. The label rides along as one extra action
   dimension.
3. **Speed modulation at rollout.** Execute at `fast_control_freq` (100 Hz)
   normally and drop to `slow_control_freq` (20 Hz) inside a window around any
   step whose precision label is set. Speed is a property of the *step*, not the
   episode.
4. **Error-Adaptive Guidance (EAG).** Condition the next diffusion prediction on
   the tail of the previous one via classifier-free guidance, but only while the
   controller's tracking error is small. Fast execution makes the arm lag its
   goal; when it lags, the previous plan is no longer a trustworthy reference and
   the guidance is dropped for that step.

## Structure

The repo **is** robomimic — a fork, not a package alongside one. `setup.py`
installs it as `robomimic` 0.3.1. Everything SAIL adds lives under
[sail/robomimic/SAIL/](sail/robomimic/SAIL/) (~5.7k lines) plus a handful of
touched files in the core:

| Path | What changed |
|---|---|
| [sail/robomimic/algo/diffusion_policy.py](sail/robomimic/algo/diffusion_policy.py) | The guidance machinery. `future_action_condition` training (the observation condition is concatenated with a short slice of the ground-truth future actions, or with a null token), `step_classifier_free_guidance`, `step_consistency_guidance`, a SPARC smoothness loss, and `_forward_noise`/`_prev_step` for time-travel resampling. |
| [sail/robomimic/config/guide_config.py](sail/robomimic/config/guide_config.py) | `GuideConfig`, registered as algo `guided_diffusion_policy`. Read at eval time only, from a `--guide_config` JSON. |
| `robomimic/utils/train_utils.py`, `robomimic/utils/torch_utils.py` | Small hooks for the above. |
| [sail/robomimic/robosuite_configs/](sail/robomimic/robosuite_configs/) | `osc_pose_SAIL.json`, `joint_position_SAIL.json`, `joint_velocity_SAIL.json` — high-fidelity tracking variants. `osc_pose_SAIL.json` runs `kp: 600` with `control_delta: false`, against stock `osc_pose.json`'s 150 and deltas. |

Everything else (`robomimic/scripts/`, `robomimic/algo/*` besides DP,
`robomimic/utils/`, `docs/`, `tests/`, `examples/`) is stock robomimic. The
`docs/` tree is robomimic's own Sphinx documentation and says nothing about SAIL.

Inside `robomimic/SAIL/`:

| Path | What it holds |
|---|---|
| `dataset_processing/` | Absolute-action generation and HDF5 housekeeping. |
| `precision_processing/` | AWE waypoints, precision labels, and the visualiser. |
| `run_trained_agent_receding_horizon.py` | The evaluation entry point (704 lines). |
| `guide_template/*.json` | Four ready-made eval configs: no guidance, and CFG weight 0.3 / 1 / 2. |
| `eval_suite.py` | Post-hoc analysis: loads rollout pickles, groups by speed and guidance weight, plots comparisons. |
| `metrics/` | Success, horizon, throughput, tracking error, and SPARC (spectral arc length, the smoothness measure the paper reports). |
| `utils/` | `dev_utils.py` is the important one — `prepare_action`, `get_slowdown_mode_from_model`, video writing at constant simulated time. |

## The pipeline

Recorded robomimic HDF5 → three processing passes → train → evaluate. Upstream
runs everything by hand from the submodule root, `sail/`; here
`python -m baselines.sail_bridge.train <sail.hdf5>` runs passes 2-4 in the SAIL
venv with a generated config, and `sail_bridge/rollout.py` is pass 5. The
upstream commands, for reference:

```bash
conda activate SAIL

# 1. reached + commanded absolute actions, joint actions, delta joint actions
python robomimic/SAIL/dataset_processing/add_all_actions.py --dataset=<HDF5>

# 2. AWE waypoints  ->  data/demo_<i>/waypoints_dp
python robomimic/SAIL/precision_processing/save_awe_waypoint_concurrent.py \
    --dataset=<HDF5> --err_threshold=0.005 --num_workers=6

# 3. precision labels -> data/demo_<i>/{precisions, absolute_actions_with_precision}
python robomimic/SAIL/precision_processing/label_awe_trajectory_precision.py \
    --dataset=<HDF5> --action_key=absolute_actions

# 4. train
python robomimic/scripts/train.py \
    --config=robomimic/exps/templates/diffusion_policy_SAIL.json

# 5. evaluate
python robomimic/SAIL/run_trained_agent_receding_horizon.py \
    --agent=<CKPT.pth> \
    --guide_config=robomimic/SAIL/guide_template/base_cfg_weight_1.json \
    --n_rollouts=100 --dataset_path=<OUT.pkl>
```

Each pass edits the HDF5 in place and the next one refuses to run without the
previous one's keys.

### Pass 1 — `add_all_actions.py`

Rebuilds the environment from the file's `env_args`, replays each demo's states,
and writes:

| Key | Meaning |
|---|---|
| `absolute_actions` | The pose the arm **reached** at each step. |
| `commanded_absolute_actions` | The pose it was **told** to go to. |
| `joint_position_actions` | Next-step joint positions plus the gripper command. |
| `delta_joint_actions` | The same minus the current joint observation. |

**We skip this pass.** It is a simulator replay whose only purpose is to
discover where the arm ended up, and a real recording already knows. Our
converter writes `absolute_actions` and `commanded_absolute_actions` directly;
see [sail_bridge/dataset.py](sail_bridge/dataset.py) and the note in
[README.md](README.md).

### Pass 2 — `save_awe_waypoint_concurrent.py`

Calls stock AWE's `dp_waypoint_selection` on `absolute_actions` to pick the
timesteps that reconstruct the trajectory within `--err_threshold`, and stores
them as `waypoints_dp`. Reads `obs/robot0_eef_pos`, `obs/robot0_eef_quat` and
`obs/robot0_joint_pos`, which is why our converter emits all three. AWE is not
vendored — `installation.sh` clones it into `third_party/awe`, and the import is
`from waypoint_extraction.extract_waypoints import dp_waypoint_selection`.

`add_awe_actions.py` is a variant that piecewise-holds the absolute action
between waypoints; it is not in the documented path.

### Pass 3 — `label_awe_trajectory_precision.py`

DBSCAN (`eps=0.025`, `min_samples=3`, both hardcoded in `__main__`) over the
waypoint positions. A waypoint in a cluster is "precise"; the label is filled
forward between waypoints. Writes `precisions` and
`<action_key>_with_precision`, which is `absolute_actions` with the label
appended as a final column. `INV_PROP_DIST` is an alternative continuous
labelling, selected by editing the file.

`dynamic_speed_awe.py` renders the resulting speed schedule as a video with an
agentview overlay — the fastest way to see whether the labels are sane before
spending a training run on them.

### Pass 4 — training

`robomimic/scripts/train.py` is stock; the SAIL-ness is entirely in
`robomimic/exps/templates/diffusion_policy_SAIL.json`. Against stock
`diffusion_policy.json`:

| Setting | Stock | SAIL |
|---|---|---|
| `train.action_keys` / `dataset_keys` | `actions` | `absolute_actions_with_precision` |
| `algo.horizon.prediction_horizon` | 16 | 32 |
| `algo.horizon.action_horizon` | 8 | 16 |
| `algo.ddpm` / `algo.ddim` | DDPM | DDIM, 10 inference steps |
| `algo.future_action_condition` | absent | enabled, `horizon: 4`, `p_cond: 0.3`, `null_token: zero` |
| `observation.modalities.obs.rgb` | none (low-dim `object`) | `agentview_image`, `robot0_eye_in_hand_image` |
| `experiment.rollout.enabled` | true | false (rollouts happen in the eval script) |

`future_action_condition` is what makes EAG possible at eval time: the model is
trained to accept a short future-action reference alongside the observation
condition, so at inference the same input can serve as a classifier-free
guidance condition. The eval script asserts this — CFG against a checkpoint
trained without it fails loudly.

Two details of `_get_action_condition` are worth knowing before tuning it.
`p_cond` reads as the probability of *conditioning*, but the code is
`use_action_cond = rand() < p_cond` followed by `if use_action_cond: return
null_token` — so `p_cond: 0.3` means 30% null and 70% conditioned, the opposite
of the name. And the coin is flipped once per **batch**, not per sample, so
every sample in a batch is conditioned or not together.

Set `train.data` to the processed HDF5. Output goes to
`train.output_dir` (`../bc_trained_models/SAIL/` by default, i.e. outside the
repo). `sail_bridge/train.py` generates the config from this template, changing
only what our data changes: the camera keys under `observation.modalities.obs.rgb`,
`CropRandomizer`'s crop (76/84 of the file's image size, which reproduces the
template's 76 at 84), the dataset and output paths, and the epoch budget.

Two things `train.py` does that a wrapper has to know: it **catches every
exception and exits 0** (the wrapper watches for "run failed with error"), and
when `<output_dir>/<name>` already exists it **asks whether to delete it** --
every earlier run of that task. Answering `n` makes it add a new timestamped
subdirectory instead, which is what the wrapper does.

### Pass 5 — evaluation

`run_trained_agent_receding_horizon.py` is where speed and guidance actually
happen. Its rollout loop:

- Predicts an action sequence, then executes `inf_delay` (4) steps from the
  **previous** prediction before switching to `execute_n_actions` (8) steps of
  the new one — a receding horizon that also models inference latency honestly.
- Per step, `get_slowdown_mode_from_model` looks at the precision column over a
  `slowdown_window_size` (6) window centred on the current step; if any is
  `> 0.5` the step executes at `slow_control_freq`, otherwise
  `fast_control_freq`. The label column is stripped before the action reaches
  the controller.
- Rollouts are bounded by **simulated time** (`--max_sim_time`, 20 s), not step
  count — the point being throughput, and a variable control rate makes step
  counts incomparable.
- With guidance enabled, the end of each step stores the next
  `future_action_condition.horizon` actions of the current plan as the
  reference. Before the next prediction, `check_if_tracking_error_low` compares
  the controller's actual EE pose against the first pose of that reference
  (`pos_teb` 0.02 m, `ori_teb` 0.05 rad); under both bounds the reference is
  normalised and passed as the CFG condition, over either it is dropped for that
  step. `num_guided_inferences` in the output records how often it fired.

The knobs above are a `kwargs` dict literal at the bottom of the file, not CLI
flags — `fast_control_freq`, `slow_control_freq`, `execute_n_actions`,
`inf_delay`, `slowdown_window_size`, `pos_teb`, `ori_teb`, `osc_control`,
`joint_position_control`, `torque_scale`. Edit the file to change them. There is
a `# TODO: move these somewhere else` on it.

CLI flags: `--agent`, `--n_rollouts`, `--max_sim_time`, `--guide_config`,
`--N_eval` (repeat the whole evaluation N times, since the policy is
stochastic), `--dataset_path` (a **pickle**, not an HDF5), `--dataset_obs`,
`--video_path`, `--camera_names`, `--render`, `--env`, `--seed`.

Stock robomimic scripts still worth knowing, all in `robomimic/scripts/`:
`get_dataset_info.py` (print an HDF5's keys, shapes and demo count — the fastest
check that a converted file is well-formed), `playback_dataset.py` (render demos
to video), `split_train_val.py` (write a train/valid filter key),
`run_trained_agent.py` (the ordinary fixed-rate rollout, i.e. the 1x baseline).

`eval_suite.py` then consumes those pickles:

```bash
python robomimic/SAIL/eval_suite.py --result_root <DIR> --task can \
    --speeds 1x 2x 3x 4x 5x
```

It expects `<result_root>/<speed>/` subdirectories and infers the guidance
weight from each filename (`none`, `weight_0.0`, `weight_0.3`, `weight_1.0`),
raising on anything else — so name rollout outputs to match, or call
`plot_metric_comparison` directly.

## Environment

Here: `scripts/setup_baseline_envs.sh sail` builds `.venv-sail` (python 3.11,
a current cu128 torch, robomimic-SAIL editable, AWE from git, an unpatched pip
robosuite for AWE's imports). Upstream's recipe is below; it is not used
because its torch 2.1 / cu118 has no kernels for the RTX 5090, and because two
of its pins contradict its own code: `diffusers==0.11.1` predates the
`EMAModel(parameters=...)` call in `diffusion_policy.py` (0.12+; the venv
carries 0.21.4), and AWE's `setup.py` imports `pkg_resources`, which a fresh
setuptools no longer ships (installed with `--no-build-isolation`).

Upstream: `bash sail/robomimic/SAIL/installation.sh` creates a conda env
(`SAIL`, python 3.9, torch 2.1 / cu118), clones ARISE robosuite at the pinned
commit `b9d8d3de` and applies `third_party/patches/robosuite-v1.4.1-sail.patch`,
installs this package editable, and clones AWE. The robosuite patch is
load-bearing for the *simulator* and must not be applied to a newer robosuite;
nothing here builds the simulator, so it is not applied. It adds:

- `env.step(action, control_freq=...)` — the variable control frequency the
  whole method rests on, plus `motion_profile` helpers
- `torque_scale` on actuator control ranges
- `observable_sampling_rate` decoupled from control frequency
- a faster Panda gripper
- `SingleArm.last_eef_pose`

## Caveats found while reading

- **Real-hardware code is not released.** The README's phase 2 (Franka via
  Deoxys, UR5 via ROS 2) is marked "coming soon" and is not in this checkout.
  Everything here assumes a robosuite env: `run_trained_agent_receding_horizon`
  reaches into `env.unwrapped.env.robots[0].controller` for the tracking-error
  check and calls `env.getSimTimeInfo()` for its clock, neither of which exists
  outside the patched simulator. **Running SAIL on our arm means writing the
  rollout loop ourselves**; the trained policy and the precision labels transfer,
  the executor does not.
- **Inference wants what the simulator env handed it.** `RolloutPolicy` does
  no observation processing of its own: robomimic's env wrapper delivered images
  already CHW float in [0, 1], and with `train.frame_stack: 2` a
  `FrameStackWrapper` delivered every key as a `[T, ...]` stack seeded with
  copies of the first frame -- the fork comments out the policy's own obs queue
  ("already handled by frame_stack") and asserts on a single frame. And
  `get_action` returns one action and then serves an internal queue unless
  called with `return_action_sequence=True`, which the eval script passes in
  its kwargs literal. `sail_bridge/policy_server.py` does all three.
- **Only CFG is live.** `guide_config` also describes inpainting and a
  consistency (SPARC) loss, and the rollout loop `assert False, "we are not
  using this"` on both. `guide_template/base_cfg_weight_*.json` are the
  configurations that work.
- **robomimic needs two pieces of bookkeeping to open a file at all**: a
  `num_samples` attribute per demo and an `env_args` attribute on `data`. Our
  converter writes both, with `type: 6` (`EnvType.REAL_TYPE`) so
  `is_robosuite_env()` stays false and training does not try to build a
  simulator.
- Precision-label quality is the whole method. `--err_threshold` in pass 2 and
  the DBSCAN parameters in pass 3 are the tuning surface, and the upstream README
  says the defaults were set for robomimic Lift / Can / Square.
