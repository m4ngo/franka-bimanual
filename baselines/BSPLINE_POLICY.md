# B-Spline Policy — repo overview

Upstream: <https://github.com/B-spline-policy/bspline-policy>, submodule at
[bspline_policy/](bspline_policy/), pinned at `61ed5f4` on `main`. We do not
edit it. What we feed it is described in [README.md](README.md); this file is a
map of the repo itself.

## The idea

A diffusion policy normally predicts a fixed-length action chunk at the rate the
demonstrations were recorded, so execution speed is baked into the training
data. B-Spline Policy predicts the **parameters of a B-spline** instead: one
knot column plus control points. Deployment evaluates that spline at whatever
wall-clock time it likes, so playing a plan back at 2x is a change of variable
in `t`, not a different policy.

The policy-facing action is a dense matrix

```
(chunk_size + 2*degree, 1 + action_dim)
    column 0        -> knot vector
    columns 1..end  -> B-spline control points, one column per action dim
```

with `chunk_size = horizon - 2*degree` and `degree = 3`. Everything else in the
stack is stock Diffusion Policy: same UNet, same DDIM scheduler, same robomimic
dataset plumbing. The B-spline part is one extra model channel for the knot
column plus a preprocessing step that fits the spline.

## Four top-level directories

Paths below are relative to this directory, so the submodule root is
`bspline_policy/`.

| Directory | What it is |
|---|---|
| [bspline_policy/bspline_policy/](bspline_policy/bspline_policy/) | The actual contribution: spline fitting, dataset, policy adapters, rollout wrapper, `train.py`. ~2.5k lines. |
| [bspline_policy/diffusion_policy/](bspline_policy/diffusion_policy/) | Vendored fork of Diffusion Policy. Supplies the training workspace, replay buffer, normalizers, UNet and image encoders that `bspline_policy` subclasses. `diffusion-policy.patch` records the four files changed from upstream. |
| [bspline_policy/real_env/](bspline_policy/real_env/) | Their hardware: a single YAM arm (i2rt motor chain), PyRoKi/J-PARSE IK, iPhone teleop, episode recording. Irrelevant to our arm except as the reference for what the data looks like. |
| [bspline_policy/inference/](bspline_policy/inference/) | A README only. Deployment instructions, no code. |

Mind the triple nesting: `bspline_policy/` is the submodule root,
`bspline_policy/bspline_policy/` is the project directory (`setup.py`,
`train.py`, `docs/`, `tests/`), and
`bspline_policy/bspline_policy/bspline_policy/` is the importable package. The
rest of this file spells the paths out in full.

## Read the code in this order

1. [.../bspline_policy/common/bspline_action.py](bspline_policy/bspline_policy/bspline_policy/common/bspline_action.py)
   — the whole representation. `ScipyBSplineCompression.compress` walks
   `scipy.interpolate.generate_knots` adding knots until max reconstruction
   error drops under `max_error` (0.002 m in the configs), then
   `chunk_bspline_trajectory` cuts the fitted spline into fixed-size parameter
   matrices, and `BSplineChunkSampler` maps every timestep in an episode to the
   chunk that starts at or before it, with the knot column shifted so `t=0` is
   the current step. `decode_bspline_action` is the inverse: parameters back to
   evaluated action vectors.
2. [.../common/knots.py](bspline_policy/bspline_policy/bspline_policy/common/knots.py) — 30
   lines. Optional re-parameterisation of the knot column as first-knot +
   successive differences (`relative_knots`, off in every shipped config).
3. [.../dataset/robomimic_replay_bspline_image_dataset.py](bspline_policy/bspline_policy/bspline_policy/dataset/robomimic_replay_bspline_image_dataset.py)
   — reuses Diffusion Policy's `_convert_robomimic_to_replay` to build the zarr
   replay buffer from a robomimic HDF5 (this is where axis-angle becomes
   `rotation_6d`), then replaces the sequence sampler with
   `BSplineChunkSampler`.
4. [.../policy/diffusion_unet_bspline_image_policy.py](bspline_policy/bspline_policy/bspline_policy/policy/diffusion_unet_bspline_image_policy.py)
   — 34 lines. Deep-copies `shape_meta`, adds 1 to the action dim for the knot
   column, and returns the full predicted horizon instead of a slice. The
   transformer variant is the same trick.
5. [.../scripts/policy_local_bspline.py](bspline_policy/bspline_policy/bspline_policy/scripts/policy_local_bspline.py)
   — 894 lines, the deployment half, described below.

## Entry points

Training (hydra; config names resolve under the package's own `config/`
directory, `bspline_policy/bspline_policy/bspline_policy/config/`):

```bash
conda activate bsp-simple
cd bspline_policy/bspline_policy
python train.py --config-name=train_diffusion_unet_real_hybrid_bspline_workspace \
  training.resume=false logging.mode=offline
```

`train.py` is a 41-line shim: it puts the project directory and
`diffusion_policy/` on `sys.path`, registers the `eval:` OmegaConf resolver, and
hands off to Diffusion Policy's `TrainDiffusionUnetHybridWorkspace`. Checkpoints
land under `data/outputs/<date>/<time>_<name>_<task>/`, relative to wherever
`train.py` was run from.

| Config | Task | Action dim | Obs keys |
|---|---|---|---|
| `train_diffusion_unet_real_hybrid_bspline_workspace` + `task/square_image_abs_bspline` | single YAM arm (their default) | 10 = xyz + rot6d + gripper | `wrist_image`, `arm_pos`, `arm_quat`, `gripper_pos` |
| `clean_bspline_policy_unet_bspline` + `task/clean_bspline_policy_haoyu_left_bspline` | single arm, two cameras | 10 | `head_image`, `left_wrist_image`, `arm_pos_l`, `arm_quat_l`, `gripper_pos_l` |
| `clean_bspline_policy_unet_bspline` + `task/clean_bspline_policy_stack_cube_teleop_10hz_fix_cam_bspline` | bimanual X5 | 20 | dual-arm |

The DP baseline they compare against is a config in the vendored fork
(`bspline_policy/diffusion_policy/diffusion_policy/config/clean_bspline_policy_unet_dp.yaml`),
trained through `bspline_policy/diffusion_policy/train.py` with the same HDF5.

Rollout:

```bash
python bspline_policy/real_env/yam_teleop/rollout_local_policy.py \
  --env yam --policy bspline --ckpt-path <CKPT> \
  --control-freq 200 --data-freq 10 \
  --origin-time-scale 10 --predict-before-end 0.1 --speed-up-times 2.0 \
  --num-inference-steps 10 --cuda-graph
```

Data collection and conversion, all under
[bspline_policy/real_env/yam_teleop/](bspline_policy/real_env/yam_teleop/):

| Script | What it does |
|---|---|
| `yam_server.py` | RPC server owning the arm. Streams joint commands at 100 Hz toward the latest Cartesian goal via PyRoKi/J-PARSE IK. Everything else talks to this. |
| `main.py --teleop --save` | The teleop/record loop at 10 Hz (`POLICY_CONTROL_FREQ` in `constants.py`). An iPhone running XR Browser is the leader and also signals episode start/end and env reset; episodes go to `data/demos/<timestamp>/`. |
| `reviewer.py` | Flask UI to watch recorded episodes and mark keep/discard. |
| `sort_demos_from_review.py` | Applies a review JSON to the episode directories. |
| `convert_to_robomimic_hdf5.py` | Episode dirs to robomimic HDF5. 64 lines, and the reference for what our own converter has to produce. |
| `mock_env.py` | Runs the rollout loop against a synthetic env with configurable observation latency. Useful for timing the policy without hardware. |
| `episode_storage.py` | `EpisodeWriter` / `EpisodeReader`. Images are stored as one MP4 per camera per episode. |

Also in `bspline_policy/bspline_policy/bspline_policy/scripts/`:

| Script | What it does |
|---|---|
| `policy_server_bspline.py` | ZMQ server returning raw spline parameters under a `bspline` key, for `main.py`'s `RemotePolicy`. Alternative to running the policy in-process. |
| `yam_replay_episodes_bspline.py` | Fit a spline through a recorded episode's actions and replay it on the arm, optionally sped up. This is the ablation that shows the speed-up is achievable by the hardware before any policy is involved. |
| `tidybot2_replay_episodes_bspline.py`, `mujoco_bsp_replay.py` | Same for their other two platforms. |
| `rollout_x5_bspline.py` | Bimanual X5 rollout. |

## What the HDF5 must contain

Per the default YAM task, one group per episode under `data/demo_<i>/`:

```
obs/arm_pos       (T, 3)
obs/arm_quat      (T, 4)   xyzw
obs/gripper_pos   (T, 1)
obs/wrist_image   (T, H, W, 3)  uint8
actions           (T, 7)   xyz + rotvec + gripper
```

The dataset loader converts the 3-vector rotation to `rotation_6d`, which is why
`shape_meta.action.shape` is `[10]` while the file holds 7. Our converter is
[bspline_bridge/dataset.py](bspline_bridge/dataset.py) and writes exactly these
keys.

## Deployment: where the speed-up actually happens

`PolicyLocalBSpline` keeps a `scipy.interpolate.BSpline` as the current plan and
a background thread doing inference:

- **Sampling.** Each control tick evaluates the spline at
  `t = (now - plan_start) * speed_up_times * origin_time_scale`.
  `origin_time_scale` converts the spline's index-space knots into seconds
  (10.0 = the 10 Hz recording rate); `speed_up_times` is the actual speed
  multiplier. So 2x execution is a scale on `t` and nothing else.
- **Re-planning.** A new spline is requested when the plan has less than
  `predict_before_end` seconds of *wall clock* left. The comparison is done in
  origin-trajectory seconds, so the threshold is scaled by `speed_up_times`;
  without that scaling the plan runs dry mid-inference at high speed-up and the
  arm stutters segment by segment.
- **Stitching.** `_align_new_plan` does a bounded scalar minimisation over the
  new spline for the `t` whose action is closest to the last action emitted from
  the old one, and starts there rather than at `t=0`, widening the search window
  by 1.5x until the residual drops under `--time-align-error-threshold`. Past
  that threshold it warns, or with `--restart-on-time-align-error` restarts at 0.
  `--consider-gripper-during-align` includes the gripper column in the match;
  by default only the pose columns count.
- **Inference cost.** `CudaGraphDDIMSampler` captures the DDIM denoising loop in
  a CUDA graph (`--cuda-graph`), which is what makes 10-step inference cheap
  enough to re-plan at speed.
  `bspline_policy/bspline_policy/tests/test_cuda_graph_ddim.py`
  runs a checkpoint through both the eager and the graph path under the same
  seed and asserts the predicted chunks match; it needs a CUDA device and a
  checkpoint, and skips otherwise.
- **Gripper.** `--gripper-slowdown-enabled` locally drops back toward 1x for a
  few steps when the commanded gripper moves more than a threshold, since a
  sped-up grasp is where speed costs success.

## Environments

Two, and they conflict:

- Training and inference: conda, from
  `bspline_policy/diffusion_policy/conda_environment.yaml` (env named
  `robodiff`, referred to as `bsp-simple` in some of their docs), plus
  `robomimic==0.2.0 --no-deps`, then `pip install -e .` from
  `bspline_policy/bspline_policy/` (where `setup.py` is — the upstream README
  says `pip install -e bspline_policy` from inside that same directory, which
  points one level too deep).
- Hardware: `uv sync` in `bspline_policy/real_env/`, plus editable installs of
  its `i2rt/` and `pyroki/` subdirectories.

Neither is our workspace venv, which is why our side of the comparison stops at
producing the HDF5.

## Caveats found while reading

- **Stale `simple_mobile/` paths.** The repo was renamed from `simple_mobile`
  and several scripts still compute paths through it:
  `yam_replay_episodes_bspline.py`, `tidybot2_replay_episodes_bspline.py`,
  `mujoco_bsp_replay.py`, `rollout_x5_bspline.py`, `policy_server_bspline.py`,
  and `rollout_local_policy.py`'s `--diffusion-policy-dir` default. The
  directory is `real_env/` in this checkout. `rollout_local_policy.py --env yam`
  survives it (the script's own directory is already on `sys.path`); the replay
  scripts do not.
- **Only `--env yam --policy bspline` is complete here.** `policy_local_dp.py`,
  `x5_env.py` and `real_yam_bimanual_hex_env.py` are all imported by
  `rollout_local_policy.py` and none of them are in the repo, so `--policy dp`
  and `--env x5|tidybot2` cannot run. The same goes for
  `bspline_policy/bspline_policy/docs/*.md`, which drive rollout through a
  `simple_mobile/tidybot2/rollout_local_policy.py` that does not exist. Treat
  those docs as historical.
- **Caches are keyed by name, not content.** The dataset writes
  `<hdf5>.<cache_suffix>.zarr.zip` next to the input and the sampler writes a
  `.bspline_sampler_<hash>.npz`. The sampler hash covers its own parameters, but
  the zarr cache is only keyed by `cache_suffix` — regenerate the HDF5 without
  changing the suffix and training silently reads the old replay buffer.
