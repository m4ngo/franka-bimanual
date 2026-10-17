"""The diffusion-policy recipe SAIL, B-Spline and our LeRobot base policy share.

All three are Chi et al.'s conv-UNet diffusion policy with per-camera ResNet-18
encoders. B-Spline's training config
(bspline_policy/config/train_diffusion_unet_real_hybrid_bspline_workspace.yaml)
is the reference; everything below is its value. What may differ is only what
each policy reads and predicts: observation keys, action layout and horizon,
SAIL's future-action conditioning, and each upstream's input normalization.
The robot state is the same for all three, end-effector pose and gripper
(our policy converts its joints with lerobot_robot_bimanual_franka.ee_state).
The crop is `train_common.crop_size`.

stdlib-only: read by the bridges in the workspace venv, by the SAIL launcher in
.venv-sail, inside the cluster image (which copies only baselines/), and by
scripts/training/lerobot_train_recipe.py.
"""

from __future__ import annotations

import copy

UNET_DOWN_DIMS = (256, 512, 1024)
UNET_STEP_EMBED_DIM = 128
UNET_KERNEL_SIZE = 5
UNET_N_GROUPS = 8

N_OBS_STEPS = 2
NUM_TRAIN_TIMESTEPS = 100
NUM_INFERENCE_STEPS = 10
BETA_SCHEDULE = "squaredcos_cap_v2"
PREDICTION_TYPE = "epsilon"

# AdamW (decoupled weight decay), no gradient clipping.
LR = 1e-4
BETAS = (0.95, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-6

# diffusers' get_cosine_schedule_with_warmup, stepped every gradient step.
LR_SCHEDULE = "cosine"
LR_WARMUP_STEPS = 500

# Decay 1 - (1 + step / inv_gamma) ** -power, clipped to [min, max]; the EMA
# weights are what inference loads.
EMA_POWER = 0.75
EMA_INV_GAMMA = 1.0
EMA_MIN_DECAY = 0.0
EMA_MAX_DECAY = 0.9999

BATCH_SIZE = 64
SEED = 42


def _hydra_list(values) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def bspline_overrides() -> list[str]:
    """Hydra overrides that state the recipe on B-Spline's workspace config."""
    return [
        f"policy.down_dims={_hydra_list(UNET_DOWN_DIMS)}",
        f"policy.diffusion_step_embed_dim={UNET_STEP_EMBED_DIM}",
        f"policy.kernel_size={UNET_KERNEL_SIZE}",
        f"policy.n_groups={UNET_N_GROUPS}",
        f"n_obs_steps={N_OBS_STEPS}",
        f"policy.noise_scheduler.num_train_timesteps={NUM_TRAIN_TIMESTEPS}",
        f"policy.noise_scheduler.beta_schedule={BETA_SCHEDULE}",
        f"policy.noise_scheduler.prediction_type={PREDICTION_TYPE}",
        f"policy.num_inference_steps={NUM_INFERENCE_STEPS}",
        "optimizer._target_=torch.optim.AdamW",
        f"optimizer.lr={LR}",
        f"optimizer.betas={_hydra_list(BETAS)}",
        f"optimizer.eps={EPS}",
        f"optimizer.weight_decay={WEIGHT_DECAY}",
        f"training.lr_scheduler={LR_SCHEDULE}",
        f"training.lr_warmup_steps={LR_WARMUP_STEPS}",
        "training.use_ema=true",
        "ema.update_after_step=0",
        f"ema.inv_gamma={EMA_INV_GAMMA}",
        f"ema.power={EMA_POWER}",
        f"ema.min_value={EMA_MIN_DECAY}",
        f"ema.max_value={EMA_MAX_DECAY}",
    ]


def apply_to_sail(cfg: dict, num_training_steps: int, start_step: int = 0) -> None:
    """Write the recipe into a robomimic config dict, in place.

    The optimizer, schedule and EMA keys beyond upstream's are registered by
    sail_bridge/recipe_patch.py, which must be installed wherever the config
    is loaded.
    """
    algo = cfg["algo"]
    unet = algo["unet"]
    unet["down_dims"] = list(UNET_DOWN_DIMS)
    unet["diffusion_step_embed_dim"] = UNET_STEP_EMBED_DIM
    unet["kernel_size"] = UNET_KERNEL_SIZE
    unet["n_groups"] = UNET_N_GROUPS
    algo["horizon"]["observation_horizon"] = N_OBS_STEPS
    cfg["train"]["frame_stack"] = N_OBS_STEPS
    for sched in ("ddpm", "ddim"):
        algo[sched]["num_train_timesteps"] = NUM_TRAIN_TIMESTEPS
        algo[sched]["beta_schedule"] = BETA_SCHEDULE
        algo[sched]["prediction_type"] = PREDICTION_TYPE
    algo["ddim"]["num_inference_timesteps"] = NUM_INFERENCE_STEPS

    policy = algo["optim_params"]["policy"]
    policy["optimizer_type"] = "adamw"
    policy["betas"] = list(BETAS)
    policy["eps"] = EPS
    policy["regularization"]["L2"] = WEIGHT_DECAY
    lr = policy["learning_rate"]
    lr["initial"] = LR
    lr["epoch_schedule"] = []
    lr["step_scheduler"] = LR_SCHEDULE
    lr["num_warmup_steps"] = LR_WARMUP_STEPS
    lr["num_training_steps"] = int(num_training_steps)
    lr["start_step"] = int(start_step)
    cfg["train"]["max_grad_norm"] = None

    ema = algo["ema"]
    ema["enabled"] = True
    ema["use_warmup"] = True
    ema["power"] = EMA_POWER
    ema["inv_gamma"] = EMA_INV_GAMMA
    ema["min_decay"] = EMA_MIN_DECAY
    ema["max_decay"] = EMA_MAX_DECAY
    # diffusers' EMAModel counts the step before computing the decay, B-Spline's
    # after; one step of delay makes the two decay sequences identical.
    ema["update_after_step"] = 1


def sail_mismatches(cfg: dict) -> list[str]:
    """Config paths where a robomimic config is not this recipe; [] when it is."""
    lr = cfg["algo"]["optim_params"]["policy"]["learning_rate"]
    want = copy.deepcopy(cfg)
    apply_to_sail(want, num_training_steps=lr.get("num_training_steps") or 0,
                  start_step=lr.get("start_step", 0))

    def walk(have, expect, path):
        if isinstance(expect, dict):
            if not isinstance(have, dict):
                return [path]
            return [p for k in expect for p in walk(have.get(k), expect[k], f"{path}.{k}" if path else k)]
        return [] if have == expect else [path]

    return walk(cfg, want, "")


def lerobot_args() -> list[str]:
    """lerobot-train flags that state the recipe for `--policy.type=diffusion`.

    The crop is not here: it depends on the image size, and
    lerobot_train_recipe.py sets it from the dataset.
    """
    return [
        f"--policy.down_dims={_hydra_list(UNET_DOWN_DIMS)}",
        f"--policy.diffusion_step_embed_dim={UNET_STEP_EMBED_DIM}",
        f"--policy.kernel_size={UNET_KERNEL_SIZE}",
        f"--policy.n_groups={UNET_N_GROUPS}",
        "--policy.use_film_scale_modulation=true",
        "--policy.vision_backbone=resnet18",
        "--policy.pretrained_backbone_weights=null",
        "--policy.use_group_norm=true",
        "--policy.spatial_softmax_num_keypoints=32",
        "--policy.use_separate_rgb_encoder_per_camera=true",
        "--policy.crop_is_random=true",
        f"--policy.n_obs_steps={N_OBS_STEPS}",
        "--policy.noise_scheduler_type=DDIM",
        f"--policy.num_train_timesteps={NUM_TRAIN_TIMESTEPS}",
        f"--policy.beta_schedule={BETA_SCHEDULE}",
        f"--policy.prediction_type={PREDICTION_TYPE}",
        "--policy.clip_sample=true",
        "--policy.clip_sample_range=1.0",
        f"--policy.num_inference_steps={NUM_INFERENCE_STEPS}",
        "--use_policy_training_preset=false",
        "--optimizer.type=adamw",
        f"--optimizer.lr={LR}",
        f"--optimizer.betas={_hydra_list(BETAS)}",
        f"--optimizer.eps={EPS}",
        f"--optimizer.weight_decay={WEIGHT_DECAY}",
        "--optimizer.grad_clip_norm=0",
        "--scheduler.type=diffuser",
        f"--scheduler.name={LR_SCHEDULE}",
        f"--scheduler.num_warmup_steps={LR_WARMUP_STEPS}",
        f"--batch_size={BATCH_SIZE}",
        f"--seed={SEED}",
    ]


def fingerprint() -> dict:
    """Every recipe value, for telling apart policies trained on different recipes."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(globals().items())
            if k.isupper() and not k.startswith("_")}
