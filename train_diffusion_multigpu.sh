#!/usr/bin/env bash
# Multi-GPU diffusion policy training script for Tillicum, using accelerate.
#
#   $1 = dataset repo id
#   $2 = policy repo id
#   $3 = per-GPU batch size (NOT effective/global batch size)
#   $4 = steps
#   $5 = resume (true/false)
#
# effective batch size = $3 * NUM_GPUS. LeRobot does NOT auto-scale learning
# rate for you -- retune if you change GPU count.
set -euo pipefail

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <per_gpu_batch_size> <steps> <resume>"
    exit 1
fi

NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "Launching accelerate with NUM_GPUS=${NUM_GPUS}"

# Verify lerobot-train is resolvable before handing off to accelerate.
# `which` is not guaranteed present in this image, and `$(which ...)`
# failures don't reliably trigger `set -e` inside command substitution --
# that silent failure (empty entrypoint passed to accelerate launch) is what
# caused the earlier garbled-argument crash. `command -v` is a bash builtin
# (always available) and its result is checked explicitly here instead.
if ! command -v lerobot-train >/dev/null 2>&1; then
    echo "ERROR: lerobot-train not found on PATH" >&2
    echo "PATH=$PATH" >&2
    exit 1
fi

LEROBOT_TRAIN_BIN="$(command -v lerobot-train)"
if [ -z "$LEROBOT_TRAIN_BIN" ]; then
    echo "ERROR: lerobot-train not found on PATH" >&2
    echo "PATH=$PATH" >&2
    exit 1
fi
echo "Resolved lerobot-train to: ${LEROBOT_TRAIN_BIN}"

# Pass "lerobot-train" by name -- accelerate/torchrun resolves it via PATH
# themselves, same as it resolves plain python entrypoints. No need to
# pre-resolve to an absolute path in bash.
accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_GPUS}" \
  "${LEROBOT_TRAIN_BIN}" \
  --resume="$5" \
  --dataset.repo_id="$1" \
  --output_dir="/gpfs/projects/${HYAK_PROJECT:?set HYAK_PROJECT env var}/franka_data/policy/train/act_$2" \
  --job_name="diffusion_$1" \
  --wandb.enable=true \
  --policy.type="diffusion" \
  --policy.noise_scheduler_type="DDIM" \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=5 \
  --policy.horizon=16 \
  --policy.n_action_steps=10 \
  --policy.n_obs_steps=2 \
  --policy.device=cuda \
  --policy.repo_id="$2" \
  --batch_size="$3" \
  --steps="$4" \
  --eval_freq=5000 \
  --num_workers=32