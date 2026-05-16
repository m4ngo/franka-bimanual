#!/usr/bin/env bash

# Script for training a policy on a given dataset.
# $1 is repo id
# $2 is policy repo id
# $3 is batch size
# $4 is steps
# $5 policy type

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <batch_size> <steps> <policy_type> <resume> <config_path>"
    exit 1
fi
lerobot-train \
  --resume=$6 \
  --dataset.repo_id="$1" \
  --policy.type=$5 \
  --policy.noise_scheduler_type="DDIM" \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=10 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.n_obs_steps=2 \
  --output_dir="../franka_data/policy/train/$5_$1" \
  --job_name="$5_$1" \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="$2" \
  --batch_size="$3" \
  --steps="$4" \
  --eval_freq=5000 \
  --num_workers=4