#!/usr/bin/env bash

# Script for training a policy on a given dataset, on the diffusion recipe SAIL
# and B-Spline share (baselines/diffusion_recipe.py, via lerobot_train_recipe.py).
# $1 is repo id
# $2 is policy repo id
# $3 is batch size (the recipe's is 64; anything else is a deviation from it)
# $4 is steps
# $5 is resume
# anything after $5 is passed to lerobot-train

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <batch_size> <steps> <resume> [lerobot-train args...]"
    exit 1
fi
python "$(dirname "$0")/lerobot_train_recipe.py" \
  --resume=$5 \
  --dataset.repo_id="$1" \
  --policy.type="diffusion" \
  --policy.horizon=16 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.repo_id="$2" \
  --output_dir="../franka_data/policy/train/diffusion_$1_$2" \
  --job_name="diffusion_$1" \
  --wandb.enable=true \
  --batch_size="$3" \
  --steps="$4" \
  --eval_freq=5000 \
  --num_workers=12 \
  "${@:6}"
