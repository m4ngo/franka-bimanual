#!/usr/bin/env bash

# Script for training a policy on a given dataset.
# $1 is repo id
# $2 is policy repo id
# $3 is batch size
# $4 is steps
# $5 policy type

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <batch_size> <steps> <resume> <config_path>"
    exit 1
fi
lerobot-train \
  --resume=$5 \
  --dataset.repo_id="$1" \
  --policy.type=act \
  --output_dir="../franka_data/policy/train/act_$1" \
  --policy.chunk_size=10 \
  --policy.n_action_steps=5 \
  --job_name="act_$1" \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="$2" \
  --batch_size="$3" \
  --steps="$4" \
  --eval_freq=5000 \
  --num_workers=4
