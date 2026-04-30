#!/usr/bin/env bash

# Script for training a policy on a given dataset.
# $1 is repo id
# $2 is policy repo id
# $3 is batch size
# $4 is steps

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  conda activate lerobot >/dev/null 2>&1 || true
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <batch_size> <steps>"
    exit 1
fi
lerobot-train \
  --dataset.repo_id="$1" \
  --policy.type=act \
  --output_dir="~/franka_data/policy/train/act_$1" \
  --job_name="act_$1" \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="$2" \
  --batch_size="$3" \
  --steps="$4"