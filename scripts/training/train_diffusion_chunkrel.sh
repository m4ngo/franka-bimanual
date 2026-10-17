#!/usr/bin/env bash

# train_diffusion.sh's flags and recipe, with the actions made chunk-relative (lerobot_train_chunkrel.py).
# $1 is repo id (an EE_POS recording)
# $2 is policy repo id
# $3 is batch size
# $4 is steps
# $5 is resume
# anything after $5 is passed to lerobot_train_chunkrel.py

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <batch_size> <steps> <resume>"
    exit 1
fi
python "$(dirname "$0")/lerobot_train_chunkrel.py" \
  --resume=$5 \
  --dataset.repo_id="$1" \
  --policy.type="diffusion" \
  --policy.horizon=16 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.repo_id="$2" \
  --output_dir="../franka_data/policy/train/diffusion_$1_$2" \
  --job_name="diffusion_chunkrel_$1" \
  --wandb.enable=true \
  --batch_size="$3" \
  --steps="$4" \
  --eval_freq=5000 \
  --num_workers=12 \
  "${@:6}"
