#!/usr/bin/env bash

# Roll out a trained policy on the bimanual rig.
# $1 repo_id  $2 num_episodes  $3 policy_repo_id  $4 output_dir
#
# Display: run a local Rerun viewer on the workstation and forward its ports
# over SSH to view remotely.

set -euo pipefail
source "$(dirname "$0")/_config.sh"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ] || [ -z "${4:-}" ]; then
    echo "Usage: $0 <repo_id> <number_of_episodes> <policy_repo_id> <output_dir>"
    exit 1
fi

lerobot-record \
    --policy.path="$3" \
    --policy.noise_scheduler_type="DDIM" \
    --policy.num_inference_steps=10 \
    --robot.type=bimanual_franka \
    --robot.control_mode=EE_POS \
    --dataset.repo_id="$1" \
    --dataset.num_episodes="$2" \
    --dataset.root="$4" \
    --dataset.single_task="Evaluating policy $3 on dataset $1" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --dataset.fps="$CONTROL_FPS" \
    --display_data=true
