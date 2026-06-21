#!/usr/bin/env bash

# Homed bimanual policy rollout.
# Each rollout episode starts with the follower arms driven to a saved
# home pose, giving the policy a consistent starting state.
#
# $1 repo_id          eval dataset (must start with eval_)
# $2 num_episodes     integer
# $3 policy_repo_id   HF model repo
# $4 output_dir       local dataset root (must not exist)
# $5 home_pose_name   name of a saved pose in ~/franka_ws/home_poses/

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <repo_id> <num_episodes> <policy_repo_id> <output_dir> <home_pose_name>"
    exit 1
fi

python "$(dirname "$0")/lerobot_record_homed.py" \
    --repo-id "$1" \
    --num-episodes "$2" \
    --task "Evaluating policy $3 on dataset $1" \
    --output-dir "$4" \
    --policy "$3" \
    --home-pose-name "$5" \
    --control-mode EE_POS
