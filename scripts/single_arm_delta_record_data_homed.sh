#!/usr/bin/env bash

# Homed single-arm delta recording via SpaceMouse.
# The robot runs in use_delta mode: each action from the SpaceMouse is an EE
# delta (dx, dy, dz + delta quaternion) applied directly without goal-pose PD.
#
# $1 repo_id          HuggingFace dataset to write
# $2 num_episodes     integer
# $3 task             single_task description
# $4 output_dir       local dataset root (must not exist unless --resume)
# $5 resume           true|false
# $6 home_pose_name   name of a saved pose in ~/franka_ws/home_poses/

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
    echo "Usage: $0 <repo_id> <num_episodes> <task> <output_dir> <resume> <home_pose_name>"
    exit 1
fi

python "$(dirname "$0")/lerobot_record_homed_single_arm.py" \
    --repo-id "$1" \
    --num-episodes "$2" \
    --task "$3" \
    --output-dir "$4" \
    --resume "$5" \
    --home-pose-name "$6" \
    --use-ee-pos false \
    --use-delta true \
    --teleop-mode spacemouse \
    --teleop-id spacemouse_single_arm_teleop
