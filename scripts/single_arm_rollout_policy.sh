#!/usr/bin/env bash

# Single-arm bimanual_franka policy rollout. EE-mode hardcoded.
#
# $1 arm             l | r
# $2 repo_id         eval dataset (must start with eval_)
# $3 num_episodes    integer
# $4 policy_repo_id  HF model repo
# $5 output_dir      local dataset root (must not exist)
#
# Park the inactive arm out of the way before running — see
# single_arm_teleop.sh comment for details.

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <l|r> <repo_id> <num_episodes> <policy_repo_id> <output_dir>"
    exit 1
fi

ARM="$1"
if [ "$ARM" != "l" ] && [ "$ARM" != "r" ]; then
    echo "arm must be 'l' or 'r'"
    exit 1
fi

lerobot-record \
    --policy.path="$4" \
    --robot.type=bimanual_franka \
    --robot.l_server_ip=192.168.3.11 \
    --robot.l_robot_ip=192.168.200.2 \
    --robot.l_gripper_ip=192.168.2.21 \
    --robot.l_port=18813 \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.2.20 \
    --robot.r_port=18812 \
    --robot.use_ee_pos=true \
    --robot.active_arms=[$ARM] \
    --dataset.repo_id="$2" \
    --dataset.num_episodes="$3" \
    --dataset.root="$5" \
    --dataset.single_task="Evaluating policy $4 on dataset $2" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --dataset.fps=30 \
    --display_data=false
