#!/usr/bin/env bash

# Script for replaying a recorded dataset.
# $1 is repo id
# $2 is episode number

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <episode_number>"
    exit 1
fi
lerobot-replay \
    --robot.type=single_arm_franka \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.201.10 \
    --robot.r_port=18812 \
    --robot.control_mode=EE_DELTA \
    --robot.active_arms=[r] \
    --dataset.repo_id="$1" \
    --dataset.episode="$2"
