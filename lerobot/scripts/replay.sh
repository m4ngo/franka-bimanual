#!/usr/bin/env bash

# Script for replaying a recorded dataset.
# $1 is repo id
# $2 is episode number

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  conda activate lerobot >/dev/null 2>&1 || true
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <episode_number>"
    exit 1
fi
lerobot-replay \
    --robot.type=bimanual_franka \
    --robot.l_server_ip=192.168.3.11 \
    --robot.l_robot_ip=192.168.200.2 \
    --robot.l_gripper_ip=192.168.2.21 \
    --robot.l_port=18813 \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.2.20 \
    --robot.r_port=18812 \
    --robot.use_ee_delta=false \
    --dataset.repo_id="$1" \
    --dataset.episode="$2"