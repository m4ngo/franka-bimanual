#!/usr/bin/env bash

# Replay one episode of a recorded single-arm dataset.
# $1 repo_id  $2 episode_number

set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 <repo_id> <episode_number>"
    exit 1
fi

lerobot-replay \
    --robot.type=single_arm_franka \
    --robot.control_mode=EE_DELTA \
    --dataset.repo_id="$1" \
    --dataset.episode="$2"
