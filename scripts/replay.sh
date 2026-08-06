#!/usr/bin/env bash

# Replay one episode of a recorded bimanual dataset.
# $1 repo_id  $2 episode_number

set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 <repo_id> <episode_number>"
    exit 1
fi

lerobot-replay \
    --robot.type=bimanual_franka \
    --robot.control_mode=EE_POS \
    --dataset.repo_id="$1" \
    --dataset.episode="$2"
