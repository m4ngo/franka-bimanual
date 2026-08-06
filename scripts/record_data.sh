#!/usr/bin/env bash

# Bimanual GELLO joint-mode recording.
# $1 repo_id  $2 num_episodes  $3 task  $4 output_dir  $5 resume
#
# Display: run a local Rerun viewer on the workstation and forward its ports
# over SSH to view remotely.
#
# If recording hangs after "Recording episode 0", try:
#   --dataset.vcodec=libsvtav1   (avoid six parallel NVENC sessions)
#   --dataset.streaming_encoding=false

set -euo pipefail
source "$(dirname "$0")/_config.sh"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ] || [ -z "${4:-}" ] || [ -z "${5:-}" ]; then
    echo "Usage: $0 <repo_id> <number_of_episodes> <task_name> <output_dir> <resume>"
    exit 1
fi

lerobot-record \
    --resume="$5" \
    --robot.type=bimanual_franka \
    --robot.control_mode=JOINT_POS \
    --teleop.type=bimanual_gello \
    --teleop.id=gello_teleop \
    --teleop.left_arm_config.use_noise=true \
    --teleop.right_arm_config.use_noise=true \
    --dataset.repo_id="$1" \
    --dataset.num_episodes="$2" \
    --dataset.single_task="$3" \
    --dataset.root="$4" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --dataset.fps="$CONTROL_FPS" \
    --display_data=false \
    --display_compressed_images=true
