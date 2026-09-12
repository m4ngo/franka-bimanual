#!/usr/bin/env bash

# Homed single-arm Franka recording.
# Each episode starts with the arm driven to a saved home pose.
#
# $1 repo_id          HuggingFace dataset to write
# $2 num_episodes     integer
# $3 task             single_task description
# $4 output_dir       local dataset root (must not exist unless --resume)
# $5 resume           true|false
# $6 home_pose_name   name of a saved pose in the home_poses dir
# $7 mode             spacemouse_delta | spacemouse_ee | gello_ee | gello
#                     (optional, default gello_ee)
# $8 depth            true | false                    (optional, default true)
# $9 rig              single_arm_franka | single_arm_right
#                     (optional, default rig.yaml's default_single_arm_profile --
#                     which PHYSICAL arm each drives is in config/rig.yaml, NOT
#                     the key prefix)
#
# The control mode is NOT passed here: it follows from the leader, and the one
# table that pairs them lives in scripts/teleop_single_arm.py. A copy of that
# mapping used to sit in this file, which is how a leader could be recorded
# against a control mode it does not speak.

set -euo pipefail
source "$(dirname "$0")/_config.sh"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ] || [ -z "${4:-}" ] || [ -z "${5:-}" ] || [ -z "${6:-}" ]; then
    echo "Usage: $0 <repo_id> <num_episodes> <task> <output_dir> <resume> <home_pose_name>" \
         "[spacemouse_delta|spacemouse_ee|gello_ee|gello] [true|false]" \
         "[single_arm_franka|single_arm_right]"
    exit 1
fi

MODE="${7:-gello_ee}"
DEPTH="${8:-true}"
RIG="${9:-$(cfg rig.default_single_arm_profile)}"

python "$(dirname "$0")/lerobot_record_homed_single_arm.py" \
    --fps "$CONTROL_FPS" \
    --repo-id "$1" \
    --num-episodes "$2" \
    --task "$3" \
    --output-dir "$4" \
    --resume "$5" \
    --home-pose-name "$6" \
    --depth "$DEPTH" \
    --rig "$RIG" \
    --teleop-mode "$MODE" \
    --teleop-id "${MODE}_single_arm_teleop" \
    --noise True
