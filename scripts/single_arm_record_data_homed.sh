#!/usr/bin/env bash

# Homed single-arm Franka recording.
# Each episode starts with the right arm driven to a saved home pose.
#
# $1 repo_id          HuggingFace dataset to write
# $2 num_episodes     integer
# $3 task             single_task description
# $4 output_dir       local dataset root (must not exist unless --resume)
# $5 resume           true|false
# $6 home_pose_name   name of a saved pose in ~/franka_ws/home_poses/
# $7 mode             gello | gello_ee | spacemouse   (optional, default gello_ee)

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
    echo "Usage: $0 <repo_id> <num_episodes> <task> <output_dir> <resume> <home_pose_name> [gello|gello_ee|spacemouse]"
    exit 1
fi

MODE="${7:-gello_ee}"

case "$MODE" in
    gello)       USE_EE_POS=false; USE_DELTA=false ;;
    gello_ee)    USE_EE_POS=true;  USE_DELTA=false ;;
    spacemouse)  USE_EE_POS=false; USE_DELTA=true  ;;
    *) echo "mode must be gello, gello_ee, or spacemouse"; exit 1 ;;
esac

python "$(dirname "$0")/lerobot_record_homed_single_arm.py" \
    --repo-id "$1" \
    --num-episodes "$2" \
    --task "$3" \
    --output-dir "$4" \
    --resume "$5" \
    --home-pose-name "$6" \
    --use-ee-pos "$USE_EE_POS" \
    --use-delta "$USE_DELTA" \
    --teleop-mode "$MODE" \
    --teleop-id "${MODE}_single_arm_teleop"