#!/usr/bin/env bash

# Single-arm bimanual_franka teleop with one GELLO leader.
#
# $1 arm    l | r
# $2 mode   gello | gello_ee   (optional, default gello_ee)
#
# The inactive arm is left untouched — neither homed nor commanded.
# Park it manually (Program mode + guide-by-hand) before running, or
# run `python scripts/home_pose.py apply <bimanual_pose>` first to home
# both arms via the default-bimanual robot config.

ARM="${1:-}"
MODE="${2:-gello_ee}"

if [ "$ARM" != "l" ] && [ "$ARM" != "r" ]; then
    echo "Usage: $0 <l|r> [gello|gello_ee]"
    exit 1
fi

case "$ARM" in
    l) PORT=/dev/ttyUSB1 ;;
    r) PORT=/dev/ttyUSB0 ;;
esac

case "$MODE" in
    gello)    USE_EE_POS=false ;;
    gello_ee) USE_EE_POS=true  ;;
    *) echo "mode must be gello or gello_ee"; exit 1 ;;
esac

lerobot-teleoperate \
    --robot.type=bimanual_franka \
    --robot.l_server_ip=192.168.3.11 \
    --robot.l_robot_ip=192.168.200.2 \
    --robot.l_gripper_ip=192.168.2.21 \
    --robot.l_port=18813 \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.2.20 \
    --robot.r_port=18812 \
    --robot.use_ee_pos=$USE_EE_POS \
    --robot.active_arms=[$ARM] \
    --teleop.type=$MODE \
    --teleop.id=${MODE}_${ARM}_teleop \
    --teleop.side=$ARM \
    --teleop.port=$PORT \
    --fps=30
