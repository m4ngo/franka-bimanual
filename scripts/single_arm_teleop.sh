#!/usr/bin/env bash

# Single-arm teleop for the right-arm-only Franka wrapper.
#
# $1 mode   gello | gello_ee   (optional, default gello_ee)

MODE="${1:-gello_ee}"

PORT=/dev/ttyUSB0

case "$MODE" in
    gello)    CONTROL_MODE=JOINT_POS ;;
    gello_ee) CONTROL_MODE=EE_POS    ;;
    *) echo "mode must be gello or gello_ee"; exit 1 ;;
esac

lerobot-teleoperate \
    --robot.type=single_arm_franka \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.201.10 \
    --robot.r_port=18812 \
    --robot.control_mode=$CONTROL_MODE \
    --robot.active_arms=[r] \
    --teleop.type=$MODE \
    --teleop.id=${MODE}_r_teleop \
    --teleop.side=r \
    --teleop.port=$PORT \
    --fps=20