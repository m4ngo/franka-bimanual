#!/usr/bin/env bash

# Bimanual GELLO EE teleoperation.
#
# Like the standard GELLO teleop but the leaders output absolute EE poses
# (via Franka FR3 forward kinematics) rather than joint angles, so the robot
# runs in EE-position mode (use_ee_pos=true).
#
# Left  GELLO: /dev/ttyUSB1   (override with --teleop.left_arm_config.port=...)
# Right GELLO: /dev/ttyUSB0   (override with --teleop.right_arm_config.port=...)

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
    --robot.use_ee_pos=true \
    --teleop.type=bimanual_gello_ee \
    --teleop.id=gello_ee_teleop \
    --teleop.left_arm_config.port=/dev/ttyUSB1 \
    --teleop.right_arm_config.port=/dev/ttyUSB0 \
    --fps=15
