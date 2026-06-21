#!/usr/bin/env bash

# Bimanual SpaceMouse teleoperation.
#
# Left SpaceMouse:  /dev/hidraw4   (override with --teleop.left_arm_config.hidraw_path=...)
# Right SpaceMouse: /dev/hidraw5   (override with --teleop.right_arm_config.hidraw_path=...)
#
# The robot runs in EE_POS mode to match the absolute pose commands produced
# by BimanualSpaceMouse.

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
    --robot.control_mode=EE_POS \
    --teleop.type=bimanual_spacemouse \
    --teleop.id=spacemouse_teleop \
    --teleop.left_arm_config.hidraw_path=/dev/hidraw2 \
    --teleop.right_arm_config.hidraw_path=/dev/hidraw3 \
    --fps=30
