#!/usr/bin/env bash

# Single-arm teleop for the right-arm-only Franka wrapper.

PORT=/dev/ttyUSB0

lerobot-teleoperate \
    --robot.type=single_arm_franka \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.201.10 \
    --robot.r_port=18812 \
    --robot.control_mode=EE_DELTA \
    --robot.active_arms=[r] \
    --teleop.type=spacemouse \
    --teleop.id=${MODE}_r_teleop \
    --teleop.hidraw_path="/dev/hidraw3" \
    --teleop.prefix="r_" \
    --teleop.use_delta=true \
    --teleop.translation_scale=0.1 \
    --teleop.rotation_scale=0.2 \
    --fps=20