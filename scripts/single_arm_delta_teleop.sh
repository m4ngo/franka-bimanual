#!/usr/bin/env bash

# Single-arm teleop for the right-arm-only Franka wrapper.
#
# Scales map full stick deflection onto osc_pose.json's output_max envelope
# (0.05 m, 0.5 rad), which is also what clip_delta enforces. The old 0.1/0.2
# saturated translation at half deflection while leaving rotation at 40% of its
# range at full deflection -- fast translation, stiff rotation.

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
    --teleop.translation_scale=0.05 \
    --teleop.rotation_scale=0.5 \
    --fps=20