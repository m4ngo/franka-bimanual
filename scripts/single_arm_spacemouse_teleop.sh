#!/usr/bin/env bash

# Single-arm (right) SpaceMouse teleoperation via VOsc EE_DELTA controller.
#
# The SpaceMouse emits per-step delta poses (use_delta=true) already in physical
# units (metres / radians via translation_scale / rotation_scale). We disable
# the robot-side OSC input normalisation (osc_output_max_pos=1.0,
# osc_output_max_rot=pi) so the spacemouse scales pass through unchanged.
#
# Usage:
#   ./single_arm_spacemouse_teleop.sh [hidraw_path]
#   ./single_arm_spacemouse_teleop.sh /dev/hidraw3

HIDRAW="${1:-/dev/hidraw3}"

lerobot-teleoperate \
    --robot.type=single_arm_franka \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.201.10 \
    --robot.r_port=18812 \
    --robot.control_mode=EE_DELTA \
    --robot.active_arms=[r] \
    --robot.osc_output_max_pos=1.0 \
    --robot.osc_output_max_rot=3.14159 \
    --teleop.type=spacemouse \
    --teleop.id=spacemouse_r_teleop \
    --teleop.hidraw_path="$HIDRAW" \
    --teleop.use_delta=true \
    --teleop.prefix="r_" \
    --teleop.translation_scale=0.02 \
    --teleop.rotation_scale=0.05 \
    --fps=20
