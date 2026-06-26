#!/usr/bin/env bash

# Single-arm (right) SpaceMouse teleoperation via OSC Cartesian velocity (EE_DELTA).
#
# Interface convention: SpaceMouse outputs normalized [-1, 1] on all axes
# (translation_scale=1.0, rotation_scale=1.0). Physical scaling lives
# entirely on the robot side:
#   osc_output_max_pos: device ±1 → EE position delta ±0.05 m per step
#   osc_output_max_rot: device ±1 → EE rotation delta ±0.5 rad per step
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
    --robot.osc_output_max_pos=0.05 \
    --robot.osc_output_max_rot=0.5 \
    --robot.osc_kp_base=2.0 \
    --robot.osc_kp_null=0.0 \
    --robot.osc_kp_ori_ratio=0.3 \
    --teleop.type=spacemouse \
    --teleop.id=spacemouse_r_teleop \
    --teleop.hidraw_path="$HIDRAW" \
    --teleop.use_delta=true \
    --teleop.prefix="r_" \
    --teleop.translation_scale=1.0 \
    --teleop.rotation_scale=1.0 \
    --fps=20