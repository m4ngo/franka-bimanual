#!/usr/bin/env bash

# Single-arm SpaceMouse teleop with OSC-style EE_DELTA control.
#
# The SpaceMouse emits per-tick EE deltas (metres + delta quaternion). The
# robot accumulates those into persistent goals and tracks them with task-space
# PD velocity (bimanual_franka._ee_delta_osc), matching robosuite OSC_POSE
# semantics at 20 Hz policy rate.
#
# Scales match OSC output_max defaults: ±0.05 m and ±0.5 rad per control step
# at full axis deflection (see env_wrapper._POS_SCALE / _ROT_SCALE).
#
# Right arm only (mario NUC). Right SpaceMouse default: /dev/hidraw3.
# Override hidraw:  HIDRAW=/dev/hidrawN ./single_arm_spacemouse_delta_teleop.sh
#
# End teleop with Ctrl+C.

HIDRAW="${HIDRAW:-/dev/hidraw3}"

lerobot-teleoperate \
    --robot.type=single_arm_franka \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.201.10 \
    --robot.r_port=18812 \
    --robot.control_mode=EE_DELTA \
    --robot.active_arms=[r] \
    --teleop.type=spacemouse \
    --teleop.id=spacemouse_r_delta_teleop \
    --teleop.hidraw_path="$HIDRAW" \
    --teleop.prefix="r_" \
    --teleop.use_delta=true \
    --teleop.translation_scale=0.05 \
    --teleop.rotation_scale=0.5 \
    --fps=20
