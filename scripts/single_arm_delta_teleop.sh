#!/usr/bin/env bash
# Single-arm teleop for the right-arm-only Franka wrapper.
#
# Scales are the SpaceMouseConfig defaults now (0.05 m / 0.5 rad = robosuite
# osc_pose.json's output_max), so full stick deflection is exactly a normalized
# +/-1 policy action -- teleop and policy rollouts drive the controller through
# identical units.
#
# uncouple_pos_ori stays at its default of TRUE, matching osc_pose.json. Setting
# it false raises commanded torque ~13x, which combined with kp_ori_scale
# saturated the joint torque clamp and produced dangerous motion. Rotation
# authority comes from kp_ori_scale / friction_kc instead.
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
    --fps=20
