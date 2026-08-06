#!/usr/bin/env bash
# Shared config accessors for the scripts in this directory.
#
# Everything resolves through `python -m franka_config`, so shell scripts and
# Python read the exact same config/*.yaml. Source this file, don't run it.
#
#   source "$(dirname "$0")/_config.sh"
#   FPS=$(cfg control.rates.control_fps)
#   eval "$(cfg_rig bimanual_franka)"     # L_SERVER_IP, R_PORT, CONTROL_FPS, …
#   eval "$(cfg_arm right)"               # R_SERVER_IP, R_SSH, R_GRIPPER_PORT, …

cfg() { python -m franka_config get "$1"; }
cfg_arm() { python -m franka_config arm "$@"; }
cfg_rig() { python -m franka_config rig "$1"; }
cfg_cameras() { python -m franka_config cameras "$@"; }

# Control rate shared by teleop, recording, rollout, sysid and residual runs.
CONTROL_FPS="$(cfg control.rates.control_fps)"
export CONTROL_FPS
