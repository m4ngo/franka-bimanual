#!/usr/bin/env bash

# Bimanual GELLO joint-mode teleoperation.
#
# Arm IPs/ports, GELLO USB ports, and the control rate all come from
# config/*.yaml through the plugin config defaults — nothing is hardcoded here.
# Override any of them on the CLI as usual, e.g. --robot.r_port=18899.

set -euo pipefail
source "$(dirname "$0")/_config.sh"

lerobot-teleoperate \
    --robot.type=bimanual_franka \
    --robot.control_mode=JOINT_POS \
    --teleop.type=bimanual_gello \
    --teleop.id=gello_teleop \
    --fps="$CONTROL_FPS"
