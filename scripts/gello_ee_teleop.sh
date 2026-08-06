#!/usr/bin/env bash

# Bimanual GELLO EE teleoperation.
#
# Like the standard GELLO teleop but the leaders output absolute EE poses
# (via Franka FR3 forward kinematics) rather than joint angles, so the robot
# runs in EE_POS mode.
#
# GELLO USB ports come from config/teleop.yaml (gello.devices).

set -euo pipefail
source "$(dirname "$0")/_config.sh"

lerobot-teleoperate \
    --robot.type=bimanual_franka \
    --robot.control_mode=EE_POS \
    --teleop.type=bimanual_gello_ee \
    --teleop.id=gello_ee_teleop \
    --fps="$CONTROL_FPS"
