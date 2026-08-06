#!/usr/bin/env bash

# Bimanual SpaceMouse teleoperation.
#
# hidraw paths and motion scales come from config/teleop.yaml
# (spacemouse.devices / spacemouse.translation_scale / .rotation_scale).
#
# The robot runs in EE_POS mode to match the absolute pose commands produced
# by BimanualSpaceMouse.

set -euo pipefail
source "$(dirname "$0")/_config.sh"

lerobot-teleoperate \
    --robot.type=bimanual_franka \
    --robot.control_mode=EE_POS \
    --teleop.type=bimanual_spacemouse \
    --teleop.id=spacemouse_teleop \
    --fps="$CONTROL_FPS"
