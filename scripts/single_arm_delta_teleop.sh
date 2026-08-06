#!/usr/bin/env bash

# Single-arm SpaceMouse EE-delta teleop.
#
# Which SpaceMouse the operator holds comes from config/rig.yaml
# (single_arm_franka.teleop_device) — NOT from which arm the rig drives. Both
# SpaceMice enumerate as identical HID devices, so the wrong one connects
# happily and then does nothing. Scales live in config/teleop.yaml.

set -euo pipefail
source "$(dirname "$0")/_config.sh"

DEVICE=$(cfg rig.profiles.single_arm_franka.teleop_device)
HIDRAW=$(cfg "teleop.spacemouse.devices.${DEVICE}.hidraw_path")
echo "using ${DEVICE}-hand SpaceMouse on ${HIDRAW}"

lerobot-teleoperate \
    --robot.type=single_arm_franka \
    --robot.control_mode=EE_DELTA \
    --teleop.type=spacemouse \
    --teleop.id=spacemouse_r_teleop \
    --teleop.hidraw_path="$HIDRAW" \
    --teleop.prefix="r_" \
    --teleop.use_delta=true \
    --fps="$CONTROL_FPS"
