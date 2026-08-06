#!/usr/bin/env bash

# Single-arm GELLO teleop. The physical arm behind the `r_` keys and the leader
# device the operator holds are SEPARATE settings, both in config/rig.yaml
# (single_arm_franka: arms / teleop_device); the port itself is in
# config/teleop.yaml.
#
# $1 mode   gello | gello_ee   (optional, default gello_ee)

set -euo pipefail
source "$(dirname "$0")/_config.sh"

MODE="${1:-gello_ee}"

case "$MODE" in
    gello)    CONTROL_MODE=JOINT_POS ;;
    gello_ee) CONTROL_MODE=EE_POS    ;;
    *) echo "mode must be gello or gello_ee"; exit 1 ;;
esac

DEVICE=$(cfg rig.profiles.single_arm_franka.teleop_device)
PORT=$(cfg "teleop.gello.devices.${DEVICE}.port")
echo "using ${DEVICE}-hand GELLO on ${PORT}"

lerobot-teleoperate \
    --robot.type=single_arm_franka \
    --robot.control_mode="$CONTROL_MODE" \
    --teleop.type="$MODE" \
    --teleop.id="${MODE}_r_teleop" \
    --teleop.side=r \
    --teleop.port="$PORT" \
    --fps="$CONTROL_FPS"
