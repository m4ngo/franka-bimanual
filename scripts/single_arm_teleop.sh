#!/usr/bin/env bash

# Single-arm teleop. One script, four leader/mode combinations:
#
#   spacemouse_delta   SpaceMouse -> EE_DELTA    per-step delta, the policy's own action
#   spacemouse_ee      SpaceMouse -> EE_POS      integrated absolute target pose
#   gello_ee           GELLO      -> EE_POS      absolute pose via FR3 forward kinematics
#   gello              GELLO      -> JOINT_POS   joint setpoints
#
# The physical arm behind the `r_` keys and the leader device the operator holds
# are SEPARATE settings, both in config/rig.yaml (single_arm_franka: arms /
# teleop_device); ports and hidraw paths are in config/teleop.yaml.
#
# This runs scripts/teleop_single_arm.py rather than `lerobot-teleoperate`
# because the EE_POS modes need the leader's target seeded from the arm's real
# pose before the first step, and the CLI has no hook for that. See that file.
#
# $1 mode   (optional, default spacemouse_delta)
# Remaining arguments are passed through, e.g. --teleop-device=left --fps=20.

set -euo pipefail
source "$(dirname "$0")/_config.sh"

MODE="${1:-spacemouse_delta}"
shift || true

case "$MODE" in
    spacemouse_delta|spacemouse_ee|gello_ee|gello) ;;
    *)
        echo "mode must be one of: spacemouse_delta spacemouse_ee gello_ee gello" >&2
        exit 1
        ;;
esac

DEVICE=$(cfg rig.profiles.single_arm_franka.teleop_device)
echo "${MODE}: using the ${DEVICE}-hand leader at ${CONTROL_FPS} Hz"

exec python "$(dirname "$0")/teleop_single_arm.py" "$MODE" --fps="$CONTROL_FPS" "$@"
