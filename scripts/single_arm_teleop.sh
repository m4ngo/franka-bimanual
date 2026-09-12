#!/usr/bin/env bash

# Single-arm teleop. One script, four leader/mode combinations:
#
#   spacemouse_delta   SpaceMouse -> EE_DELTA    per-step delta, the policy's own action
#   spacemouse_ee      SpaceMouse -> EE_POS      integrated absolute target pose
#   gello_ee           GELLO      -> EE_POS      absolute pose via FR3 forward kinematics
#   gello              GELLO      -> JOINT_POS   joint setpoints
#
# The physical arm behind the `r_` keys and the leader device the operator holds
# are SEPARATE settings, both on the rig profile in config/rig.yaml (arms /
# teleop_device); ports and hidraw paths are in config/teleop.yaml. Which profile
# runs by default is rig.yaml's `default_single_arm_profile`.
#
# This runs scripts/teleop_single_arm.py rather than `lerobot-teleoperate`
# because the EE_POS modes need the leader's target seeded from the arm's real
# pose before the first step, and the CLI has no hook for that. See that file.
#
# $1 mode   (optional, default spacemouse_delta)
# Remaining arguments are passed through, e.g. --rig=single_arm_right
# --teleop-device=left --fps=20.

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

# Read back --rig so the banner names the rig actually being driven, and pass the
# default on explicitly when the operator gave none. A default restated here and
# in the python driver is how the banner came to say single_arm_right while
# single_arm_franka connected; config/rig.yaml is the one place it is written.
RIG=""
prev=""
for arg in "$@"; do
    case "$arg" in --rig=*) RIG="${arg#--rig=}" ;; esac
    [[ "$prev" == "--rig" ]] && RIG="$arg"
    prev="$arg"
done

if [ -z "$RIG" ]; then
    RIG=$(cfg rig.default_single_arm_profile)
    set -- --rig="$RIG" "$@"
fi

DEVICE=$(cfg "rig.profiles.${RIG}.teleop_device")
echo "${MODE}: using the ${DEVICE}-hand leader on ${RIG} at ${CONTROL_FPS} Hz"

exec python "$(dirname "$0")/teleop_single_arm.py" "$MODE" --fps="$CONTROL_FPS" "$@"
