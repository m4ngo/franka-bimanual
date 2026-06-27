#!/usr/bin/env bash

# Back-compat alias for single-arm SpaceMouse + EE_DELTA teleop.
exec "$(dirname "$0")/single_arm_spacemouse_delta_teleop.sh" "$@"
