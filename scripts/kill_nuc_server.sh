#!/usr/bin/env bash
# Kill the pylibfranka torque server (and its gripper server) on a NUC.
#
# Standalone version of the kill step from deploy_nuc_server.sh, for when you
# just want to stop the server without redeploying or restarting it.
#
# Usage: kill_nuc_server.sh [mario|luigi]
set -euo pipefail
source "$(dirname "$0")/_config.sh"

TARGET="${1:-mario}"

# NUC hostnames map to arms in config/arms.yaml.
case "$TARGET" in
  mario) ARM=right ;;
  luigi) ARM=left  ;;
  left|right) ARM="$TARGET" ;;
  *) echo "unknown target '$TARGET' (expected mario/luigi or left/right)" >&2; exit 1 ;;
esac

eval "$(cfg_arm "$ARM" --prefix NUC)"
HOST="$NUC_SSH"; PORT="$NUC_PORT"; GRIPPER_PORT="$NUC_GRIPPER_PORT"

# Bracketed first char so the pattern never matches the shell running pkill --
# 'rpyc_classic -p 18812' appears verbatim in that shell's own argv, so an
# unbracketed -f pattern makes pkill kill itself before it kills the server.
echo "==> stopping anything on port $PORT"
ssh "$HOST" "pkill -f '[r]pyc_classic -p $PORT' || true; pkill -f '[r]pyc_classic -p $GRIPPER_PORT' || true; pkill -f '[p]ylibfranka_server.py --port $PORT' || true; pkill -f '[p]ylibfranka_control.py' || true; sleep 1"

echo "==> done ($TARGET)"
