#!/usr/bin/env bash
# Kill processes that occupy the local 'open_fci' port (443).
#
# Usage: kill_open_fci.sh [mario|luigi]
set -euo pipefail
source "$(dirname "$0")/_config.sh"

TARGET="${1:-mario}"

case "$TARGET" in
  mario) ARM=right ;;
  luigi) ARM=left  ;;
  left|right) ARM="$TARGET" ;;
  *) echo "unknown target '$TARGET' (expected mario/luigi or left/right)" >&2; exit 1 ;;
esac

eval "$(cfg_arm "$ARM" --prefix FCI)"
REMOTE_IP="$FCI_ROBOT_IP"

echo "==> stopping anything binding local port 443 (open_fci)"

# Kill local SSH tunnels that forward local port 443 (match '-L 443:')
# Bracket the first char so pkill doesn't match its own argv.
pkill -f '[s]sh -N -L 443:' || true

# As a fallback, kill any process using TCP port 443 (requires sudo).
sudo fuser -k 443/tcp || true

sleep 1
echo "==> done ($TARGET)"
