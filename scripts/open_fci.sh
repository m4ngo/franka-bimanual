#!/usr/bin/env bash
# Forward local port 443 to a Franka control box so Desk is reachable.
#
# Usage: open_fci.sh [mario|luigi|left|right]
#
# Arm IPs and NUC SSH targets come from config/arms.yaml. Override the key with
# SSH_KEY=/path/to/key.
set -euo pipefail
source "$(dirname "$0")/_config.sh"

TARGET="${1:-}"

case "$TARGET" in
  mario) ARM=right ;;
  luigi) ARM=left  ;;
  left|right) ARM="$TARGET" ;;
  *) echo "Invalid choice '$TARGET'. Expected mario/luigi or left/right." >&2; exit 1 ;;
esac

eval "$(cfg_arm "$ARM" --prefix FCI)"

sudo ssh -N -L "443:${FCI_ROBOT_IP}:443" \
    -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" \
    -o StrictHostKeyChecking=accept-new \
    "$FCI_SSH"
