#!/usr/bin/env bash
# Bring one task's trained policies back from Tillicum: ZAP's base diffusion
# policy, SAIL and B-Spline, plus their training logs (sync.sh fetch). Run on the
# workstation once progress.sh says finished:
#
#   bash baselines/tillicum/fetch_policies.sh <tillicum_user> HuskyMango/<dataset>
#
# They land under ~/franka_data/policies/HuskyMango/<dataset>/. The newest
# checkpoint of each is printed at the end; the rollouts take any of them.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -eq 2 ] || usage
export TILLICUM_USER=$1
REPO_ID=$2
bash "$HERE/sync.sh" fetch "$REPO_ID"

P=$HOME/franka_data/policies/$REPO_ID
newest() { ls -td "$@" 2>/dev/null | head -1 || true; }
base=$(newest "$P"/diffusion/*/checkpoints/last)
echo
echo "newest checkpoints under $P:"
echo "  ZAP base: ${base:+$base/pretrained_model}"
echo "  SAIL:     $(newest "$P"/sail/*/models/model_epoch_*.pth)"
echo "  B-Spline: $(newest "$P"/bspline/*/checkpoints/latest.ckpt)"
