#!/usr/bin/env bash
# Train the diffusion base policy on a real recording on Tillicum, one GPU, with
# scripts/training/train_diffusion.sh's policy flags and the shared diffusion
# recipe (train_diffusion.slurm runs lerobot_train_recipe.py out of $FRANKA_SRC,
# which `sync.sh code` fills on the workstation).
# Prints the sbatch line; --apply submits it.
#
#   bash submit_diffusion_train.sh HuskyMango/<dataset> --steps 200000 --apply
#   bash submit_diffusion_train.sh HuskyMango/<dataset> --steps 200000 --batch-size 32 \
#       --name bs32 --policy-repo-id HuskyMango/diffusion-<dataset> --apply
#
# The recording is read from $CLUSTER_HOME/.cache/huggingface/lerobot/<repo_id>
# (rsync it there), or downloaded from the Hub if it is missing. The policy
# lands in $CLUSTER_HOME/franka_data/policies/<repo_id>/diffusion/<name>.
# Resubmitting the same command resumes from the last checkpoint, or does
# nothing if the training finished. Anything after -- goes to lerobot-train.

set -euo pipefail

TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/scrubbed/$USER}"
CLUSTER_HOME="${CLUSTER_HOME:-$TILLICUM_ROOT/franka_home}"
SIF="${DIFFUSION_SIF:-$TILLICUM_ROOT/containers/lerobot-train.sif}"
WORKSTATION_HOME="${WORKSTATION_HOME:-/home/franka}"

HERE="$(cd "$(dirname "$0")" && pwd)"
usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || usage
REPO_ID=$1
shift
STEPS=""
BATCH=64
NAME=""
POLICY_REPO=-
APPLY=false
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --steps)          STEPS="$2"; shift 2 ;;
        --batch-size)     BATCH="$2"; shift 2 ;;
        --name)           NAME="$2"; shift 2 ;;
        --policy-repo-id) POLICY_REPO="$2"; shift 2 ;;
        --apply)          APPLY=true; shift ;;
        --)               shift; PASS=("$@"); break ;;
        *)                echo "unknown argument: $1" >&2; usage ;;
    esac
done
[ -n "$STEPS" ] || { echo "--steps is required" >&2; exit 2; }
NAME=${NAME:-bs$BATCH-$STEPS}

[ -f "$SIF" ] || { echo "no $SIF; apptainer pull it first" >&2; exit 1; }
FRANKA_SRC="${FRANKA_SRC:-$TILLICUM_ROOT/franka_src}"
for f in scripts/training/lerobot_train_recipe.py baselines/diffusion_recipe.py \
         baselines/bspline_policy/diffusion_policy/diffusion_policy/model/diffusion/ema_model.py \
         lerobot_robot_bimanual_franka/ee_state.py franka_config/__init__.py config/control.yaml; do
    [ -f "$FRANKA_SRC/$f" ] || { echo "no $FRANKA_SRC/$f; on the workstation: bash baselines/tillicum/sync.sh code" >&2; exit 1; }
done
DATA=$CLUSTER_HOME/.cache/huggingface/lerobot/$REPO_ID
if [ -f "$DATA/meta/info.json" ]; then
    echo "recording: $DATA"
else
    echo "no $DATA; the job will download $REPO_ID from the Hub (needs HF_TOKEN if private)"
fi
[ -n "${WANDB_API_KEY:-}" ] || echo "WANDB_API_KEY is not set; the job cannot log to wandb" >&2
[ "$POLICY_REPO" = - ] || [ -n "${HF_TOKEN:-}" ] || echo "HF_TOKEN is not set; the push to $POLICY_REPO will fail" >&2

LOG_DIR="$CLUSTER_HOME/franka_data/policies/$REPO_ID/diffusion/slurm_logs"
cmd=(sbatch --job-name="diffusion-${REPO_ID##*/}"
     --output="$LOG_DIR/%x_%j.out"
     --export="ALL,SIF=$SIF,CLUSTER_HOME=$CLUSTER_HOME,WORKSTATION_HOME=$WORKSTATION_HOME,FRANKA_SRC=$FRANKA_SRC"
     "$HERE/train_diffusion.slurm" "$REPO_ID" "$STEPS" "$BATCH" "$NAME" "$POLICY_REPO" "${PASS[@]}")

echo "policy: $CLUSTER_HOME/franka_data/policies/$REPO_ID/diffusion/$NAME"
echo "${cmd[@]}"
if $APPLY; then
    mkdir -p "$LOG_DIR"
    "${cmd[@]}"
else
    echo "Dry run. Re-run with --apply to submit."
fi
