#!/bin/bash
# The residual pipeline's diffusion base policy on a real recording, one GPU,
# with scripts/training/train_diffusion.sh's policy flags, through
# scripts/training/lerobot_train_recipe.py (the shared diffusion recipe and its
# EMA), run out of $FRANKA_SRC (sync.sh code), bound read-only into the stock
# LeRobot SIF. Submitted by submit_diffusion_train.sh, which exports SIF,
# CLUSTER_HOME, WORKSTATION_HOME and FRANKA_SRC and passes:
# <repo_id> <steps> <batch_size> <run_name> <policy_repo_id|-> [lerobot-train args...]
#
# At most 8 CPUs per GPU on the H200 nodes, so 8 dataloader workers, not 12.
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail
: "${SIF:?}" "${CLUSTER_HOME:?}" "${WORKSTATION_HOME:?}" "${FRANKA_SRC:?}"

REPO_ID=$1
STEPS=$2
BATCH=$3
RUN=$4
POLICY_REPO=$5
shift 5
# The container sees CLUSTER_HOME at WORKSTATION_HOME, so OUT is the
# workstation's own path to the same files.
OUT=$WORKSTATION_HOME/franka_data/policies/$REPO_ID/diffusion/$RUN
HOST_OUT=$CLUSTER_HOME/franka_data/policies/$REPO_ID/diffusion/$RUN
LAST=checkpoints/last
echo "diffusion $REPO_ID -> $HOST_OUT, job ${SLURM_JOB_ID:-local}, restart ${SLURM_RESTART_COUNT:-0}"
echo "steps $STEPS, batch $BATCH"
nvidia-smi -L

if [ -f "$HOST_OUT/$LAST/pretrained_model/train_config.json" ]; then
    done_step=$(sed -n 's/.*"step": *\([0-9]*\).*/\1/p' "$HOST_OUT/$LAST/training_state/training_step.json" 2>/dev/null || true)
    if [ -n "$done_step" ] && [ "$done_step" -ge "$STEPS" ]; then
        echo "already trained to $done_step steps"
        exit 0
    fi
    # The saved config carries every flag below; lerobot-train restores the step,
    # optimizer and wandb run from the checkpoint, the wrapper the live weights
    # and EMA.
    echo "resuming from step ${done_step:-?}"
    ARGS=(--resume=true --config_path="$OUT/$LAST/pretrained_model/train_config.json" --output_dir="$OUT")
else
    # No --dataset.root: lerobot looks in ~/.cache/huggingface/lerobot/<repo_id>,
    # where the rsync puts it, and downloads from the Hub if it is not there.
    ARGS=(--dataset.repo_id="$REPO_ID"
          --policy.type=diffusion
          --policy.horizon=16
          --policy.n_action_steps=10
          --policy.device=cuda
          --output_dir="$OUT"
          --job_name="diffusion-${REPO_ID##*/}-$RUN"
          --wandb.enable=true
          --batch_size="$BATCH"
          --steps="$STEPS"
          --eval_freq=5000
          --num_workers=8)
    if [ "$POLICY_REPO" = - ]; then
        ARGS+=(--policy.push_to_hub=false)
    else
        ARGS+=(--policy.repo_id="$POLICY_REPO" --policy.push_to_hub=true)
    fi
    ARGS+=("$@")
    # lerobot-train refuses an existing output_dir unless it resumes; this one
    # died before its first checkpoint.
    [ -d "$HOST_OUT" ] && mv "$HOST_OUT" "$HOST_OUT.aborted.$(date +%s)"
    mkdir -p "$(dirname "$HOST_OUT")"
fi

JOB_TMP=$CLUSTER_HOME/tmp/${SLURM_JOB_ID:-local}
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT
ENVS=()
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ENVS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
[ -n "${WANDB_API_KEY:-}" ] && ENVS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
[ -n "${HF_TOKEN:-}" ] && ENVS+=(--env "HF_TOKEN=$HF_TOKEN")

# The interpreter lerobot-train itself runs on, whatever `python` is on PATH.
apptainer exec --nv --containall --writable-tmpfs \
    --home "$CLUSTER_HOME:$WORKSTATION_HOME" \
    --bind "$JOB_TMP:/tmp" --bind "$FRANKA_SRC:/opt/franka_ws:ro" "${ENVS[@]}" \
    "$SIF" \
    sh -c 'exec "$(dirname "$(command -v lerobot-train)")/python" "$@"' sh \
    /opt/franka_ws/scripts/training/lerobot_train_recipe.py "${ARGS[@]}"
