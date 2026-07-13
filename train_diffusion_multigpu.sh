#!/usr/bin/env bash
# Multi-GPU version of train.sh for Tillicum, using accelerate.
#
# Same interface as the original single-GPU train.sh:
#   $1 = dataset repo id
#   $2 = policy repo id
#   $3 = per-GPU batch size (NOT effective/global batch size -- see note below)
#   $4 = steps
#   $5 = resume (true/false)
#
# IMPORTANT: effective batch size = $3 * NUM_GPUS (accelerate documents this
# explicitly). If you're used to running with --batch_size=32 on 1 GPU and
# want the same effective batch size on 4 GPUs, pass batch_size=8 here, not 32.
# If you instead want to keep per-GPU batch_size the same as your single-GPU
# runs (i.e. scale up total throughput/effective batch), just pass the same
# number as before -- but note this also changes the effective batch size
# your model sees, so loss curves/hyperparameters (esp. LR) may need retuning.
# LeRobot does NOT auto-scale learning rate for you.
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <per_gpu_batch_size> <steps> <resume>"
    exit 1
fi

NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "Launching accelerate with NUM_GPUS=${NUM_GPUS}"

LEROBOT_TRAIN_BIN="$(command -v lerobot-train || true)"
if [ -z "$LEROBOT_TRAIN_BIN" ]; then
    echo "ERROR: lerobot-train not found on PATH. Did you activate your venv/module?" >&2
    echo "PATH=$PATH" >&2
    exit 1
fi
echo "Using lerobot-train at: $LEROBOT_TRAIN_BIN"

accelerate launch \
  --multi_gpu \
  --num_processes="${NUM_GPUS}" \
  "$LEROBOT_TRAIN_BIN" \
  --resume=$5 \
  --dataset.repo_id="$1" \
  --policy.type=act \
  --output_dir="/gpfs/projects/${HYAK_PROJECT:?set HYAK_PROJECT env var}/franka_data/policy/train/act_$2" \
  --policy.chunk_size=100 \
  --policy.n_action_steps=5 \
  --job_name="act_$1" \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="$2" \
  --batch_size="$3" \
  --steps="$4" \
  --eval_freq=5000 \
  --num_workers=8