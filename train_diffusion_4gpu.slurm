#!/bin/bash
#SBATCH --job-name=act-train
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/weirdlab/atsai06/logs/slurm-%j.out
# NOTE: Slurm expands #SBATCH directives before the script body runs, so you
# cannot reference the $HYAK_PROJECT variable set below in this line --
# replace YOUR_GROUP_NAME literally, matching the export below.

# --- Fill these in for your project -----------------------------------------
export HYAK_PROJECT="weirdlab/atsai06"          # e.g. matches /gpfs/projects/<this>
SIF_PATH="/gpfs/projects/${HYAK_PROJECT}/lerobot-train.sif"
DATASET_REPO_ID="HuskyMango/bowl-7-10"               # HF dataset repo_id
POLICY_REPO_ID="HuskyMango/diffusion-bowl-7-13"            # HF model repo_id to push to
PER_GPU_BATCH_SIZE=16                            # effective batch = this * num GPUs
STEPS=80000
RESUME=false
# -----------------------------------------------------------------------------

# HF + WANDB auth tokens should be set as environment variables in your shell
# profile or passed via --env to apptainer, NOT hardcoded here or committed
# to the repo. e.g.:
#   export HF_TOKEN=...
#   export WANDB_API_KEY=...
# Apptainer passes through host env vars by default unless --cleanenv is used.

apptainer exec --nv --bind /gpfs \
  "${SIF_PATH}" \
  bash /workspace/scripts/train_diffusion_multigpu.sh \
    "${DATASET_REPO_ID}" \
    "${POLICY_REPO_ID}" \
    "${PER_GPU_BATCH_SIZE}" \
    "${STEPS}" \
    "${RESUME}"
