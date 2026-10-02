#!/bin/bash
# One LIBERO baseline training per array element, one GPU each. Submitted by
# submit_libero_train.sh, which exports SIF, CLUSTER_HOME and WORKSTATION_HOME
# and passes: <suite> <steps> <task_i:backend,...> [train.py args...]
#
# At most 8 CPUs per GPU on the H200 nodes; B-Spline's dataloader alone uses 8
# workers, so one training per GPU.
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail
: "${SIF:?}" "${CLUSTER_HOME:?}" "${WORKSTATION_HOME:?}"

SUITE=$1
STEPS=$2
IFS=, read -ra UNITS <<< "$3"
shift 3
UNIT=${UNITS[$SLURM_ARRAY_TASK_ID]}
TASK=${UNIT%:*}
BACKEND=${UNIT#*:}
# The container sees CLUSTER_HOME at WORKSTATION_HOME, so PREP is the
# workstation's own path to the same files.
PREP=$WORKSTATION_HOME/franka_data/baseline_prep/$SUITE
HOST_PREP=$CLUSTER_HOME/franka_data/baseline_prep/$SUITE
echo "unit $UNIT, job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi -L

SCRATCH=/dev/shm/libero_$SLURM_JOB_ID
JOB_TMP=$CLUSTER_HOME/tmp/$SLURM_JOB_ID
mkdir -p "$SCRATCH" "$JOB_TMP"
BINDS=(--bind "$JOB_TMP:/tmp")
ENVS=()
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ENVS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
[ -n "${WANDB_API_KEY:-}" ] && ENVS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")

SRC=$HOST_PREP/$TASK/sail.hdf5
cleanup() {
    # SAIL labels its HDF5 in place; keep the labels, and never leave a half-copied file.
    if [ -f "$SCRATCH/sail.hdf5" ] && [ "$SCRATCH/sail.hdf5" -nt "$SRC" ]; then
        cp -p "$SCRATCH/sail.hdf5" "$SRC.tmp" && mv "$SRC.tmp" "$SRC"
    fi
    rm -rf "$SCRATCH" "$JOB_TMP"
}
trap cleanup EXIT

# robomimic reads image windows from the HDF5 for every sample; GPFS is slow at
# that, RAM is not. B-Spline trains from an in-memory copy it builds itself.
if [ "$BACKEND" = sail ]; then
    cp -p "$SRC" "$SCRATCH/sail.hdf5"
    BINDS+=(--bind "$SCRATCH/sail.hdf5:$PREP/$TASK/sail.hdf5")
fi

# The wrappers need only h5py, numpy and cv2, which the SAIL env has; each
# trainer then runs in its own env (baselines/interpreters.py).
apptainer exec --nv --containall --writable-tmpfs \
    --home "$CLUSTER_HOME:$WORKSTATION_HOME" \
    "${BINDS[@]}" "${ENVS[@]}" \
    --pwd /opt/franka_ws \
    "$SIF" \
    /opt/franka_ws/.venv-sail/bin/python -m baselines.libero_bridge.train "$PREP" \
        --tasks "$TASK" --backend "$BACKEND" --steps "$STEPS" --resume "$@"
