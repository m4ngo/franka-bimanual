#!/bin/bash
# LIBERO rollouts, PER_GPU units at once on one GPU per array element. Submitted
# by submit_libero_eval.sh, which exports SIF, CLUSTER_HOME and WORKSTATION_HOME
# and passes: <suite> <units file> <per-gpu>
#
# A unit is one line of the units file, `|`-separated: task, backend, sweep id,
# evaluate.py args, rollout args. Rollouts are mostly CPU (mujoco); the GPU only
# holds the policies and renders, so several share one.
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --requeue
# EGL rendering stalls at startup on g011 (multi-fast/docs/EGL_STARTUP_STALL.md).
#SBATCH --exclude=g011

set -euo pipefail
: "${SIF:?}" "${CLUSTER_HOME:?}" "${WORKSTATION_HOME:?}"

SUITE=$1
UNITS_FILE=$2
PER_GPU=$3
mapfile -t UNITS < "$UNITS_FILE"
FIRST=$((SLURM_ARRAY_TASK_ID * PER_GPU))
# The container sees CLUSTER_HOME at WORKSTATION_HOME, so PREP is the
# workstation's own path to the same files.
PREP=$WORKSTATION_HOME/franka_data/baseline_prep/$SUITE
HOST_PREP=$CLUSTER_HOME/franka_data/baseline_prep/$SUITE
echo "units $FIRST+ of $UNITS_FILE, job $SLURM_JOB_ID on $(hostname), restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi -L

JOB_TMP=$CLUSTER_HOME/tmp/$SLURM_JOB_ID
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / PER_GPU ))
ENVS=(--env NVIDIA_DRIVER_CAPABILITIES=all --env "OMP_NUM_THREADS=$(( THREADS > 0 ? THREADS : 1 ))")
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ENVS+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

PIDS=()
NAMES=()
for ((slot = 0; slot < PER_GPU; slot++)); do
    i=$((FIRST + slot))
    [ "$i" -lt "${#UNITS[@]}" ] || break
    IFS='|' read -r TASK BACKEND SWEEP EVAL_ARGS ROLLOUT_ARGS <<< "${UNITS[$i]}"
    read -ra EVAL_ARGS <<< "$EVAL_ARGS"
    read -ra ROLLOUT_ARGS <<< "$ROLLOUT_ARGS"
    # Below the ephemeral range, so no outgoing connection holds it; distinct
    # per job and slot. libero_rollout.sh refuses a port someone else holds.
    PORT=$((20000 + (SLURM_JOB_ID * 8 + slot) % 12000))
    mkdir -p "$HOST_PREP/rollout_logs_$SWEEP"
    echo "[$slot] $TASK $BACKEND $SWEEP port $PORT ${EVAL_ARGS[*]} -- ${ROLLOUT_ARGS[*]}"
    apptainer exec --nv --containall --writable-tmpfs \
        --home "$CLUSTER_HOME:$WORKSTATION_HOME" \
        --bind "$JOB_TMP:/tmp" \
        "${ENVS[@]}" \
        --pwd /opt/franka_ws \
        "$SIF" \
        python -m baselines.libero_bridge.evaluate "$PREP" \
            --tasks "$TASK" --backend "$BACKEND" --sweep-id "$SWEEP" --resume \
            --log-dir "$PREP/rollout_logs_$SWEEP" "${EVAL_ARGS[@]}" \
            --extra --port "$PORT" "${ROLLOUT_ARGS[@]}" \
        > >(sed -u "s/^/[$TASK $SWEEP] /") 2>&1 &
    PIDS+=($!)
    NAMES+=("$TASK $SWEEP")
    # Staggered, so the policies do not all load from GPFS at once.
    sleep 5
done

FAILED=0
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
        echo "FAILED: ${NAMES[$k]}; see $HOST_PREP/rollout_logs_${NAMES[$k]#* }/"
        FAILED=$((FAILED + 1))
    fi
done
echo "finished ${#PIDS[@]} unit(s), $FAILED failed"
[ "$FAILED" -eq 0 ]
