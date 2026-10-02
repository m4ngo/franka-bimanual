#!/usr/bin/env bash
# Train LIBERO baselines on Tillicum, every task_<i>:<backend> at once: one Slurm
# array element per training, each on its own GPU (baselines/LIBERO_SIM.md,
# "Training on Tillicum"). Prints the sbatch line; --apply submits it.
#
#   bash submit_libero_train.sh libero_10 --tasks "3 4 5 6 7 8 9" --steps 200000 --apply
#   bash submit_libero_train.sh libero_10 --tasks "3:bspline 4" --steps 200000 --apply
#   bash submit_libero_train.sh libero_10 --steps 200000 --apply -- --extra --wandb
#
# --tasks takes indices, `<i>:<backend>` for one backend of a task; default
# every task_<i> directory in the prep dir. Anything after -- goes to
# baselines.libero_bridge.train. Resubmitting skips what has finished and
# resumes what has not.

set -euo pipefail

# Where things live on Tillicum; each can be overridden from the environment.
TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/scrubbed/$USER}"
CLUSTER_HOME="${CLUSTER_HOME:-$TILLICUM_ROOT/franka_home}"
SIF="${SIF:-$TILLICUM_ROOT/containers/franka-baselines.sif}"
# The job mounts CLUSTER_HOME here, so every path a trainer records is valid
# on the workstation once synced back.
WORKSTATION_HOME="${WORKSTATION_HOME:-/home/franka}"

HERE="$(cd "$(dirname "$0")" && pwd)"
usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || usage
SUITE=$1
shift
TASKS=""
BACKENDS="sail bspline"
STEPS=""
APPLY=false
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tasks)   TASKS="$2"; shift 2 ;;
        --backend) BACKENDS="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --apply)   APPLY=true; shift ;;
        --)        shift; PASS=("$@"); break ;;
        *)         echo "unknown argument: $1" >&2; usage ;;
    esac
done
# Both trainers default to different budgets; one number trains them equally long.
[ -n "$STEPS" ] || { echo "--steps is required" >&2; exit 2; }

HOST_PREP="$CLUSTER_HOME/franka_data/baseline_prep/$SUITE"
[ -d "$HOST_PREP" ] || { echo "no $HOST_PREP; rsync the converted tasks up first" >&2; exit 1; }
[ -f "$SIF" ] || { echo "no $SIF; apptainer pull it first" >&2; exit 1; }
if [ -z "$TASKS" ]; then
    TASKS=$(find "$HOST_PREP" -maxdepth 1 -name 'task_*' -printf '%f\n' | sed 's/^task_//' | sort -n)
fi

UNITS=()
ERRORS=0
for t in $TASKS; do
    index=${t%%:*}
    index=${index#task_}
    only=""
    [ "$t" != "${t#*:}" ] && only=${t#*:}
    for b in $BACKENDS; do
        case "$b" in sail|bspline) ;; *) echo "unknown backend: $b" >&2; exit 2 ;; esac
        [ -n "$only" ] && [ "$b" != "$only" ] && continue
        if [ ! -f "$HOST_PREP/task_$index/$b.hdf5" ]; then
            echo "missing $HOST_PREP/task_$index/$b.hdf5" >&2
            ERRORS=$((ERRORS + 1))
            continue
        fi
        UNITS+=("task_$index:$b")
    done
done
[ "$ERRORS" -eq 0 ] || { echo "$ERRORS missing dataset(s); refusing to submit" >&2; exit 1; }
[ "${#UNITS[@]}" -gt 0 ] || { echo "nothing to train" >&2; exit 1; }

LOG_DIR="$HOST_PREP/slurm_logs"
cmd=(sbatch --array="0-$((${#UNITS[@]} - 1))" --job-name="$SUITE"
     --output="$LOG_DIR/%x_%A_%a.out"
     --export="ALL,SIF=$SIF,CLUSTER_HOME=$CLUSTER_HOME,WORKSTATION_HOME=$WORKSTATION_HOME"
     "$HERE/train_libero.slurm" "$SUITE" "$STEPS" "$(IFS=,; echo "${UNITS[*]}")" "${PASS[@]}")

for i in "${!UNITS[@]}"; do
    echo "  [$i] ${UNITS[$i]}"
done
echo "${cmd[@]}"
if $APPLY; then
    mkdir -p "$LOG_DIR"
    "${cmd[@]}"
else
    echo "Dry run. Re-run with --apply to submit."
fi
