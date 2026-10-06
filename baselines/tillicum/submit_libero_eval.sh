#!/usr/bin/env bash
# Roll LIBERO policies out on Tillicum (baselines/LIBERO_SIM.md, "Evaluating on
# Tillicum"): pi05, SAIL at each OSC gain setting and B-Spline at each speed,
# one sweep id per setting, named as libero_report_ep50 names them. Prints the
# sbatch line; --apply submits it.
#
#   bash submit_libero_eval.sh libero_goal --apply
#   bash submit_libero_eval.sh libero_goal --methods sail --sail-gains "300:0.5" --tasks "0 1" --apply
#   bash submit_libero_eval.sh libero_goal --episodes 1 --prefix smoke --apply
#
# A unit is one task at one setting; --per-gpu units share a GPU. Anything after
# -- goes to every rollout. Resubmitting skips units their sweep has completed.

set -euo pipefail

TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/scrubbed/$USER}"
CLUSTER_HOME="${CLUSTER_HOME:-$TILLICUM_ROOT/franka_home}"
SIF="${SIF:-$TILLICUM_ROOT/containers/franka-sim-eval.sif}"
WORKSTATION_HOME="${WORKSTATION_HOME:-/home/franka}"

HERE="$(cd "$(dirname "$0")" && pwd)"
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || usage
SUITE=$1
shift
TASKS=""
METHODS="pi05 sail bspline"
# kp:damping_ratio; the default and three inside SAIL's paper range (LIBERO_SIM.md).
SAIL_GAINS="300:0.5 1000:1.0 2000:0.75 3000:0.5"
BSPLINE_SPEEDS="1 2 4 8"
EPISODES=50
PREFIX=""
PER_GPU=4
SAVE_VIDEO=""
APPLY=false
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tasks)          TASKS="$2"; shift 2 ;;
        --methods)        METHODS="$2"; shift 2 ;;
        --sail-gains)     SAIL_GAINS="$2"; shift 2 ;;
        --bspline-speeds) BSPLINE_SPEEDS="$2"; shift 2 ;;
        --episodes)       EPISODES="$2"; shift 2 ;;
        --prefix)         PREFIX="$2"; shift 2 ;;
        --per-gpu)        PER_GPU="$2"; shift 2 ;;
        --save-video)     SAVE_VIDEO="--save-video"; shift ;;
        --apply)          APPLY=true; shift ;;
        --)               shift; PASS=("$@"); break ;;
        *)                echo "unknown argument: $1" >&2; usage ;;
    esac
done
PREFIX=${PREFIX:-ep$EPISODES}

HOST_PREP="$CLUSTER_HOME/franka_data/baseline_prep/$SUITE"
POLICIES="$CLUSTER_HOME/franka_data/policies/$SUITE"
[ -d "$HOST_PREP" ] || { echo "no $HOST_PREP; rsync the converted tasks up first" >&2; exit 1; }
[ -f "$SIF" ] || { echo "no $SIF; apptainer pull it first" >&2; exit 1; }
if [ -z "$TASKS" ]; then
    TASKS=$(find "$HOST_PREP" -maxdepth 1 -name 'task_*' -printf '%f\n' | sed 's/^task_//' | sort -n)
fi

# setting -> "backend|sweep id|rollout args"
SETTINGS=()
for m in $METHODS; do
    case "$m" in
        pi05) SETTINGS+=("pi05|${PREFIX}_pi05|") ;;
        sail)
            for g in $SAIL_GAINS; do
                kp=${g%%:*}
                zeta=${g#*:}
                tag=$(awk -v z="$zeta" 'BEGIN { printf "%03d", z * 100 + 0.5 }')
                SETTINGS+=("sail|${PREFIX}_sail_kp${kp}_d${tag}|--osc-kp $kp --osc-damping-ratio $zeta")
            done ;;
        bspline)
            for s in $BSPLINE_SPEEDS; do
                SETTINGS+=("bspline|${PREFIX}_bsp_${s}x|--speed-up-times $s")
            done ;;
        *) echo "unknown method: $m" >&2; exit 2 ;;
    esac
done

# Setting-major, so an array element mostly holds one setting's tasks.
UNITS=()
UNTRAINED=()
for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r backend sweep rollout_args <<< "$setting"
    for t in $TASKS; do
        task=task_${t#task_}
        if [ "$backend" != pi05 ] && [ ! -f "$POLICIES/$task/$backend/.trained.json" ]; then
            UNTRAINED+=("$task:$backend")
            continue
        fi
        UNITS+=("$task|$backend|$sweep|--num-episodes $EPISODES $SAVE_VIDEO|$rollout_args ${PASS[*]:-}")
    done
done
if [ "${#UNTRAINED[@]}" -gt 0 ]; then
    echo "not trained, left out: $(printf '%s\n' "${UNTRAINED[@]}" | sort -uV | tr '\n' ' ')" >&2
fi
[ "${#UNITS[@]}" -gt 0 ] || { echo "nothing to evaluate" >&2; exit 1; }

ELEMENTS=$(( (${#UNITS[@]} + PER_GPU - 1) / PER_GPU ))
UNITS_FILE="$HOST_PREP/eval_units/${PREFIX}_$(date +%Y%m%d_%H%M%S).txt"
LOG_DIR="$HOST_PREP/slurm_logs"
cmd=(sbatch --array="0-$((ELEMENTS - 1))" --job-name="eval_$SUITE"
     --output="$LOG_DIR/%x_%A_%a.out"
     --export="ALL,SIF=$SIF,CLUSTER_HOME=$CLUSTER_HOME,WORKSTATION_HOME=$WORKSTATION_HOME"
     "$HERE/eval_libero.slurm" "$SUITE" "$UNITS_FILE" "$PER_GPU")

for i in "${!UNITS[@]}"; do
    echo "  [$((i / PER_GPU))] ${UNITS[$i]}"
done
echo "${#UNITS[@]} unit(s) on $ELEMENTS GPU(s)"
echo "${cmd[@]}"
if $APPLY; then
    mkdir -p "$LOG_DIR" "$(dirname "$UNITS_FILE")"
    printf '%s\n' "${UNITS[@]}" > "$UNITS_FILE"
    "${cmd[@]}"
else
    echo "Dry run. Re-run with --apply to submit."
fi
