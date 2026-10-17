#!/usr/bin/env bash
# How far along each training is, from its logs (baselines/LIBERO_SIM.md,
# "Training on Tillicum"). Run on the login node:
#
#   bash progress.sh real                                          # SAIL and B-Spline, train_logs
#   bash progress.sh libero_10
#   bash progress.sh diffusion HuskyMango/filter-10-6-bowl-plate   # our policy, slurm_logs
#
# One line per unit: progress against the total, its rate, and the time left at that rate.

set -euo pipefail

TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/scrubbed/$USER}"
CLUSTER_HOME="${CLUSTER_HOME:-$TILLICUM_ROOT/franka_home}"
usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

if [ "${1:-}" = diffusion ]; then
    [ $# -eq 2 ] || usage
    REPO_ID=$2
    LOGS="$CLUSTER_HOME/franka_data/policies/$REPO_ID/diffusion/slurm_logs"
    [ -d "$LOGS" ] || { echo "no $LOGS yet; has a job started?" >&2; exit 1; }
    if command -v squeue >/dev/null; then
        squeue -u "$USER" -n "diffusion-${REPO_ID##*/}" -o "%.18i %.9T %.11M %.11l %R"
        echo
    fi
    now=$(date +%s)
    seen=" "
    # Newest log first; a resubmitted run has one log per job, and only its latest counts.
    for log in $(ls -t "$LOGS"/*.out 2>/dev/null); do
        run=$(sed -n 's/^diffusion .* -> \(.*\), job.*/\1/p' "$log" | head -1)
        run=${run##*/}
        [ -n "$run" ] || continue
        case "$seen" in *" $run "*) continue ;; esac
        seen="$seen$run "
        age=$(( now - $(stat -c %Y "$log") ))
        tr '\r' '\n' < "$log" | awk -v unit="diffusion/$run" -v age="$age" -v logf="$log" '
            function hms(s) { return sprintf("%dh%02dm", s / 3600, (s % 3600) / 60) }
            /^steps [0-9]+, batch/ { total = $2 + 0 }
            /^already trained to/  { finished = 1 }
            /End of training/      { finished = 1 }
            /^Training: / && match($0, /[0-9]+\/[0-9]+ \[[^]]*\]/) {
                bar = substr($0, RSTART, RLENGTH)
                split(bar, a, /[\/ \[<,\]]+/); n = a[1] + 0; N = a[2] + 0; left = a[4]
                if (match(bar, /[0-9.]+(step\/s|s\/step)/)) rate = substr(bar, RSTART, RLENGTH)
            }
            /^[A-Za-z_.]*(Error|Exception)(:|$)/ { err = $0 }
            END {
                status = (age > 300) ? sprintf("log quiet for %s", hms(age)) : "log live"
                if (finished) { printf "%-34s finished\n", unit; exit }
                if (err != "") { printf "%-34s FAILED: %s (%s)\n", unit, substr(err, 1, 120), logf; exit }
                if (N == 0) { printf "%-34s starting (%s)\n", unit, status; exit }
                done = total - N + n
                printf "%-34s step %d/%d (%.0f%%), %s, ~%s left (%s)\n",
                       unit, done, total, 100 * done / total, rate, left, status
            }'
    done
    exit 0
fi

[ $# -eq 1 ] || usage
SUITE=$1
LOGS="$CLUSTER_HOME/franka_data/baseline_prep/$SUITE/train_logs"
[ -d "$LOGS" ] || { echo "no $LOGS yet; has a job started?" >&2; exit 1; }

if command -v squeue >/dev/null; then
    squeue -u "$USER" -n "$SUITE" -o "%.18i %.9T %.11M %.11l %R"
    echo
fi

now=$(date +%s)
for log in "$LOGS"/*.log; do
    [ -e "$log" ] || continue
    unit=$(basename "$log" .log)
    age=$(( now - $(stat -c %Y "$log") ))
    # tqdm redraws its bar with \r; each redraw becomes its own line.
    tr '\r' '\n' < "$log" | awk -v unit="$unit" -v age="$age" '
        function hms(s) { return sprintf("%dh%02dm", s / 3600, (s % 3600) / 60) }
        # SAIL (robomimic). Its fork trains epochs 0..N, so N + 1 of them; each
        # "Train Epoch k" is printed once epoch k has finished.
        /^--steps [0-9]+ -> [0-9]+ epochs of/ { total = $4 + 1; per = $7; sail = 1 }
        /^Train Epoch [0-9]+/                  { done = $3 + 1 }
        /"Time_Epoch":/                        { gsub(/,/, "", $2); min_per_epoch = $2 }
        /Saving Waypoints:/                    { awe = $0 }
        # B-Spline (hydra + tqdm)
        /^--steps [0-9]+: .* steps per epoch -> [0-9]+ epochs/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "->") total = $(i + 1)
                if ($i == "steps" && $(i + 1) == "per") per = $(i - 1)
            }
        }
        /^Training epoch [0-9]+:/ {
            ep = $3; sub(/:/, "", ep)
            if (match($0, /\| *[0-9]+\/[0-9]+ \[/)) { split(substr($0, RSTART + 1), f, "/"); k = f[1] + 0 }
            if (match($0, /[0-9.]+it\/s/)) rate = substr($0, RSTART, RLENGTH - 4) + 0
            else if (match($0, /[0-9.]+s\/it/)) rate = 1 / substr($0, RSTART, RLENGTH - 4)
        }
        /^trained in [0-9]+s/ { finished = $3 }
        /already holds the whole run/ { finished = 0.1 }
        END {
            status = (age > 300) ? sprintf("log quiet for %s", hms(age)) : "log live"
            if (finished != "") { printf "%-34s finished, trained in %s\n", unit, hms(finished + 0); exit }
            if (total == "") { printf "%-34s starting (%s)\n", unit, status; exit }
            if (sail) {
                if (done == 0 && awe != "") {
                    match(awe, /[0-9]+\/[0-9]+ \[[^],]*/)
                    printf "%-34s labelling waypoints %s], before training (%s)\n", unit, substr(awe, RSTART, RLENGTH), status
                    exit
                }
                if (done == 0) { printf "%-34s loading data, no epoch finished yet (%s)\n", unit, status; exit }
                left = (total - done) * min_per_epoch * 60
                printf "%-34s epoch %d/%d (%.0f%%), %.2f min/epoch, ~%s left (%s)\n",
                       unit, done, total, 100 * done / total, min_per_epoch, hms(left), status
            } else {
                if (ep == "") { printf "%-34s loading data, no epoch finished yet (%s)\n", unit, status; exit }
                steps = ep * per + k
                left = (rate > 0) ? (total * per - steps) / rate : 0
                printf "%-34s epoch %d/%d (%.0f%%), step %d of %d, %.1f it/s, ~%s left (%s)\n",
                       unit, ep, total, 100 * steps / (total * per), steps, total * per, rate, hms(left), status
            }
        }'
done
