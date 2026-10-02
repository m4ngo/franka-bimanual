#!/usr/bin/env bash
# How far along each training of a suite is, from its train_logs (baselines/LIBERO_SIM.md,
# "Training on Tillicum"). Run on the login node:
#
#   bash progress.sh real
#   bash progress.sh libero_10
#
# One line per unit: epochs done of the total, its rate, and the time left at that rate.

set -euo pipefail

TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/scrubbed/$USER}"
CLUSTER_HOME="${CLUSTER_HOME:-$TILLICUM_ROOT/franka_home}"

[ $# -eq 1 ] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
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
        # SAIL (robomimic)
        /^--steps [0-9]+ -> [0-9]+ epochs of/ { total = $4; per = $7; sail = 1 }
        /^Train Epoch [0-9]+/                  { done = $3 }
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
