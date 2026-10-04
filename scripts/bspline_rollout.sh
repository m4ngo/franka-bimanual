#!/usr/bin/env bash

# Roll a trained B-Spline Policy out on the arm.
#
# Two processes: the policy lives in its own venv (.venv-bspline, made by
# scripts/setup_baseline_envs.sh) which cannot coexist with this workspace's,
# so it runs behind ZMQ. With --start-server this script launches
# baselines/bspline_bridge/policy_server.py, which subclasses upstream's own
# server to add the meta handshake and changes nothing else -- upstream is a
# submodule and stays unedited. $BSPLINE_PYTHON overrides which interpreter
# runs the server (baselines/interpreters.py).
#
# The server returns spline PARAMETERS, so this side owns the clock and no
# per-tick round trip lands between the state read and the goal write.
#
#   ./scripts/bspline_rollout.sh --start-server --ckpt ~/franka_data/bsp/latest.ckpt \
#       --rig=single_arm_right --speed 2 --num-episodes 10
#
# Everything not consumed below is passed through to the python entrypoint.

set -euo pipefail
# Job control: each background job gets its own process group whose PGID
# is $!, which is what cleanup() below kills. Without it the server's
# python child survives the trap.
set -m
source "$(dirname "$0")/_config.sh"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="$(cfg policy.baselines.zmq.bspline_port)"
SERVER="$REPO_ROOT/baselines/bspline_bridge/policy_server.py"

START_SERVER=0
CKPT=""
RIG=single_arm_franka
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-server) START_SERVER=1; shift ;;
        --ckpt)         CKPT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --ckpt=*)       CKPT="${1#--ckpt=}"; ARGS+=("$1"); shift ;;
        --port)         PORT="$2"; shift 2 ;;
        --port=*)       PORT="${1#--port=}"; shift ;;
        --rig)          RIG="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --rig=*)        RIG="${1#--rig=}"; ARGS+=("$1"); shift ;;
        *)              ARGS+=("$1"); shift ;;
    esac
done

SERVER_PGID=""
cleanup() {
    # Kill the GROUP, not the pid: a `conda run` interpreter spawns python as a
    # child, and killing the parent alone orphans the server holding the port.
    [[ -n "$SERVER_PGID" ]] || return 0
    kill -TERM -"$SERVER_PGID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 -"$SERVER_PGID" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL -"$SERVER_PGID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$START_SERVER" == 1 ]]; then
    [[ -n "$CKPT" ]] || { echo "--start-server needs --ckpt" >&2; exit 1; }
    mapfile -t BSP_PY < <(python -m baselines.interpreters bspline)
    echo "starting the B-Spline policy server with '${BSP_PY[*]}' on port ${PORT}"
    # n_obs_steps comes from the checkpoint; the server refuses a mismatch.
    "${BSP_PY[@]}" "$SERVER" --ckpt-path "$CKPT" --port "$PORT" &
    SERVER_PGID=$!
    python "$(dirname "$0")/_wait_policy_server.py" bspline "$PORT"
fi

echo "bspline rollout on ${RIG} in EE_POS"
cd "$REPO_ROOT"
STATUS=0
python -m baselines.bspline_bridge.rollout --port "$PORT" "${ARGS[@]}" || STATUS=$?
# Not `exec`: that would replace this shell and discard the EXIT trap,
# orphaning the policy server started above. A stale server keeps the
# port, so the next run would handshake with it and silently evaluate
# the PREVIOUS checkpoint under the new one's recorded sha256.
exit "$STATUS"
