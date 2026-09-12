#!/usr/bin/env bash

# Roll a trained SAIL policy out on the arm.
#
# Two processes: the policy lives in the `SAIL` conda env (python 3.9, torch 2.1,
# patched robosuite) which cannot coexist with this workspace's venv, so it runs
# behind ZMQ. With --start-server this script launches it; otherwise start it
# yourself and point --port at it.
#
# The control mode is resolved from the checkpoint, not chosen here -- see
# baselines/ROLLOUT.md.
#
#   ./scripts/sail_rollout.sh --start-server --ckpt ~/franka_data/sail/best.pth \
#       --rig=single_arm_right --num-episodes 10
#
# Everything not consumed below is passed through to the python entrypoint.

set -euo pipefail
# Job control: each background job gets its own process group whose PGID
# is $!, which is what cleanup() below kills. Without it the server's
# python child survives the trap.
set -m
source "$(dirname "$0")/_config.sh"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="$(cfg policy.baselines.zmq.sail_port)"
CONDA_ENV="${SAIL_CONDA_ENV:-SAIL}"

START_SERVER=0
CKPT=""
GUIDE=""
RIG=single_arm_franka
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-server) START_SERVER=1; shift ;;
        --ckpt)         CKPT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --ckpt=*)       CKPT="${1#--ckpt=}"; ARGS+=("$1"); shift ;;
        # Consumed by the server, not the client.
        --guide-config) GUIDE="$2"; shift 2 ;;
        --guide-config=*) GUIDE="${1#--guide-config=}"; shift ;;
        --port)         PORT="$2"; shift 2 ;;
        --port=*)       PORT="${1#--port=}"; shift ;;
        --rig)          RIG="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --rig=*)        RIG="${1#--rig=}"; ARGS+=("$1"); shift ;;
        *)              ARGS+=("$1"); shift ;;
    esac
done

SERVER_PGID=""
cleanup() {
    # Kill the GROUP, not the pid: `conda run` spawns python as a child, so
    # killing conda alone orphans the server holding the port and the GPU.
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
    SRV=("$REPO_ROOT/baselines/sail_bridge/policy_server.py"
         --ckpt-path "$CKPT" --port "$PORT")
    [[ -n "$GUIDE" ]] && SRV+=(--guide-config "$GUIDE")
    echo "starting the SAIL policy server in conda env '${CONDA_ENV}' on port ${PORT}"
    conda run --no-capture-output -n "$CONDA_ENV" python "${SRV[@]}" &
    SERVER_PGID=$!
    # Wait on the handshake rather than a fixed sleep: loading a diffusion
    # checkpoint onto the GPU takes tens of seconds and varies.
    python "$(dirname "$0")/_wait_policy_server.py" sail "$PORT"
fi

echo "sail rollout on ${RIG} (goals faster than ${CONTROL_FPS} Hz; see --exec-fps)"
cd "$REPO_ROOT"
STATUS=0
python -m baselines.sail_bridge.rollout --port "$PORT" "${ARGS[@]}" || STATUS=$?
# Not `exec`: that would replace this shell and discard the EXIT trap,
# orphaning the policy server started above. A stale server keeps the
# port, so the next run would handshake with it and silently evaluate
# the PREVIOUS checkpoint under the new one's recorded sha256.
exit "$STATUS"
