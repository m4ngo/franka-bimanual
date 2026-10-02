#!/usr/bin/env bash

# Roll a trained SAIL or B-Spline policy out on LIBERO, in simulation.
#
# Three interpreters, none of which can be merged: the policy runs in
# .venv-sail / .venv-bspline (baselines/interpreters.py), the LIBERO env in
# multi-fast/.venv -- the only one with robosuite and libero -- and this
# script's `cfg` helper in the workspace venv. With --start-server this script
# launches the policy server; otherwise start it yourself and pass --port.
#
#   ./scripts/libero_rollout.sh --backend sail --start-server \
#       --ckpt ~/franka_data/policies/libero_10/task_2/sail/<run>/models/model_epoch_N.pth \
#       --suite libero_10 --task 2 --num-episodes 20
#
# Everything not consumed below is passed through to the python entrypoint.

set -euo pipefail
# Job control: each background job gets its own process group whose PGID is $!,
# which is what cleanup() kills. Without it the server's python child survives.
set -m
source "$(dirname "$0")/_config.sh"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIM_PYTHON="${SIM_PYTHON:-$REPO_ROOT/multi-fast/.venv/bin/python}"

BACKEND=""
START_SERVER=0
CKPT=""
# SAIL's guidance config as its README evaluates; --guide-config "" runs unguided.
GUIDE="$REPO_ROOT/baselines/sail/robomimic/SAIL/guide_template/base_cfg_weight_1.json"
PORT=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      BACKEND="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --backend=*)    BACKEND="${1#--backend=}"; ARGS+=("$1"); shift ;;
        --start-server) START_SERVER=1; shift ;;
        --ckpt)         CKPT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
        --ckpt=*)       CKPT="${1#--ckpt=}"; ARGS+=("$1"); shift ;;
        # Consumed by the server, not the client.
        --guide-config) GUIDE="$2"; shift 2 ;;
        --guide-config=*) GUIDE="${1#--guide-config=}"; shift ;;
        --port)         PORT="$2"; shift 2 ;;
        --port=*)       PORT="${1#--port=}"; shift ;;
        *)              ARGS+=("$1"); shift ;;
    esac
done

case "$BACKEND" in
    # pi05 is multi-fast's base policy; it loads inside the rollout process
    # (openpi is already in multi-fast/.venv), so it has no server and no port.
    sail|bspline|pi05) ;;
    *) echo "--backend must be sail, bspline or pi05 (got '${BACKEND}')" >&2; exit 1 ;;
esac
if [[ "$BACKEND" != pi05 && -z "$PORT" ]]; then
    PORT="$(cfg "policy.baselines.zmq.${BACKEND}_port")"
fi
[[ -x "$SIM_PYTHON" ]] || {
    echo "no LIBERO interpreter at $SIM_PYTHON; set \$SIM_PYTHON" >&2; exit 1; }

# pid listening on $1, empty if the port is free. `ss` absent -> empty, so the
# check degrades to trusting the handshake rather than failing the run.
# `|| true` is required: under `set -e` with pipefail an empty grep fails the
# pipeline, and a failing command substitution in an assignment kills the script.
port_holder() {
    command -v ss >/dev/null 2>&1 || return 0
    ss -ltnp 2>/dev/null | grep -oE "[:.]$1[[:space:]].*pid=[0-9]+" |
        grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true
}

SERVER_PGID=""
cleanup() {
    # The GROUP and the pid: `conda run` spawns python as a child so the group
    # matters, but `kill -0` on the group succeeds while a dead leader is still a
    # zombie, so waiting on the port is what actually confirms it is gone.
    [[ -n "$SERVER_PGID" ]] || return 0
    kill -TERM -"$SERVER_PGID" 2>/dev/null || kill -TERM "$SERVER_PGID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        [[ -n "$(port_holder "$PORT")" ]] || break
        sleep 0.5
    done
    if [[ -n "$(port_holder "$PORT")" ]]; then
        kill -KILL -"$SERVER_PGID" 2>/dev/null || kill -KILL "$SERVER_PGID" 2>/dev/null || true
    fi
    wait "$SERVER_PGID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$BACKEND" == pi05 && "$START_SERVER" == 1 ]]; then
    echo "pi05 has no policy server; ignoring --start-server" >&2
    START_SERVER=0
fi

if [[ "$START_SERVER" == 1 ]]; then
    [[ -n "$CKPT" ]] || { echo "--start-server needs --ckpt" >&2; exit 1; }
    # A server already on this port answers our handshake indistinguishably from
    # one we started, so refuse before spending a checkpoint load on it.
    HOLDER="$(port_holder "$PORT")"
    if [[ -n "$HOLDER" ]]; then
        echo "port ${PORT} is already held by pid ${HOLDER}:" >&2
        ps -o pid=,etime=,cmd= -p "$HOLDER" >&2 2>/dev/null || true
        echo "that is a previous run's policy server. kill ${HOLDER} and retry." >&2
        exit 1
    fi
    SRV=("$REPO_ROOT/baselines/${BACKEND}_bridge/policy_server.py"
         --ckpt-path "$CKPT" --port "$PORT")
    [[ "$BACKEND" == sail && -n "$GUIDE" ]] && SRV+=(--guide-config "$GUIDE")
    mapfile -t POLICY_PY < <(python -m baselines.interpreters "$BACKEND")
    echo "starting the ${BACKEND} policy server with '${POLICY_PY[*]}' on port ${PORT}"
    "${POLICY_PY[@]}" "${SRV[@]}" &
    SERVER_PGID=$!
    # Wait on the handshake rather than a fixed sleep: loading a diffusion
    # checkpoint onto the GPU takes tens of seconds and varies.
    python "$(dirname "$0")/_wait_policy_server.py" "$BACKEND" "$PORT"
fi

echo "${BACKEND} rollout on LIBERO (stock robosuite plant)"
cd "$REPO_ROOT"
STATUS=0
PORT_ARG=()
[[ -n "$PORT" ]] && PORT_ARG=(--port "$PORT")
"$SIM_PYTHON" -m baselines.libero_bridge.rollout "${PORT_ARG[@]}" "${ARGS[@]}" || STATUS=$?
# Not `exec`: that would replace this shell and discard the EXIT trap, orphaning
# the policy server. A stale server keeps the port, so the next run would
# handshake with it and silently evaluate the PREVIOUS checkpoint.
exit "$STATUS"
