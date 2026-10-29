#!/usr/bin/env bash
# Our sim diffusion base's policy server inside one Slurm job, for multi-fast's `_dp`
# runs (teacher training, base stats, references, collection, eval). multi-fast's
# container launchers source this when a job exports DP_SERVER_SH=<this file>
# (multi-fast/scripts/task_config.sh, start_dp_server); by hand:
#
#   source /gpfs/scrubbed/$USER/tillicum/dp_server.sh
#   dp_server_start <checkpoint> <port> [log]   # returns once the server answers
#   ...                                         # the run, base_policy.server_port=<port>
#   dp_server_stop
#
# The server runs in $DP_SIF (Dockerfile.dp: the LeRobot image plus pyzmq) out of
# $FRANKA_SRC (sync.sh code), with $CLUSTER_HOME mounted at $WORKSTATION_HOME as
# train_diffusion.slurm mounts it. So <checkpoint> is the path the training job
# wrote, /home/franka/franka_data/policies/<dataset>/diffusion/<run>/checkpoints/last/pretrained_model,
# and the multi-fast run gets that same string as DP_BASE_CKPT: its base wrapper
# refuses a server holding any other.

# A Slurm job exported with --export=<list> may not have USER set.
TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/scrubbed/${USER:-$(id -un)}}"
DP_SIF="${DP_SIF:-$TILLICUM_ROOT/containers/lerobot-dp-server.sif}"
FRANKA_SRC="${FRANKA_SRC:-$TILLICUM_ROOT/franka_src}"
CLUSTER_HOME="${CLUSTER_HOME:-$TILLICUM_ROOT/franka_home}"
WORKSTATION_HOME="${WORKSTATION_HOME:-/home/franka}"

_dp_exec() {
    # The interpreter lerobot-train itself runs on, as train_diffusion.slurm finds it.
    apptainer exec --nv --containall --home "$CLUSTER_HOME:$WORKSTATION_HOME" \
        --bind "$FRANKA_SRC:/opt/franka_ws:ro" "$DP_SIF" \
        sh -c 'exec "$(dirname "$(command -v lerobot-train)")/python" "$@"' sh "$@"
}

dp_server_start() {
    local ckpt=$1 port=$2 log=${3:-/dev/stderr}
    [ -f "$DP_SIF" ] || { echo "no $DP_SIF; build baselines/tillicum/Dockerfile.dp and apptainer pull it" >&2; return 1; }
    [ -f "$FRANKA_SRC/baselines/dp_bridge/policy_server.py" ] || {
        echo "no $FRANKA_SRC/baselines/dp_bridge/policy_server.py; on the workstation: bash baselines/tillicum/sync.sh code" >&2
        return 1; }
    _dp_exec /opt/franka_ws/baselines/dp_bridge/policy_server.py --ckpt-path "$ckpt" --port "$port" \
        > "$log" 2>&1 &
    DP_SERVER_PID=$!
    # The handshake, not a sleep: load and warm-up times vary.
    if ! _dp_exec /opt/franka_ws/scripts/_wait_policy_server.py dp "$port" 300; then
        echo "the dp policy server on port $port did not answer; its log, $log:" >&2
        tail -20 "$log" >&2 2>/dev/null
        dp_server_stop
        return 1
    fi
}

dp_server_stop() {
    [ -n "${DP_SERVER_PID:-}" ] || return 0
    kill "$DP_SERVER_PID" 2>/dev/null
    wait "$DP_SERVER_PID" 2>/dev/null
    DP_SERVER_PID=""
}
