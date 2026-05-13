#!/usr/bin/env bash
# One-shot bring-up: preflight network, open Franka Desk in Chrome,
# launch start_control.sh on both NUCs, wait for RPyC to come up.
#
# Manual steps still required:
#   - Power on both Franka control boxes (physical switch).
#   - Click "Unlock" and "Enable FCI" in each Chrome tab once it loads.
#
# First-time setup:
#   ./bringup.sh --setup-keys   # copies your SSH key to mario and luigi
#                               # (one-time; key auth is required after that)

set -u

LEFT_NUC_HOST="luigi@192.168.3.11"
RIGHT_NUC_HOST="mario@192.168.3.10"
LEFT_ARM_IP="192.168.200.2"
RIGHT_ARM_IP="192.168.201.10"
LEFT_RPYC_PORT=18813
RIGHT_RPYC_PORT=18812
CAMERA_IPS=(192.168.0.142 192.168.0.116 192.168.1.138 192.168.1.139 192.168.1.143 192.168.1.102)
TMUX_SESSION="franka-control"
READY_TIMEOUT=45

C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YLW=$'\033[33m'; C_BLU=$'\033[34m'; C_RST=$'\033[0m'
ok()   { printf "  ${C_GRN}[ ok ]${C_RST} %s\n" "$*"; }
warn() { printf "  ${C_YLW}[warn]${C_RST} %s\n" "$*"; }
fail() { printf "  ${C_RED}[fail]${C_RST} %s\n" "$*"; }
step() { printf "\n${C_BLU}==>${C_RST} %s\n" "$*"; }

check_tcp() { timeout 2 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null; }
check_ping() { ping -c 1 -W 1 "$1" >/dev/null 2>&1; }

if [[ "${1:-}" == "--setup-keys" ]]; then
    step "One-time SSH key setup for mario and luigi"
    [[ -f ~/.ssh/id_ed25519 ]] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
    ssh-copy-id "$RIGHT_NUC_HOST"
    ssh-copy-id "$LEFT_NUC_HOST"
    ok "Done. Re-run ./bringup.sh with no arguments."
    exit 0
fi

step "Preflight: network"
preflight_fail=0
for host_label in "left NUC $LEFT_NUC_HOST" "right NUC $RIGHT_NUC_HOST" \
                  "left arm $LEFT_ARM_IP" "right arm $RIGHT_ARM_IP"; do
    ip="${host_label##* }"
    if check_ping "${ip#*@}"; then ok "ping $host_label"
    else fail "ping $host_label — is it powered on / on the right subnet?"; preflight_fail=1; fi
done

cam_down=()
for cam in "${CAMERA_IPS[@]}"; do
    check_ping "$cam" || cam_down+=("$cam")
done
if (( ${#cam_down[@]} == 0 )); then ok "all 6 cameras reachable"
else warn "cameras down: ${cam_down[*]} (record/teleop may degrade those frames)"; fi

if (( preflight_fail )); then
    fail "Preflight failed on a critical host. Fix power/network and re-run."
    exit 1
fi

step "Verifying SSH key auth to NUCs"
ssh_fail=0
for h in "$LEFT_NUC_HOST" "$RIGHT_NUC_HOST"; do
    if timeout 4 ssh -o BatchMode=yes -o ConnectTimeout=3 "$h" 'true' 2>/dev/null; then
        ok "ssh $h"
    else
        fail "ssh $h — key auth not set up"
        ssh_fail=1
    fi
done
if (( ssh_fail )); then
    fail "Run: ./bringup.sh --setup-keys   (one-time; you'll be prompted for the NUC passwords)"
    exit 1
fi

step "Opening Franka Desk in Chrome (Unlock + Enable FCI manually in each tab)"
nohup google-chrome --new-window "https://$LEFT_ARM_IP" "https://$RIGHT_ARM_IP" \
    >/dev/null 2>&1 &
disown
ok "Chrome launched"

step "Starting controllers on both NUCs"
start_remote_controller() {
    local host="$1"
    # Idempotent: if a tmux session with the controller already exists, leave it alone.
    ssh -o BatchMode=yes "$host" "
        if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
            echo 'already-running'
        else
            tmux new-session -d -s $TMUX_SESSION './start_control.sh'
            echo 'started'
        fi
    "
}
for h in "$LEFT_NUC_HOST" "$RIGHT_NUC_HOST"; do
    result=$(start_remote_controller "$h" 2>&1) || { fail "ssh failure on $h: $result"; exit 1; }
    case "$result" in
        already-running) ok "$h: controller already running" ;;
        started)         ok "$h: controller started in tmux ($TMUX_SESSION)" ;;
        *)               warn "$h: unexpected response: $result" ;;
    esac
done

step "Waiting for RPyC ports to come up (timeout ${READY_TIMEOUT}s)"
deadline=$(( SECONDS + READY_TIMEOUT ))
left_ok=0; right_ok=0
while (( SECONDS < deadline )); do
    (( left_ok ))  || { check_tcp "${LEFT_NUC_HOST#*@}"  "$LEFT_RPYC_PORT"  && { left_ok=1;  ok "luigi rpyc :$LEFT_RPYC_PORT"; }; }
    (( right_ok )) || { check_tcp "${RIGHT_NUC_HOST#*@}" "$RIGHT_RPYC_PORT" && { right_ok=1; ok "mario rpyc :$RIGHT_RPYC_PORT"; }; }
    (( left_ok && right_ok )) && break
    sleep 1
done
if (( ! left_ok || ! right_ok )); then
    fail "RPyC port(s) never came up. Check the tmux session on the NUC:"
    fail "  ssh $LEFT_NUC_HOST 'tmux attach -t $TMUX_SESSION'"
    fail "  ssh $RIGHT_NUC_HOST 'tmux attach -t $TMUX_SESSION'"
    exit 1
fi

step "Opening live view of controllers (side-by-side panes via local tmux)"
if ! command -v tmux >/dev/null 2>&1; then
    fail "tmux not installed locally. Install with: sudo apt install -y tmux"
    exit 1
fi
# One gnome-terminal window, one tab, containing a local tmux session
# 'franka-view' with two vertical panes — each ssh-attached to a NUC's remote
# 'franka-control' session. The local tmux prefix is remapped to Ctrl-A so
# Ctrl-B passes through unchanged to the remote tmux running inside each pane.
# Closing the window leaves both the remote controllers AND the local view
# session alive; re-running bringup.sh re-attaches to the existing view.
nohup gnome-terminal --window --title="franka controllers" -- bash -c '
    SESSION=franka-view
    LEFT_HOST="$1"
    RIGHT_HOST="$2"
    REMOTE_TMUX="$3"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        exec tmux attach -t "$SESSION"
    fi
    tmux new-session  -d -s "$SESSION" "ssh -t $LEFT_HOST  \"tmux attach -t $REMOTE_TMUX\""
    tmux split-window -h -t "$SESSION" "ssh -t $RIGHT_HOST \"tmux attach -t $REMOTE_TMUX\""
    tmux set-option   -g prefix C-a
    tmux set-option   -g mouse on
    exec tmux attach -t "$SESSION"
' _ "$LEFT_NUC_HOST" "$RIGHT_NUC_HOST" "$TMUX_SESSION" >/dev/null 2>&1 &
disown
ok "Live view opened"

printf "\n${C_GRN}READY.${C_RST} Next steps:\n"
printf "  1. In each Chrome tab: Unlock + Enable FCI (green light on the arm).\n"
printf "  2. Activate venv:   ${C_BLU}source ~/.venv/bin/activate${C_RST}\n"
printf "  3. Run one of:      scripts/teleop.sh, scripts/record_data.sh, scripts/rollout_policy.sh\n"
printf "\n  Live view: ${C_BLU}Ctrl-A${C_RST} = local prefix (switch panes / detach with D).\n"
printf "             ${C_BLU}Ctrl-B${C_RST} passes through to the remote controller's tmux.\n"
