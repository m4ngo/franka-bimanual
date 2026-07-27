#!/usr/bin/env bash
# Deploy the pylibfranka torque server to a NUC and restart it.
#
# The server and the OSC controller are copied flat into ~/ on the NUC so the
# server's `from osc_torque_controller import ...` fallback resolves without a
# package install. Restarting kills whatever is bound to the port first --
# including the old net_franky `rpyc_classic` server, which cannot coexist with
# this one on 18812.
#
# Usage: deploy_nuc_server.sh [mario|luigi] [--no-restart]
set -euo pipefail

TARGET="${1:-mario}"
RESTART=1
[[ "${2:-}" == "--no-restart" ]] && RESTART=0

case "$TARGET" in
  mario) HOST=mario@192.168.3.10; PORT=18812 ;;
  luigi) HOST=luigi@192.168.3.11; PORT=18813 ;;
  *) echo "unknown target '$TARGET' (expected mario or luigi)" >&2; exit 1 ;;
esac

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka"

echo "==> copying server to $HOST"
scp -q "$SRC/pylibfranka_server.py" "$SRC/osc_torque_controller.py" "$SRC/franka_jacobian.py" "$HOST:~/"

echo "==> writing run_server.sh"
ssh "$HOST" "cat > ~/run_server.sh" <<EOF
#!/usr/bin/env bash
# pylibfranka RPyC torque server for the FR3 1 kHz control loop.
#
# SCHED_FIFO prio 80 (chrt -f) so the libfranka realtime thread outranks normal
# nice-0 kernel workers -- without it the loop runs at the inherited nice and the
# OS preempts it past the 1 kHz deadline -> "UDP receive: Timeout". taskset keeps
# it off CPU0/1 where ACPI/PM kworkers cluster. (User must be in @realtime.)
#
# Unlike the old net_franky launcher this runs our own module rather than the
# generic rpyc_classic entrypoint: the torque loop has to live inside the
# process that holds the ActiveControlBase.
source ~/pylibfranka_env/bin/activate
exec taskset -c 2-7 chrt -f 80 python ~/pylibfranka_server.py --port $PORT --host 0.0.0.0
EOF
ssh "$HOST" "chmod +x ~/run_server.sh"

echo "==> import check"
ssh "$HOST" "source ~/pylibfranka_env/bin/activate && python -c '
import ast, sys
for f in (\"pylibfranka_server.py\", \"osc_torque_controller.py\", \"franka_jacobian.py\"):
    ast.parse(open(f).read())
import pylibfranka, rpyc, numpy
print(\"pylibfranka\", pylibfranka.__version__ if hasattr(pylibfranka, \"__version__\") else \"ok\", \"rpyc\", rpyc.__version__)
import osc_torque_controller, franka_jacobian
print(\"osc_torque_controller ok\")
'"

if [[ "$RESTART" == "1" ]]; then
  # Bracketed first char so the pattern never matches the shell running pkill --
  # 'rpyc_classic -p 18812' appears verbatim in that shell's own argv, so an
  # unbracketed -f pattern makes pkill kill itself before it kills the server.
  echo "==> stopping anything on port $PORT"
  ssh "$HOST" "pkill -f '[r]pyc_classic -p $PORT' || true; pkill -f '[p]ylibfranka_server.py --port $PORT' || true; sleep 1"
  # The redirect binds to the subshell, not to run_server.sh: bash forks a
  # subshell for the backgrounded compound, and that subshell inheriting ssh's
  # stdout keeps the channel open, so ssh never returns even though the daemon
  # itself is fully detached.
  echo "==> starting server"
  ssh -n "$HOST" "(cd ~ && setsid ./run_server.sh) </dev/null >~/pylibfranka_server.log 2>&1 &"
  sleep 3
  ssh -n "$HOST" "tail -5 ~/pylibfranka_server.log; pgrep -af '[p]ylibfranka_server.py' || { echo 'SERVER NOT RUNNING' >&2; exit 1; }"
fi

echo "==> done ($TARGET)"
