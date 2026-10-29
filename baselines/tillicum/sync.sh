#!/usr/bin/env bash
# Move code, data and results between the workstation and Tillicum. Run on the
# workstation from anywhere:
#
#   bash baselines/tillicum/sync.sh code                       # these scripts + the diffusion wrapper's and dp server's source
#   bash baselines/tillicum/sync.sh real filter-10-6-bowl-plate  # baseline_prep/real/task_<name> (or a path to it): sail.hdf5, bspline.hdf5, B-Spline's caches
#   bash baselines/tillicum/sync.sh lerobot HuskyMango/filter-10-6-bowl-plate   # the LeRobot recording, for our policy
#   bash baselines/tillicum/sync.sh fetch HuskyMango/filter-10-6-bowl-plate     # trained policies and logs back
#   bash baselines/tillicum/sync.sh sim ~/franka_data/baseline_prep/libero_90_sysid_2026_09_16/task_9   # a LIBERO task's sail.hdf5, bspline.hdf5
#   bash baselines/tillicum/sync.sh multifast                  # multi-fast's code and cfg (base stats included), for the ZAP jobs
#   bash baselines/tillicum/sync.sh student 1a2b3c4d           # a trained ZAP student back, into multi-fast/logs/distill/
#
# send_datasets.sh and fetch_policies.sh run these for one task. Every transfer
# resumes where it stopped when run again (--partial). TILLICUM_USER is the
# account (default atsai06); TILLICUM_DEST overrides the whole remote root, e.g.
# a local directory to try it out.

set -euo pipefail

TILLICUM_USER="${TILLICUM_USER:-atsai06}"
DEST="${TILLICUM_DEST:-$TILLICUM_USER@tillicum.hyak.uw.edu:/gpfs/scrubbed/$TILLICUM_USER}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RSYNC=(rsync -a --mkpath --partial --info=progress2)
if [[ "$DEST" == *:* ]]; then
    # One login (password + Duo) serves every rsync here and in later runs for
    # 30 min; Tillicum drops new logins that arrive right after one another.
    mkdir -p "$HOME/.ssh"
    RSYNC+=(-e "ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%C -o ControlPersist=30m")
fi
usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -ge 1 ] || usage
case "$1" in
    code)
        "${RSYNC[@]}" "$REPO/baselines/tillicum/" "$DEST/tillicum/"
        # What lerobot_train_recipe.py and the dp policy server import, laid out the
        # way they import it; the diffusion job binds this tree into the stock LeRobot
        # SIF, dp_server.sh into Dockerfile.dp's.
        stage=$(mktemp -d)
        trap 'rm -rf "$stage"' EXIT
        (cd "$REPO" && cp -p --parents \
            baselines/__init__.py baselines/diffusion_recipe.py baselines/train_common.py \
            baselines/run_record.py \
            baselines/bspline_policy/diffusion_policy/diffusion_policy/model/diffusion/ema_model.py \
            scripts/training/lerobot_train_recipe.py config/*.yaml \
            baselines/lerobot_inference.py baselines/dp_bridge/__init__.py \
            baselines/dp_bridge/policy_server.py baselines/libero_bridge/__init__.py \
            baselines/libero_bridge/sim_env.py scripts/_wait_policy_server.py "$stage/")
        # The EE-state step and what it imports, under an empty package: the real
        # __init__ loads the robot driver, which needs rpyc, and the image has none.
        mkdir -p "$stage/lerobot_robot_bimanual_franka" "$stage/franka_config"
        cp -p "$REPO"/lerobot_robot_bimanual_franka/lerobot_robot_bimanual_franka/{ee_state,ee_kinematics,franka_fk,lerobot_source}.py \
            "$stage/lerobot_robot_bimanual_franka/"
        echo '"""ee_state and the modules it imports, without the robot driver."""' \
            > "$stage/lerobot_robot_bimanual_franka/__init__.py"
        cp -p "$REPO"/franka_config/franka_config/*.py "$stage/franka_config/"
        "${RSYNC[@]}" "$stage/" "$DEST/franka_src/"
        ;;
    real)
        [ $# -eq 2 ] || usage
        case "$2" in
            */*) src=$(cd "$2" && pwd) ;;
            *)   src="$HOME/franka_data/baseline_prep/real/task_${2#task_}" ;;
        esac
        task=$(basename "$src")
        # submit_libero_train.sh puts task_ in front of every --tasks entry.
        [ "${task#task_}" != "$task" ] || { echo "$src: the folder must be named task_<dataset>" >&2; exit 1; }
        [ -f "$src/sail.hdf5" ] && [ -f "$src/bspline.hdf5" ] || { echo "no sail.hdf5/bspline.hdf5 in $src" >&2; exit 1; }
        # -a keeps mtimes, which B-Spline's cache name is keyed on.
        "${RSYNC[@]}" --exclude='*.lock' "$src/" \
            "$DEST/franka_home/franka_data/baseline_prep/real/$task/"
        ;;
    lerobot)
        [ $# -eq 2 ] || usage
        # Under ~/franka_data or the HF cache, wherever the recorder or the Hub put it.
        src=$("$REPO/.venv/bin/python" -c "from lerobot_robot_bimanual_franka.lerobot_source import resolve_root; print(resolve_root('$2'))")
        # images/ is the recorder's scratch; training reads videos/.
        "${RSYNC[@]}" --exclude images/ "$src/" "$DEST/franka_home/.cache/huggingface/lerobot/$2/"
        ;;
    fetch)
        [ $# -eq 2 ] || usage
        # B-Spline's top-k epoch=*.ckpt are ~1.5 GB each; rollout reads latest.ckpt.
        # training_state/ is our policy's optimizer and live weights, which only a
        # resume on the cluster reads.
        "${RSYNC[@]}" --exclude='epoch=*.ckpt' --exclude='training_state/' \
            "$DEST/franka_home/franka_data/policies/$2/" "$HOME/franka_data/policies/$2/"
        # A LIBERO task's id is <suite>_<plant>/task_<i>, its prep folder baseline_prep/<suite>_<plant>/.
        prep=real
        case "$2" in libero_*/task_*) prep=${2%%/*} ;; esac
        "${RSYNC[@]}" --include='train_logs/***' --include='slurm_logs/***' --exclude='*' \
            "$DEST/franka_home/franka_data/baseline_prep/$prep/" "$HOME/franka_data/baseline_prep/$prep/"
        ;;
    sim)
        [ $# -eq 2 ] || usage
        src=$(cd "$2" && pwd)
        suite=$(basename "$(dirname "$src")")
        task=$(basename "$src")
        [ "${task#task_}" != "$task" ] || { echo "$src: expected baseline_prep/<suite>/task_<i>" >&2; exit 1; }
        [ -f "$src/sail.hdf5" ] && [ -f "$src/bspline.hdf5" ] || { echo "no sail.hdf5/bspline.hdf5 in $src" >&2; exit 1; }
        "${RSYNC[@]}" --exclude='*.lock' "$src/" \
            "$DEST/franka_home/franka_data/baseline_prep/$suite/$task/"
        ;;
    multifast)
        # What multi-fast's jobs bind into multi-fast.sif: code, cfg and submodule sources.
        "${RSYNC[@]}" --exclude=.venv --exclude=.git --exclude='**/.git' --exclude=data \
            --exclude=logs --exclude=hydra_logs --exclude=wandb --exclude='**/__pycache__' \
            --exclude='*.pyc' --exclude='*.sif' --exclude='*.egg-info' \
            --exclude='scripts/cluster/*_tillicum.slurm' \
            "$REPO/multi-fast/" "$DEST/multi-fast/"
        # The job scripts once only: each account repoints their paths there (scripts/cluster/README.md).
        "${RSYNC[@]}" --ignore-existing --include='*_tillicum.slurm' --exclude='*' \
            "$REPO/multi-fast/scripts/cluster/" "$DEST/multi-fast/scripts/cluster/"
        ;;
    student)
        [ $# -eq 2 ] || usage
        "${RSYNC[@]}" --exclude=videos/ --exclude=pcd_videos/ --exclude='epoch_*.pt' \
            "$DEST/multi-fast/logs/distill/$2/" "$REPO/multi-fast/logs/distill/$2/"
        ;;
    *) usage ;;
esac
