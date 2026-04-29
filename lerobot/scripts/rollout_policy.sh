# Script for rolling out a trained policy.
# $1 is repo id
# $2 is number of episodes
# $3 is policy repo id
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <number_of_episodes>"
    exit 1
fi
lerobot-record \
    --robot.type=bimanual_franka \
    --robot.l_server_ip=192.168.3.11 \
    --robot.l_robot_ip=192.168.200.2 \
    --robot.l_gripper_ip=192.168.2.21 \
    --robot.l_port=18813 \
    --robot.r_server_ip=192.168.3.10 \
    --robot.r_robot_ip=192.168.201.10 \
    --robot.r_gripper_ip=192.168.2.20 \
    --robot.r_port=18812 \
    --robot.use_ee_delta=false \
    --dataset.repo_id=$1 \
    --dataset.num_episodes=$2 \
    --dataset.single_task="Evaluating policy $3 on dataset $1" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --dataset.fps=20 \
    --display_data=true \
    --policy.path=$3
