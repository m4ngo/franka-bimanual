# $1 is repo id
# $2 is episode number
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <episode_number>"
    exit 1
fi
lerobot-replay \
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
    --dataset.episode=$2