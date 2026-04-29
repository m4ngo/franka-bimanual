# Script for training a policy on a given dataset.
# $1 is repo id
# $2 is policy repo id
# $3 is batch size
# $4 is steps
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <repo_id> <policy_repo_id> <batch_size> <steps>"
    exit 1
fi
lerobot-train \  
  --dataset.repo_id=$1 \  
  --policy.type=act \  
  --output_dir=outputs/train/act_$1 \  
  --job_name=act_$1 \  
  --policy.device=cuda \  
  --wandb.enable=true \  
  --policy.repo_id=$2 \
  --batch_size=$3 \
  --steps=$4