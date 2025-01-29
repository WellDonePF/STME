#!/bin/bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/cpfs/user/wangpengfei2/hf
export PATH=/cpfs/user/wangpengfei2/miniconda3/envs/internvl/bin:$PATH

MODEL_NAME="internvl2-1B"
num_worker_per_node=$1
MACHINE_RANK=$2
num_worker_all=$3

cd /cpfs/user/wangpengfei2/project/InternVL

for ((num_rank=0; num_rank<num_worker_per_node; num_rank++))
do
    new_num_rank=$((MACHINE_RANK * num_worker_per_node + num_rank))
    python3 attn_score_analysis/${MODEL_NAME}/analyasis_from_merge_data.py \
        --num_worker ${num_worker_all} \
        --rank ${new_num_rank} \
        --delete_tensor \
        --task_type medium &
    sleep 1
done

wait