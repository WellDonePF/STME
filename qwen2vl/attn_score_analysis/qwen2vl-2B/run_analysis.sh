#!/bin/bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/cpfs/user/wangpengfei2/hf
export PATH=/cpfs/user/wangpengfei2/miniconda3/envs/qwen2vl/bin:$PATH
MODEL_NAME="qwen2vl-2B"
num_all_worker=10

cd /cpfs/user/wangpengfei2/project/Qwen2VL

for ((num_rank=0; num_rank<num_all_worker; num_rank++))
do
    python3 attn_score_analysis/${MODEL_NAME}/analyasis_from_merge_data.py \
        --num_worker ${num_all_worker} \
        --rank ${num_rank} \
        --delete_tensor \
        --task_type easy &
    sleep 1
done

for ((num_rank=0; num_rank<num_all_worker; num_rank++))
do
    # device_num=$((num_rank / 3 + 4))
    python3 attn_score_analysis/${MODEL_NAME}/analyasis_from_merge_data.py \
        --num_worker ${num_all_worker} \
        --rank ${num_rank} \
        --delete_tensor \
        --task_type medium &
    sleep 1
done

wait