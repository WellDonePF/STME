#!/bin/bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/cpfs/user/wangpengfei2/hf
export PATH=/cpfs/user/wangpengfei2/miniconda3/envs/llava/bin:$PATH

MODEL_NAME="llava-ov-7B"
GPUS_PER_NODE=`nvidia-smi -L | wc -l`
num_worker_per_GPU=3

cd /cpfs/user/wangpengfei2/project/LLaVA-NeXT

# for ((num_rank=0; num_rank<15; num_rank++))
# do
#     GPU_idx=$((num_rank / num_worker_per_GPU))
#     CUDA_VISIBLE_DEVICES=${GPU_idx} python3 attn_score_analysis/${MODEL_NAME}/analyasis_from_merge_data.py \
#         --num_worker 15 \
#         --rank ${num_rank} \
#         --delete_tensor \
#         --task_type medium &
#     sleep 0.5
# done

for ((num_rank=0; num_rank<24; num_rank++))
do
    GPU_idx=$((num_rank / num_worker_per_GPU + 5))
    CUDA_VISIBLE_DEVICES=${GPU_idx} python3 attn_score_analysis/${MODEL_NAME}/analyasis_from_merge_data.py \
        --num_worker 9 \
        --rank ${num_rank} \
        --delete_tensor \
        --task_type easy &
    sleep 0.5
done

wait