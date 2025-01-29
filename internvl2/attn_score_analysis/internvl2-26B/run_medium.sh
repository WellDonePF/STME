#!/bin/bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/cpfs/user/wangpengfei2/hf
export PATH=/cpfs/user/wangpengfei2/miniconda3/envs/internvl/bin:$PATH

MODEL_NAME="internvl2-26B"
GPUS_PER_NODE=`nvidia-smi -L | wc -l`
MASTER_PORT=12345

cd /cpfs/user/wangpengfei2/project/InternVL

# torchrun \
#     --nproc_per_node ${GPUS_PER_NODE} \
#     --nnodes ${WORLD_SIZE} \
#     --node_rank ${RANK} \
#     --master_addr ${MASTER_ADDR}  \
#     --master_port ${MASTER_PORT}  \
# attn_score_analysis/${MODEL_NAME}/get_attn_score_from_merge_data.py \
#     --task_type medium \
#     --attn_mode 3

for ((num_rank=0; num_rank<GPUS_PER_NODE; num_rank++))
do
    process_index=$((RANK * GPUS_PER_NODE + num_rank))
    CUDA_VISIBLE_DEVICES=${num_rank} python3 attn_score_analysis/${MODEL_NAME}/get_attn_score_from_merge_data.py \
        --num_processes $((WORLD_SIZE * GPUS_PER_NODE)) \
        --local_process_index ${num_rank} \
        --process_index ${process_index} \
        --task_type medium \
        --attn_mode 3 &
    sleep 1
done

wait