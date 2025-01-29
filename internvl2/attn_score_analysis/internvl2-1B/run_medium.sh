#!/bin/bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/cpfs/user/wangpengfei2/hf
export PATH=/cpfs/user/wangpengfei2/miniconda3/envs/internvl/bin:$PATH

MODEL_NAME="internvl2-1B"
GPUS_PER_NODE=`nvidia-smi -L | wc -l`
MASTER_PORT=12345

cd /cpfs/user/wangpengfei2/project/InternVL

torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR}  \
    --master_port ${MASTER_PORT}  \
attn_score_analysis/${MODEL_NAME}/get_attn_score_from_merge_data.py \
    --task_type medium \
    --attn_mode 3

