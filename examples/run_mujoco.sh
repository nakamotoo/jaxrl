#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# halfcheetah-expert-v2, halfcheetah-medium-expert-v2, halfcheetah-medium-replay-v2, halfcheetah-medium-v2, halfcheetah-random-v2

env=halfcheetah-random-v2
# 4 5 6
# for seed in 1 3 5
for seed in 1 2 3
do
python train_finetuning.py \
--env_name=$env \
--dataset_name=d4rl \
--save_dir=/raid/mitsuhiko/logs \
--seed=$seed \
--log_interval=10000 \
--max_steps=1000000 \
--num_pretraining_steps=500000 \
--eval_interval=50000 \
--eval_episodes=10 \
--logging.project=0809-AWAC-locomotion \
--config.beta=1.0
done