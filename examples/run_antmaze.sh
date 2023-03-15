#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=7
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2
env=antmaze-large-diverse-v2

# 4 5 6
# for seed in 1 3 5
for seed in 2 4 6
do
python train_finetuning.py \
--env_name=$env \
--dataset_name=d4rl \
--save_dir=/raid/mitsuhiko/logs \
--seed=$seed \
--log_interval=5000 \
--max_steps=1000000 \
--num_pretraining_steps=1000000 \
--eval_interval=50000 \
--eval_episodes=50 \
--logging.project=AWAC-antmaze-rerun-Mar15 \
--config.beta=1.0
done