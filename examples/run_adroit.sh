#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=5
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2

# dataset_name=adroit
dataset_name=adroit-trunc

# max_steps=300000
# env=pen-binary-v0

max_steps=1250000
# env=door-binary-v0
env=relocate-binary-v0

# 4 5 6
# for seed in 1 2 3
# for seed in 4 5 6

# for seed in 1 4
# for seed in 2 5
for seed in 3 6
do
python train_finetuning.py \
--env_name=$env \
--dataset_name=$dataset_name \
--save_dir=/raid/mitsuhiko/logs \
--seed=$seed \
--log_interval=5000 \
--max_steps=$max_steps \
--num_pretraining_steps=25000 \
--eval_interval=5000 \
--eval_episodes=50 \
--logging.project=AWAC-adroit-rerun-Mar15
done