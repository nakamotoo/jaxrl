#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2


env=pen-binary-v0

# 4 5 6
for seed in 1 
# for seed in 4 5 6

do
python train_finetuning.py \
--env_name=$env \
--dataset_name=adroit \
--save_dir=/raid/mitsuhiko/logs \
--seed=$seed \
--log_interval=5000 \
--max_steps=3000000 \
--num_pretraining_steps=25000 \
--eval_interval=5000 \
--eval_episodes=50 \
--logging.project=AWAC-adroit-test
done