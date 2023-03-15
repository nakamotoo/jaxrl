#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True


# env=kitchen-partial-v0
# env=kitchen-mixed-v0
env=kitchen-complete-v0

# beta=1.0
beta=0.3
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
--max_steps=1500000 \
--num_pretraining_steps=500000 \
--eval_interval=50000 \
--eval_episodes=20 \
--logging.project=AWAC-kitchen-Mar15 \
--config.beta=$beta
done