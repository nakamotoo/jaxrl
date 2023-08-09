#!/bin/bash
#SBATCH --job-name=che
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TITAN:1
#SBATCH --account=co_rail
#SBATCH --partition=savio3_gpu
#SBATCH --qos=rail_gpu3_normal

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))

BETAS="halfcheetah-medium-replay-v2;halfcheetah-medium-v2;halfcheetah-random-v2"

arrBETAS=(${BETAS//;/ })

BETA=${arrBETAS[$TASK_ID]}

module load gnu-parallel

export PROJECT_DIR="/global/home/users/$USER/jaxrl"
export LOG_DIR="/global/scratch/users/$USER/jaxrl"
export PROJECT_NAME="0805-AWAC-locomotion-sweep"

run_singularity ()
{
singularity exec --userns --nv -B /usr/lib64 -B /var/lib/dcv-gl --overlay /global/scratch/users/nakamoto/singularity/50G.img:ro /global/scratch/users/nakamoto/singularity/cuda11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
    source ~/.bashrc
    cd $PROJECT_DIR/examples
    export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

    XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --env_name=$1 \
    --dataset_name=d4rl \
    --save_dir=/raid/mitsuhiko/logs \
    --seed=$2 \
    --log_interval=10000 \
    --max_steps=1000000 \
    --num_pretraining_steps=500000 \
    --eval_interval=50000 \
    --eval_episodes=10 \
    --logging.project=$PROJECT_NAME \
    --config.beta=1.0 \
    --config.replay_buffer_size=3e6
"
}

export -f run_singularity
parallel --delay 20 --linebuffer -j 3 run_singularity $BETA {} \
    ::: 1 2 3