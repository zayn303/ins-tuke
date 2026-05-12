#!/bin/bash
#SBATCH --job-name=pd-dg-array
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-44

# TODO: fill in your cluster's module names
# module load cuda/12.1
# module load anaconda3/2023.09

conda activate pd-dg

# 45 combinations: 5 methods x 3 models x 3 held_out_domains
METHODS=(erm difl mixup maml coral)
MODELS=(wav2vec2 hubert wavlm)
HELD_OUTS=(0 1 2)

N_METHODS=${#METHODS[@]}
N_MODELS=${#MODELS[@]}
N_HELD=${#HELD_OUTS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
HELD_IDX=$((IDX % N_HELD))
MODEL_IDX=$(((IDX / N_HELD) % N_MODELS))
METHOD_IDX=$((IDX / (N_HELD * N_MODELS)))

METHOD=${METHODS[$METHOD_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
HELD_OUT=${HELD_OUTS[$HELD_IDX]}

echo "Running: method=${METHOD} model=${MODEL} held_out_domain=${HELD_OUT}"

python scripts/train.py \
    method=${METHOD} \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    data_root=${DATA_ROOT:-Data} \
    wandb_offline=true
