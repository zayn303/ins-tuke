#!/bin/bash
#SBATCH --job-name=pd-dg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# TODO: fill in your cluster's module names
# module load cuda/12.1
# module load anaconda3/2023.09

conda activate pd-dg

python scripts/train.py \
    method=${METHOD} \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    data_root=${DATA_ROOT} \
    wandb_offline=true
