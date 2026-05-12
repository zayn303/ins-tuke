#!/bin/bash
#SBATCH --job-name=pd-dg-coral
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-8

source /home/ak562fx/ins-tuke/venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error

MODELS=(wav2vec2 hubert wavlm)
HELD_OUTS=(0 1 2)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((IDX % 3))
HELD_IDX=$((IDX / 3))

MODEL=${MODELS[$MODEL_IDX]}
HELD_OUT=${HELD_OUTS[$HELD_IDX]}

echo "Running: method=coral model=${MODEL} held_out_domain=${HELD_OUT}"

python scripts/train.py \
    method=coral \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
    wandb_offline=true
