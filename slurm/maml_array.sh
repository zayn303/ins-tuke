#!/bin/bash
#SBATCH --job-name=pd-dg-maml
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-8%3

source /home/ak562fx/ins-tuke/venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(wav2vec2 hubert wavlm)
HELD_OUTS=(0 1 2)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((IDX % 3))
HELD_IDX=$((IDX / 3))

MODEL=${MODELS[$MODEL_IDX]}
HELD_OUT=${HELD_OUTS[$HELD_IDX]}

mkdir -p logs
exec >"logs/maml_${MODEL}_held${HELD_OUT}_${IDX}.out" 2>"logs/maml_${MODEL}_held${HELD_OUT}_${IDX}.err"

echo "Running: method=maml model=${MODEL} held_out_domain=${HELD_OUT}"

python scripts/train.py \
    method=maml \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
    wandb_offline=true
