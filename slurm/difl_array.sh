#!/bin/bash
#SBATCH --job-name=pd-dg-difl
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%9

# Submit with:
#   sbatch --export=ALL,RUN_TS=$(date +%Y-%m-%d_%H%M) slurm/difl_array.sh
: "${RUN_TS:?must export RUN_TS via sbatch --export=ALL,RUN_TS=\$(date +%Y-%m-%d_%H%M)}"

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

RUN_DIR="${RUN_TS}_job${SLURM_ARRAY_JOB_ID}"
LOG_DIR="logs/${RUN_DIR}"
CKPT_DIR="checkpoints/${RUN_DIR}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

exec >"${LOG_DIR}/difl_${MODEL}_held${HELD_OUT}_${IDX}.out" 2>"${LOG_DIR}/difl_${MODEL}_held${HELD_OUT}_${IDX}.err"

echo "=== SLURM env: job=${SLURM_JOB_ID} array_job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=$(hostname) ==="
echo "RUN_DIR=${RUN_DIR}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "--- nvidia-smi -L ---"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L unavailable"
echo "--- nvidia-smi query ---"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi query unavailable"
GPU_UUID_VAL=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)
echo "GPU_UUID: ${GPU_UUID_VAL}"

echo "Running: method=difl model=${MODEL} held_out_domain=${HELD_OUT}"

python scripts/train.py \
    method=difl \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
    checkpoint_dir=${CKPT_DIR} \
    wandb_offline=true
