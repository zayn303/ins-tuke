#!/bin/bash
#SBATCH --job-name=pd-dg-mixup-smoke
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-14%15
#SBATCH --output=/dev/null

# Smoke test: 1 epoch per task, 15 variants (3 models x 5 held-out domains). Run before full mixup_array.sh.
# RUN_TS and HF_HOME auto-generated if not exported. To override:
#   sbatch --export=ALL,RUN_TS=$(date +%Y-%m-%d_%H%M),HF_HOME=/home/ak562fx/.cache/huggingface slurm/mixup_smoke.sh
RUN_TS="${RUN_TS:-$(date +%Y-%m-%d_%H%M)}"
HF_HOME="${HF_HOME:-/home/ak562fx/.cache/huggingface}"
export HF_HOME

source /home/ak562fx/ins-tuke/venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(wav2vec2 hubert wavlm)
HELD_OUTS=(0 1 2 3 4)
NUM_MODEL=${#MODELS[@]}
NUM_HELD=${#HELD_OUTS[@]}
EXPECTED=$((NUM_HELD * NUM_MODEL - 1))

IDX=${SLURM_ARRAY_TASK_ID}
if [[ "$IDX" -lt 0 || "$IDX" -gt "$EXPECTED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of [0,$EXPECTED]" >&2
    exit 2
fi

MODEL_IDX=$((IDX % NUM_MODEL))
HELD_IDX=$((IDX / NUM_MODEL))
MODEL=${MODELS[$MODEL_IDX]}
HELD_OUT=${HELD_OUTS[$HELD_IDX]}

if [[ -z "$HELD_OUT" || -z "$MODEL" ]]; then
    echo "ERROR: empty MODEL=$MODEL or HELD_OUT=$HELD_OUT at IDX=$IDX" >&2
    exit 2
fi

RUN_DIR="${RUN_TS}_job${SLURM_ARRAY_JOB_ID}_smoke"
LOG_DIR="logs/${RUN_DIR}"
CKPT_DIR="checkpoints/${RUN_DIR}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

exec >"${LOG_DIR}/mixup_${MODEL}_held${HELD_OUT}_${IDX}.out" 2>"${LOG_DIR}/mixup_${MODEL}_held${HELD_OUT}_${IDX}.err"

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

echo "Running: method=mixup model=${MODEL} held_out_domain=${HELD_OUT} epochs=1 [SMOKE]"

python scripts/train.py \
    method=mixup \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    epochs=1 \
    data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
    checkpoint_dir=${CKPT_DIR} \
    wandb_offline=true
