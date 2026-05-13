#!/bin/bash
#SBATCH --job-name=pd-dg-coral-smoke
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-11%12

# Smoke test: 1 epoch per task, all 12 variants. Run before full coral_array.sh.
# Submit with:
#   sbatch --export=ALL,RUN_TS=$(date +%Y-%m-%d_%H%M),HF_HOME=/home/ak562fx/.cache/huggingface slurm/coral_smoke.sh
: "${RUN_TS:?must export RUN_TS via sbatch --export=ALL,RUN_TS=\$(date +%Y-%m-%d_%H%M)}"
: "${HF_HOME:?must export HF_HOME on shared FS}"

source /home/ak562fx/ins-tuke/venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(wav2vec2 hubert wavlm)
HELD_OUTS=(0 1 2 3)
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

exec >"${LOG_DIR}/coral_${MODEL}_held${HELD_OUT}_${IDX}.out" 2>"${LOG_DIR}/coral_${MODEL}_held${HELD_OUT}_${IDX}.err"

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

echo "Running: method=coral model=${MODEL} held_out_domain=${HELD_OUT} epochs=1 [SMOKE]"

python scripts/train.py \
    method=coral \
    model=${MODEL} \
    held_out_domain=${HELD_OUT} \
    epochs=1 \
    data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
    checkpoint_dir=${CKPT_DIR} \
    wandb_offline=true
