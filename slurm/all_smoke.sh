#!/bin/bash
#SBATCH --job-name=pd-dg-all-smoke
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-47%20

# Smoke all 5 methods x 3 models x 4 held_outs = 60 variants, 1 epoch each.
# MAML: epochs=1 + n_episodes_per_epoch=2 (default 100 defeats smoke purpose).
# Submit with:
#   sbatch --export=ALL,RUN_TS=$(date +%Y-%m-%d_%H%M),HF_HOME=/home/ak562fx/.cache/huggingface slurm/all_smoke.sh
: "${RUN_TS:?must export RUN_TS via sbatch --export=ALL,RUN_TS=\$(date +%Y-%m-%d_%H%M)}"
: "${HF_HOME:?must export HF_HOME on shared FS}"

source /home/ak562fx/ins-tuke/venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHODS=(difl mixup coral maml)
MODELS=(wav2vec2 hubert wavlm)
HELD_OUTS=(0 1 2 3)
NUM_METHOD=${#METHODS[@]}
NUM_MODEL=${#MODELS[@]}
NUM_HELD=${#HELD_OUTS[@]}
INNER=$((NUM_MODEL * NUM_HELD))   # 12
EXPECTED=$((NUM_METHOD * INNER - 1))  # 59

IDX=${SLURM_ARRAY_TASK_ID}
if [[ "$IDX" -lt 0 || "$IDX" -gt "$EXPECTED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of [0,$EXPECTED]" >&2
    exit 2
fi

METHOD_IDX=$((IDX / INNER))
INNER_IDX=$((IDX % INNER))
MODEL_IDX=$((INNER_IDX % NUM_MODEL))
HELD_IDX=$((INNER_IDX / NUM_MODEL))

METHOD=${METHODS[$METHOD_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
HELD_OUT=${HELD_OUTS[$HELD_IDX]}

if [[ -z "$METHOD" || -z "$MODEL" || -z "$HELD_OUT" && "$HELD_OUT" != "0" ]]; then
    echo "ERROR: empty METHOD=$METHOD MODEL=$MODEL HELD_OUT=$HELD_OUT at IDX=$IDX" >&2
    exit 2
fi

RUN_DIR="${RUN_TS}_job${SLURM_ARRAY_JOB_ID}_all_smoke"
LOG_DIR="logs/${RUN_DIR}"
CKPT_DIR="checkpoints/${RUN_DIR}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

exec >"${LOG_DIR}/${METHOD}_${MODEL}_held${HELD_OUT}_${IDX}.out" \
    2>"${LOG_DIR}/${METHOD}_${MODEL}_held${HELD_OUT}_${IDX}.err"

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

echo "Running: method=${METHOD} model=${MODEL} held_out_domain=${HELD_OUT} epochs=1 [ALL-SMOKE]"

if [[ "$METHOD" == "maml" ]]; then
    python scripts/train.py \
        method=${METHOD} \
        model=${MODEL} \
        held_out_domain=${HELD_OUT} \
        epochs=1 \
        n_episodes_per_epoch=2 \
        data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
        checkpoint_dir=${CKPT_DIR} \
        wandb_offline=true
else
    python scripts/train.py \
        method=${METHOD} \
        model=${MODEL} \
        held_out_domain=${HELD_OUT} \
        epochs=1 \
        data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
        checkpoint_dir=${CKPT_DIR} \
        wandb_offline=true
fi
