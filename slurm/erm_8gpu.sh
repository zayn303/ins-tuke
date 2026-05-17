#!/bin/bash
#SBATCH --job-name=pd-dg-erm
#SBATCH --partition=dgx
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# Submit with:
#   sbatch --export=ALL,RUN_TS=$(date +%Y-%m-%d_%H%M),HF_HOME=/home/ak562fx/.cache/huggingface slurm/erm_8gpu.sh
: "${RUN_TS:?must export RUN_TS via sbatch --export=ALL,RUN_TS=\$(date +%Y-%m-%d_%H%M)}"
: "${HF_HOME:?must export HF_HOME on shared FS}"

source /home/ak562fx/ins-tuke/venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${RUN_TS}_job${SLURM_JOB_ID}"
LOG_DIR="logs/${RUN_DIR}"
CKPT_DIR="checkpoints/${RUN_DIR}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

echo "=== SLURM env: job=${SLURM_JOB_ID} node=$(hostname) ==="
echo "RUN_DIR=${RUN_DIR}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi unavailable"

run_one() {
    local GPU_ID=$1 MODEL=$2 HELD=$3 IDX=$4
    echo "[start] GPU${GPU_ID}: erm/${MODEL}/held${HELD} (idx=${IDX})"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train.py \
        method=erm \
        model=${MODEL} \
        held_out_domain=${HELD} \
        epochs=40 \
        unfreeze_top_n_layers=4 \
        lr_schedule=cosine \
        data_root=${DATA_ROOT:-/home/ak562fx/ins-tuke/Data} \
        checkpoint_dir=${CKPT_DIR} \
        wandb_offline=true \
        num_workers=4 \
        >"${LOG_DIR}/erm_${MODEL}_held${HELD}_${IDX}.out" \
        2>"${LOG_DIR}/erm_${MODEL}_held${HELD}_${IDX}.err"
    local ec=$?
    echo "[done]  GPU${GPU_ID}: erm/${MODEL}/held${HELD} (idx=${IDX}) exit=${ec}"
    return $ec
}

# 15 jobs total: 5 folds × 3 models
# Smoke run 2026-05-17_1934_job6193 epoch timing (40 epochs):
#   WavLM   600–738 s/ep × 40 = ~8.2h  ← Round 1 bottleneck
#   wav2vec2 200–330 s/ep × 40 = ~3.7h
#   HuBERT  140–225 s/ep × 40 = ~2.5h
# Round 1 (8 jobs): 5×WavLM + 3×wav2vec2 → wall ~8.2h
# Round 2 (7 jobs): 2×wav2vec2 + 5×HuBERT → wall ~2.7h
# Total ≈ 11h; 24h limit gives 2× buffer

JOBS=(
    # Round 1 — indices 0-7 on GPUs 0-7
    "wavlm   0"
    "wavlm   1"
    "wavlm   2"
    "wavlm   3"
    "wavlm   4"
    "wav2vec2 0"
    "wav2vec2 1"
    "wav2vec2 2"
    # Round 2 — indices 8-14 on GPUs 0-6
    "wav2vec2 3"
    "wav2vec2 4"
    "hubert   0"
    "hubert   1"
    "hubert   2"
    "hubert   3"
    "hubert   4"
)

echo "=== Round 1: 8 jobs in parallel ==="
r1_pids=()
for IDX in $(seq 0 7); do
    read -r MODEL HELD <<< "${JOBS[$IDX]}"
    run_one "$IDX" "$MODEL" "$HELD" "$IDX" &
    r1_pids+=($!)
done

r1_fail=0
for pid in "${r1_pids[@]}"; do
    wait "$pid"
    ec=$?
    ((ec != 0)) && ((r1_fail++)) || true
done
echo "=== Round 1 done: ${r1_fail} failures ==="

echo "=== Round 2: 7 jobs in parallel ==="
r2_pids=()
for IDX in $(seq 8 14); do
    GPU_ID=$((IDX - 8))
    read -r MODEL HELD <<< "${JOBS[$IDX]}"
    run_one "$GPU_ID" "$MODEL" "$HELD" "$IDX" &
    r2_pids+=($!)
done

r2_fail=0
for pid in "${r2_pids[@]}"; do
    wait "$pid"
    ec=$?
    ((ec != 0)) && ((r2_fail++)) || true
done
echo "=== Round 2 done: ${r2_fail} failures ==="

total_fail=$((r1_fail + r2_fail))
echo "=== All 15 ERM jobs complete ==="
echo "=== Summary: ${total_fail}/15 failed ==="
echo "  Round 1: ${r1_fail} failures"
echo "  Round 2: ${r2_fail} failures"
if ((total_fail > 0)); then
    echo "FAIL: ${total_fail} jobs failed — check .err logs in ${LOG_DIR}/"
    exit 1
fi
echo "PASS: all 15 jobs exit 0"
