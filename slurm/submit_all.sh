#!/bin/bash
# Submit all per-method arrays. MAML submitted last — smoke-test it interactively first:
#   python scripts/train.py method=maml model=hubert held_out_domain=0 epochs=1 wandb_offline=true
#
# REQUIRED pre-submit steps (run on login node):
#   1. Prefetch HF models:
#      HF_HOME=/home/ak562fx/.cache/huggingface python scripts/prefetch_hf.py
#   2. Verify data + HF cache:
#      HF_HOME=/home/ak562fx/.cache/huggingface python scripts/precheck_data.py --check-hf
#
# A single RUN_TS is computed once and exported to every sbatch. All 5 methods
# land in folders prefixed with the same timestamp, distinguished by SLURM job ID:
#   logs/2026-05-12_1430_job<JOBID>/
#   checkpoints/2026-05-12_1430_job<JOBID>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y-%m-%d_%H%M)}"
: "${HF_HOME:?must set HF_HOME before calling submit_all.sh}"
export RUN_TS

echo "RUN_TS=${RUN_TS}"
echo "HF_HOME=${HF_HOME}"

echo "Submitting ERM..."
sbatch --export=ALL,RUN_TS="${RUN_TS}",HF_HOME="${HF_HOME}" "${SCRIPT_DIR}/erm_array.sh"

echo "Submitting DIFL..."
sbatch --export=ALL,RUN_TS="${RUN_TS}",HF_HOME="${HF_HOME}" "${SCRIPT_DIR}/difl_array.sh"

echo "Submitting Mixup..."
sbatch --export=ALL,RUN_TS="${RUN_TS}",HF_HOME="${HF_HOME}" "${SCRIPT_DIR}/mixup_array.sh"

echo "Submitting CORAL..."
sbatch --export=ALL,RUN_TS="${RUN_TS}",HF_HOME="${HF_HOME}" "${SCRIPT_DIR}/coral_array.sh"

echo "Submitting MAML..."
sbatch --export=ALL,RUN_TS="${RUN_TS}",HF_HOME="${HF_HOME}" "${SCRIPT_DIR}/maml_array.sh"

echo "Done. Check queue: squeue -u \$(whoami)"
