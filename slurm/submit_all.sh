#!/bin/bash
# Submit all per-method arrays. MAML submitted last — smoke-test it interactively first:
#   python scripts/train.py method=maml model=hubert held_out_domain=0 epochs=1 wandb_offline=true
#
# A single RUN_TS is computed once and exported to every sbatch. All 5 methods
# land in folders prefixed with the same timestamp, distinguished by SLURM job ID:
#   logs/2026-05-12_1430_job<JOBID>/
#   checkpoints/2026-05-12_1430_job<JOBID>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y-%m-%d_%H%M)}"
export RUN_TS

echo "RUN_TS=${RUN_TS}"

echo "Submitting ERM..."
sbatch --export=ALL,RUN_TS="${RUN_TS}" "${SCRIPT_DIR}/erm_array.sh"

echo "Submitting DIFL..."
sbatch --export=ALL,RUN_TS="${RUN_TS}" "${SCRIPT_DIR}/difl_array.sh"

echo "Submitting Mixup..."
sbatch --export=ALL,RUN_TS="${RUN_TS}" "${SCRIPT_DIR}/mixup_array.sh"

echo "Submitting CORAL..."
sbatch --export=ALL,RUN_TS="${RUN_TS}" "${SCRIPT_DIR}/coral_array.sh"

echo "Submitting MAML..."
sbatch --export=ALL,RUN_TS="${RUN_TS}" "${SCRIPT_DIR}/maml_array.sh"

echo "Done. Check queue: squeue -u \$(whoami)"
