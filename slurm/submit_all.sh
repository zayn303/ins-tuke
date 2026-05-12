#!/bin/bash
# Submit all per-method arrays. MAML submitted last — smoke-test it interactively first:
#   python scripts/train.py method=maml model=hubert held_out_domain=0 epochs=1 wandb_offline=true

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting ERM..."
sbatch "${SCRIPT_DIR}/erm_array.sh"

echo "Submitting DIFL..."
sbatch "${SCRIPT_DIR}/difl_array.sh"

echo "Submitting Mixup..."
sbatch "${SCRIPT_DIR}/mixup_array.sh"

echo "Submitting CORAL..."
sbatch "${SCRIPT_DIR}/coral_array.sh"

echo "Submitting MAML..."
sbatch "${SCRIPT_DIR}/maml_array.sh"

echo "Done. Check queue: squeue -u \$(whoami)"
