#!/usr/bin/env bash
# January's best acceleration architecture, trained fresh on cluster B holdout.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi
mkdir -p "${PROJECT_DIR}/runs/slurm"
printf 'Temporal DeepONet acceleration baseline: single FiLM, January best architecture\n'
printf 'Fresh training | train 0,1,2,3,6,7,8,9,10 | validation 11 | test 4,5\n'
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/temporal_deeponet_1704_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/temporal_deeponet_1704_%j.err" \
    "$@" "${PROJECT_DIR}/run_temporal_deeponet_1704.sbatch"
