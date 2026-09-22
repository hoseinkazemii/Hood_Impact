#!/usr/bin/env bash
# Submit the unchanged V6 CNN architecture on the canonical 1704 dataset.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi
mkdir -p "${PROJECT_DIR}/runs/slurm"
printf 'Multiscale CNN | 1704 samples | train 0-9 | validation 10 | test 11 (defaults)\n'
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/gpu_multiscale_hic_1704_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/gpu_multiscale_hic_1704_%j.err" \
    "$@" "${PROJECT_DIR}/run_gpu_multiscale_hic_1704.sbatch"
