#!/usr/bin/env bash
# Submit from any directory. Additional arguments are passed to sbatch.
# Example: HOOD_MESH_EPOCHS=1 bash submit_mesh_impact_history_1704.sh --time=00:30:00
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this script on the cluster login node." >&2
    exit 127
fi

# Slurm opens these paths before the batch script itself starts.
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_impact_history_1704.sbatch"
