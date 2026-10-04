#!/usr/bin/env bash
# Fresh cluster-B holdout. Training: 0,1,2,3,6,7,8,9,10; validation: 11; test: 4,5.
# Model/loss options remain configurable for comparisons on this fixed split.
# For the combined neighborhood experiment use the local_sensitivity launcher.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if [[ -n "${HOOD_MESH_RESUME_FROM:-}" ]]; then
    echo "ERROR: This launcher starts a fresh cluster-B experiment. Unset HOOD_MESH_RESUME_FROM, or use the resume launcher to keep a checkpoint's saved split." >&2
    exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi

# Pin the new holdout even if a previous cluster-D experiment is still exported.
export HOOD_MESH_TEST_DESIGNS="4 5"
export HOOD_MESH_VAL_DESIGNS="11"
export HOOD_MESH_ALLOW_CLONE_LEAK="0"
export HOOD_MESH_DECODER="${HOOD_MESH_DECODER:-temporal}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterB_${HOOD_MESH_DECODER}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Decoder arm: %s\n' "${HOOD_MESH_DECODER}"
printf 'Holdout: train 0,1,2,3,6,7,8,9,10 | validation 11 | test 4,5 (whole cluster B)\n'
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name=mesh_hist_clusterB \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_clusterB_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_clusterB_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_impact_history_1704.sbatch"
