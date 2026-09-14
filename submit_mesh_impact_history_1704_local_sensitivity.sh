#!/usr/bin/env bash
# Neighborhood attention + matched-location design-difference training.
# Same temporal decoder and cluster-D holdout as 20260912_005426_3134539_1704.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi

# Pin the two additions and holdout even if older experiments remain exported.
export HOOD_MESH_DECODER="temporal"
export HOOD_MESH_NEIGHBORHOOD_LAYERS="2"
export HOOD_MESH_NEIGHBORHOOD_K="16"
export HOOD_MESH_NEIGHBORHOOD_SCALE_MM="20"
export HOOD_MESH_NEIGHBORHOOD_CHUNK_SIZE="1024"
# The 1704 .inp *NODE blocks list the rigid headform first. Preflight
# verifies the boundary against two runs before training starts.
export HOOD_MESH_IMPACTOR_NODES="286"
export HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="1.0"
export HOOD_MESH_TEST_DESIGNS="10 11"
export HOOD_MESH_VAL_DESIGNS="5"
export HOOD_MESH_ALLOW_CLONE_LEAK="0"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterD_local_sensitivity}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Experiment: 2 neighborhood layers, k=16, headform (286 nodes) held out of the local graph\n'
printf 'Decoder: temporal | design-difference weight=1\n'
printf 'Holdout: train 0,1,2,3,4,6,7,8,9 | validation 5 | test 10,11\n'
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name=mesh_hist_local_sensitivity \
    --time=48:00:00 \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_local_sensitivity_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_local_sensitivity_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_impact_history_1704.sbatch"
