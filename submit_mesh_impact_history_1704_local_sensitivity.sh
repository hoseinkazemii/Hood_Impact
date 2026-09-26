#!/usr/bin/env bash
# Neighborhood attention with ordinary acceleration MSE.
# Fresh cluster-B holdout: test 4,5; validation 11; training uses the other nine.
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

# Pin the neighborhood architecture and holdout even if older experiments remain exported.
export HOOD_MESH_DECODER="temporal"
export HOOD_MESH_NEIGHBORHOOD_LAYERS="2"
export HOOD_MESH_NEIGHBORHOOD_K="16"
export HOOD_MESH_NEIGHBORHOOD_SCALE_MM="20"
export HOOD_MESH_NEIGHBORHOOD_CHUNK_SIZE="1024"
# The 1704 .inp *NODE blocks list the rigid headform first. Preflight
# verifies the boundary against two runs before training starts.
export HOOD_MESH_IMPACTOR_NODES="286"
export HOOD_MESH_TEST_DESIGNS="4 5"
export HOOD_MESH_VAL_DESIGNS="11"
export HOOD_MESH_ALLOW_CLONE_LEAK="0"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterB_local_sensitivity}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Experiment: 2 neighborhood layers, k=16, headform (286 nodes) held out of the local graph\n'
printf 'Decoder: temporal | ordinary acceleration MSE\n'
printf 'Holdout: train 0,1,2,3,6,7,8,9,10 | validation 11 | test 4,5 (whole cluster B)\n'
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name=mesh_hist_clusterB_local \
    --time=48:00:00 \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_clusterB_local_sensitivity_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_clusterB_local_sensitivity_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_impact_history_1704.sbatch"
