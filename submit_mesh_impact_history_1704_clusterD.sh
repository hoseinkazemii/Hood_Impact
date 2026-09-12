#!/usr/bin/env bash
# Cluster-D holdout run of MeshImpactHistoryNet on the 1704 EuroNCAP set.
#
#   train      designs 0-9   (geometry clusters A, B, C)
#   validation design 10     (cluster D)
#   test       design 11     (cluster D)
#
# Designs 10 and 11 are the only members of their geometry cluster, so training
# keeps no near-identical copy of either. That is the point of this run: under
# the previous single-design holdout the test design's clones stayed in
# training, and the reported score largely measured duplicate retrieval. The
# preflight now refuses any split that leaves clones behind.
#
# Stricter variant -- validate outside cluster D so checkpoint selection never
# sees a clone of the test geometry, at the cost of a noisier validation signal:
#   HOOD_MESH_VAL_DESIGNS="5" HOOD_MESH_TEST_DESIGNS="10 11" \
#       bash submit_mesh_impact_history_1704_clusterD.sh
# The preflight warns that design 4 stays in training beside validation design
# 5, and proceeds: only a clone of a *test* design invalidates the score.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_clusterD.sh
#   bash submit_mesh_impact_history_1704_clusterD.sh --time=08:00:00
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi

export HOOD_MESH_TEST_DESIGNS="${HOOD_MESH_TEST_DESIGNS:-11}"
export HOOD_MESH_VAL_DESIGNS="${HOOD_MESH_VAL_DESIGNS:-10}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterD}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Holdout: train designs 0-9 | validation %s | test %s\n' \
    "${HOOD_MESH_VAL_DESIGNS}" "${HOOD_MESH_TEST_DESIGNS}"

# Slurm opens these paths before the batch script itself starts.
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name=mesh_hist_clusterD \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_clusterD_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_clusterD_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_impact_history_1704.sbatch"
