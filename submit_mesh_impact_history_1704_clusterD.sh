#!/usr/bin/env bash
# Cluster-D holdout run of MeshImpactHistoryNet on the 1704 EuroNCAP set.
# Historical reproduction; fresh B-holdout runs use the clusterB or local_sensitivity launcher.
#
#   train      designs 0,1,2,3,4,6,7,8,9   (geometry clusters A, B, C)
#   validation design 5                    (cluster B)
#   test       designs 10,11               (whole cluster D)
#   decoder    temporal (global/local mesh attention + temporal self-attention)
#
# Match the hardest existing run, 20260912_005426_3134539_1704: checkpoint
# selection never sees cluster D. Preflight rejects test clones in training
# and warns that validation design 5 shares cluster B with training design 4.
#
# Temporal attention is the default after the ablation performed worse.
# To reproduce the ablation use submit_mesh_impact_history_1704_no_temporal.sh.
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

export HOOD_MESH_TEST_DESIGNS="${HOOD_MESH_TEST_DESIGNS:-10 11}"
export HOOD_MESH_VAL_DESIGNS="${HOOD_MESH_VAL_DESIGNS:-5}"
export HOOD_MESH_DECODER="${HOOD_MESH_DECODER:-temporal}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterD_${HOOD_MESH_DECODER}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Decoder arm: %s\n' "${HOOD_MESH_DECODER}"
printf 'Holdout: validation %s | test %s | training uses remaining designs\n' \
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
