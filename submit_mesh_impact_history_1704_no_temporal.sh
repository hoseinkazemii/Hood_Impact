#!/usr/bin/env bash
# Temporal-attention ablation of MeshImpactHistoryNet on the 1704 EuroNCAP set.
#
#   architecture  decoder="mesh_only" -- the decoder keeps its cross-attention
#                 into the mesh memory, but the self-attention ACROSS TIME is
#                 gone. Each output uses geometry, impact and its own time.
#                 The impact-conditioned global+local mesh attention and the
#                 latent mixing blocks are untouched.
#   train         designs 0,1,2,3,4,6,7,8,9  (1278 impacts)
#   validation    design 5                    (142 impacts, cluster B)
#   test          designs 10,11               (284 impacts, whole cluster D)
#
# Match the hardest temporal baseline: 20260912_005426_3134539_1704.
# Cluster D is absent from training and validation. Design 4 remains in training
# beside validation design 5, exactly as in that baseline; preflight reports it.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_no_temporal.sh
#   bash submit_mesh_impact_history_1704_no_temporal.sh --time=08:00:00
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi

# Pin this experiment even when older decoder/split settings are exported in
# the login shell. Use the generic launcher for other splits or the baseline.
export HOOD_MESH_DECODER="mesh_only"
export HOOD_MESH_NEIGHBORHOOD_LAYERS="0"
export HOOD_MESH_IMPACTOR_NODES="0"
export HOOD_MESH_DESIGN_DIFFERENCE_WEIGHT="0"
export HOOD_MESH_TEST_DESIGNS="10 11"
export HOOD_MESH_VAL_DESIGNS="5"
export HOOD_MESH_ALLOW_CLONE_LEAK="0"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterD_no_temporal}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Decoder arm: %s (temporal self-attention removed)\n' "${HOOD_MESH_DECODER}"
printf 'Holdout: train designs 0,1,2,3,4,6,7,8,9 | validation %s | test %s\n' \
    "${HOOD_MESH_VAL_DESIGNS}" "${HOOD_MESH_TEST_DESIGNS}"
printf 'Default architecture: 1131281 parameters (temporal baseline: 1263889).\n'

# Slurm opens these paths before the batch script itself starts.
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name=mesh_hist_no_temporal \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_no_temporal_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_no_temporal_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_impact_history_1704.sbatch"
