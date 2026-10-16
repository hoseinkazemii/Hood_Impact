#!/usr/bin/env bash
# Location-shortcut removal, as an A/B against 20261004_050747_3303364_1704:
# the same k256 FiLM neighborhood network, Fourier time, all 142 locations,
# stride 16 and cluster-B holdout (test 4,5 | validation 11), with the two
# geometry-free impact paths closed:
#   --pooling-query-residual false  memory = attention-weighted hood-node values only
#   --pool-impactor-nodes false     the 286 headform rows get zero pooling weight
# Compare test_design_sensitivity.json and hic_metrics.json with that run.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_no_location_shortcuts.sh
#   WANDB_MODE=online bash submit_mesh_impact_history_1704_no_location_shortcuts.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_POOLING_QUERY_RESIDUAL="false"
export HOOD_MESH_POOL_IMPACTOR_NODES="false"
# Pin the comparison settings even if another experiment's values are exported.
export HOOD_MESH_TIME_SUBSAMPLE_STRIDE="16"
export HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="0"
export HOOD_MESH_MAX_TRAIN_TIME=""
export HOOD_MESH_EPOCHS="${HOOD_MESH_EPOCHS:-100}"
export HOOD_MESH_BATCH_SIZE="${HOOD_MESH_BATCH_SIZE:-8}"
export HOOD_MESH_LR="${HOOD_MESH_LR:-0.0003}"
export HOOD_MESH_WEIGHT_DECAY="${HOOD_MESH_WEIGHT_DECAY:-0.00001}"
export HOOD_MESH_SEED="${HOOD_MESH_SEED:-42}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_hic0pct_no_shortcuts}"
# The online W&B API timed out before data loading in job 3317059; record on
# disk by default and upload later with `wandb sync`.
export WANDB_MODE="${WANDB_MODE:-offline}"

printf 'Location shortcuts: query residual OFF | headform rows excluded from pooling\n'
printf 'Baseline for comparison: 20261004_050747_3303364_1704 (both shortcuts on)\n'
printf 'Training: %s epochs | batch %s | lr %s | weight decay %s | seed %s\n' \
    "${HOOD_MESH_EPOCHS}" "${HOOD_MESH_BATCH_SIZE}" "${HOOD_MESH_LR}" \
    "${HOOD_MESH_WEIGHT_DECAY}" "${HOOD_MESH_SEED}"
printf 'W&B tracking: %s (offline records can be uploaded later with wandb sync)\n' "${WANDB_MODE}"
mkdir -p "${PROJECT_DIR}/runs/slurm"
# The baseline needed 11.5 h on one GH200; 24 h leaves margin.
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_fourier_k256.sh" \
    --job-name=mesh_hist_no_shortcuts --time=24:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_no_location_shortcuts_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_no_location_shortcuts_%j.err" \
    "$@"
