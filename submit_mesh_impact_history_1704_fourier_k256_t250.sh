#!/usr/bin/env bash
# Time-resolution comparison with the all-location 20261004 Fourier k256 run.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_TIME_SUBSAMPLE_STRIDE="4"
export HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="0"
export HOOD_MESH_EPOCHS="${HOOD_MESH_EPOCHS:-100}"
export HOOD_MESH_BATCH_SIZE="${HOOD_MESH_BATCH_SIZE:-8}"
export HOOD_MESH_LR="${HOOD_MESH_LR:-0.0003}"
export HOOD_MESH_WEIGHT_DECAY="${HOOD_MESH_WEIGHT_DECAY:-0.00001}"
export HOOD_MESH_SEED="${HOOD_MESH_SEED:-42}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_hic0pct_t250}"
# Job 3317059 stopped before data loading when the online W&B API timed out.
# Keep tracking on disk so this comparison can train without service access.
export WANDB_MODE="${WANDB_MODE:-offline}"

printf 'Time-resolution comparison: stride 4 | 250 queries from 1000 source samples | all 142 locations\n'
printf 'W&B tracking: %s (offline records can be uploaded later with wandb sync)\n' "${WANDB_MODE}"
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_fourier_k256.sh" \
    --job-name=mesh_hist_fourier_k256_t250 --time=24:00:00 "$@"
