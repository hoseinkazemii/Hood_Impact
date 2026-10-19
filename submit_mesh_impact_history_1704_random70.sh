#!/usr/bin/env bash
# Per-design random location subset: every design keeps its own random 70% of
# the 142 impact locations (99), drawn independently per design with seed 42
# (location_splits/euroncap1704_random70_seed42.json). The design split is the
# usual cluster-B holdout, so designs stay separate while locations may repeat
# across train, validation and test:
#   train designs 0-3,6-10: 891 runs | validation design 11: 99 | test designs 4,5: 198
# Designs 4 and 5 share 72 locations, which the matched-location design
# sensitivity uses. Otherwise the no-location-shortcut settings (k256 FiLM
# neighborhood, Fourier time, stride 16, query residual off, headform out of
# pooling, no geometry FiLM).
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_random70.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_LOCATION_HOLDOUT="${PROJECT_DIR}/location_splits/euroncap1704_random70_seed42.json"
if [[ ! -r "${HOOD_MESH_LOCATION_HOLDOUT}" ]]; then
    echo "ERROR: Location split not found: ${HOOD_MESH_LOCATION_HOLDOUT}" >&2
    exit 2
fi
export HOOD_MESH_POOLING_QUERY_RESIDUAL="false"
export HOOD_MESH_POOL_IMPACTOR_NODES="false"
# Pin the comparison settings even if another experiment's values are exported.
export HOOD_MESH_GEOMETRY_FILM="none"
export HOOD_MESH_GEOMETRY_FOURIER_OCTAVES=""
export HOOD_MESH_TIME_SUBSAMPLE_STRIDE="16"
export HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="0"
export HOOD_MESH_MAX_TRAIN_TIME=""
export HOOD_MESH_EPOCHS="${HOOD_MESH_EPOCHS:-100}"
export HOOD_MESH_BATCH_SIZE="${HOOD_MESH_BATCH_SIZE:-8}"
export HOOD_MESH_LR="${HOOD_MESH_LR:-0.0003}"
export HOOD_MESH_WEIGHT_DECAY="${HOOD_MESH_WEIGHT_DECAY:-0.00001}"
export HOOD_MESH_SEED="${HOOD_MESH_SEED:-42}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_no_shortcuts_random70}"
export WANDB_MODE="${WANDB_MODE:-online}"

printf 'Location split: %s\n' "${HOOD_MESH_LOCATION_HOLDOUT}"
printf '  random 99/142 locations per design | train 0-3,6-10 | validation 11 | test 4,5\n'
printf 'Location shortcuts: query residual OFF | headform rows excluded from pooling | geometry FiLM off\n'
printf 'Training: %s epochs | batch %s | lr %s | weight decay %s | seed %s\n' \
    "${HOOD_MESH_EPOCHS}" "${HOOD_MESH_BATCH_SIZE}" "${HOOD_MESH_LR}" \
    "${HOOD_MESH_WEIGHT_DECAY}" "${HOOD_MESH_SEED}"
printf 'W&B tracking: %s (startup retries online; WANDB_MODE=offline records to disk only)\n' "${WANDB_MODE}"
mkdir -p "${PROJECT_DIR}/runs/slurm"
# 891 training runs vs 1278 in the 11.5 h full-location baseline; 16 h leaves margin.
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_fourier_k256.sh" \
    --job-name=mesh_hist_random70 --time=16:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_random70_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_random70_%j.err" \
    "$@"
