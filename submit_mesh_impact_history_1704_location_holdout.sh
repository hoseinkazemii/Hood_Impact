#!/usr/bin/env bash
# Unseen-location test: is the design collapse caused by test impact locations
# that the network already saw on the training designs? Same network and
# settings as the no-location-shortcut run (k256 FiLM neighborhood, Fourier
# time, stride 16, query residual off, headform out of pooling, cluster-B
# holdout), but the 142 locations are split as a checkerboard on the Euro-NCAP
# lattice (location_splits/euroncap1704_checkerboard.json):
#   71 even-parity locations: training designs 0-3,6-10 and validation design 11
#   71 odd-parity locations:  test designs 4,5 only
# Every test location is new to the network and has its lattice neighbours in
# training. Lateral mirror images (HIC within ~2%) share parity, so no test
# location's mirror twin is trained on. Split sizes: 639 train | 71 val | 142 test.
# Compare test_design_sensitivity.json and hic_metrics.json with the
# no-location-shortcut run.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_location_holdout.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_LOCATION_HOLDOUT="${PROJECT_DIR}/location_splits/euroncap1704_checkerboard.json"
if [[ ! -r "${HOOD_MESH_LOCATION_HOLDOUT}" ]]; then
    echo "ERROR: Location split not found: ${HOOD_MESH_LOCATION_HOLDOUT}" >&2
    exit 2
fi
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
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_no_shortcuts_location_holdout}"
export WANDB_MODE="${WANDB_MODE:-offline}"

printf 'Location holdout: %s\n' "${HOOD_MESH_LOCATION_HOLDOUT}"
printf '  71 locations on designs 0-3,6-11 (train/validation) | other 71 on designs 4,5 (test)\n'
printf 'Location shortcuts: query residual OFF | headform rows excluded from pooling\n'
printf 'Training: %s epochs | batch %s | lr %s | weight decay %s | seed %s\n' \
    "${HOOD_MESH_EPOCHS}" "${HOOD_MESH_BATCH_SIZE}" "${HOOD_MESH_LR}" \
    "${HOOD_MESH_WEIGHT_DECAY}" "${HOOD_MESH_SEED}"
printf 'W&B tracking: %s (offline records can be uploaded later with wandb sync)\n' "${WANDB_MODE}"
mkdir -p "${PROJECT_DIR}/runs/slurm"
# Half the training runs of the 11.5 h full-location baseline; 16 h leaves margin.
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_fourier_k256.sh" \
    --job-name=mesh_hist_location_holdout --time=16:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_location_holdout_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_location_holdout_%j.err" \
    "$@"
