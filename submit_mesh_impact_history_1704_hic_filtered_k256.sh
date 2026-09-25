#!/usr/bin/env bash
# Fresh neighborhood experiment on locations with cross-design HIC15 variation.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="${HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT:-10}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_hic_range_${HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT}pct}"
printf 'Location filter: HIC15 range / mean >= %s%% across all 12 designs\n' "${HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT}"
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_local_sensitivity_k256.sh" \
    --job-name=mesh_hist_hic_filtered_k256 "$@"
