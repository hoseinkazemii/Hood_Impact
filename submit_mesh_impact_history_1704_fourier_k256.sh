#!/usr/bin/env bash
# Fresh Fourier-time experiment: retain the k256 FiLM neighborhood architecture.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"

export HOOD_MESH_TIME_ENCODING="fourier"
export HOOD_MESH_IMPACT_CONDITIONING="film"
export HOOD_MESH_FOURIER_NUM_FREQUENCIES="${HOOD_MESH_FOURIER_NUM_FREQUENCIES:-6}"
export HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ="${HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ:-20}"
export HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ="${HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ:-640}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_hic${HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT:-10}pct}"

printf 'Time encoding: Fourier | %s frequency pairs from %s to %s Hz\n' \
    "${HOOD_MESH_FOURIER_NUM_FREQUENCIES}" "${HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ}" \
    "${HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ}"
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_hic_filtered_k256.sh" \
    --job-name=mesh_hist_fourier_k256 "$@"
