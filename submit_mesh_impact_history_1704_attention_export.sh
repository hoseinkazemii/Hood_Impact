#!/usr/bin/env bash
# Fresh 30%-HIC-filtered k256 MSE training; export exactly ten test cases.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_EXPORT_ATTENTION=1
export HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT=30
export HOOD_MESH_ATTENTION_SPLIT=test
export HOOD_MESH_ATTENTION_SUMMARY_ONLY=0
# Five qualifying locations spanning the retained location-ID list, on both
# held-out designs (4 and 5): locations 9, 54, 102, 125, 142.
export HOOD_MESH_ATTENTION_RUNS="577 622 670 693 710 719 764 812 835 852"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_k256_hic30_attention10}"
printf 'HIC threshold: 30%% | Export: 10 test cases (5 locations on each test design)\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_hic_filtered_k256.sh" \
    --job-name=mesh_hist_attention --time=48:00:00 "$@"
