#!/usr/bin/env bash
# Design-change nodes (DesignChangeHistoryNet, mesh_design_change.py), as an A/B
# against the no-location-shortcut run 20261016_151923_3394500_1704. Everything
# matches that run (k256 FiLM neighborhood network, Fourier time, all 142
# locations, stride 16, query residual off, headform out of pooling, cluster-B
# holdout: test 4,5 | validation 11, 100 epochs, batch 8, seed 42, plain
# acceleration MSE) except:
#   --design-change-nodes true   every node that differs between near-clone
#                                designs (pairs with < 300 nodes moved > 0.1 mm:
#                                110-333 nodes per design) becomes its own token,
#                                so pooling cannot average it away. After every
#                                decoder block the time queries read these tokens
#                                with a second cross-attention whose values scale
#                                with each node's displacement from its cluster
#                                base (designs 0, 4, 6 and 10).
# The readout starts at zero, so with seed 42 the network starts exactly as the
# baseline did, and the base designs always get the backbone's prediction.
# The change path defaults to the sweep winner (run 3447060): no change
# self-attention layers. Since 2026-10-26 each change token is just the node's
# XYZ (--design-change-features xyz), and the network sees only the hood inner
# panel (--mesh-nodes inner_panel: no headform, outer skin, hinges or
# fasteners). The 2026-10 runs used the whole deck and 18- or 30-input tokens;
# the ..._small/_regularized wrappers pin those settings to repeat them.
# Compare design_difference_r2.json, test_design_sensitivity.json and
# hic_metrics.json with the baseline run. See README_design_change_nodes.md.
#
# Preflight finds the change nodes from the decks (building
# Data/HoodImpact_1704_EuroNCAP/design_difference_cache on first use) and runs a
# CUDA forward/backward through the change path before the dataset loads.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_nodes.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_NODES="true"
export HOOD_MESH_DESIGN_CHANGE_LAYERS="${HOOD_MESH_DESIGN_CHANGE_LAYERS:-0}"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="${HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES:-2}"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="${HOOD_MESH_DESIGN_CHANGE_FEATURES:-xyz}"
export HOOD_MESH_MESH_NODES="${HOOD_MESH_MESH_NODES:-inner_panel}"
export HOOD_MESH_GEOMETRY_FILM="none"
export HOOD_MESH_POOLING_QUERY_RESIDUAL="false"
export HOOD_MESH_POOL_IMPACTOR_NODES="false"
# Pin the comparison settings even if another experiment's values are exported.
export HOOD_MESH_TIME_SUBSAMPLE_STRIDE="16"
export HOOD_MESH_HIC_RANGE_THRESHOLD_PERCENT="0"
export HOOD_MESH_MAX_TRAIN_TIME=""
export HOOD_MESH_LOCATION_HOLDOUT=""
export HOOD_MESH_EPOCHS="${HOOD_MESH_EPOCHS:-100}"
export HOOD_MESH_BATCH_SIZE="${HOOD_MESH_BATCH_SIZE:-8}"
export HOOD_MESH_LR="${HOOD_MESH_LR:-0.0003}"
export HOOD_MESH_WEIGHT_DECAY="${HOOD_MESH_WEIGHT_DECAY:-0.00001}"
export HOOD_MESH_SEED="${HOOD_MESH_SEED:-42}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_nodes}"
# Online by default so epochs and plots reach the W&B website; training retries
# a slow W&B start (job 3317059 timed out) before failing ahead of data loading.
export WANDB_MODE="${WANDB_MODE:-online}"

printf 'Design-change nodes: ON | %s change self-attention layers | token features %s\n' \
    "${HOOD_MESH_DESIGN_CHANGE_LAYERS}" "${HOOD_MESH_DESIGN_CHANGE_FEATURES}"
printf 'Mesh nodes: %s\n' "${HOOD_MESH_MESH_NODES}"
printf 'Location shortcuts: query residual OFF | headform rows excluded from pooling | geometry FiLM none\n'
printf 'Baseline for comparison: 20261016_151923_3394500_1704 (same settings, no change nodes)\n'
printf 'Training: %s epochs | batch %s | lr %s | weight decay %s | seed %s | plain acceleration MSE\n' \
    "${HOOD_MESH_EPOCHS}" "${HOOD_MESH_BATCH_SIZE}" "${HOOD_MESH_LR}" \
    "${HOOD_MESH_WEIGHT_DECAY}" "${HOOD_MESH_SEED}"
printf 'W&B tracking: %s (startup retries online; WANDB_MODE=offline records to disk only)\n' "${WANDB_MODE}"
mkdir -p "${PROJECT_DIR}/runs/slurm"
# The baseline needed 11.5 h on one GH200; the change path adds a few percent.
# 14 h leaves margin while still backfilling (job 3440305 queued ~1.5 days at 24 h);
# a timeout resumes from the latest checkpoint via the *_resume.sh wrapper.
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_fourier_k256.sh" \
    --job-name=mesh_hist_design_change --time=14:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_nodes_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_nodes_%j.err" \
    "$@"
