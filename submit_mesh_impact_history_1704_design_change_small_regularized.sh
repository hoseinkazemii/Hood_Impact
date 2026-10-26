#!/usr/bin/env bash
# Design-change sweep, variant "small_regularized": both changes together, the
# small change path (no change self-attention, 2 octaves; 0.15 M parameters)
# with dropout 0.2 and AdamW weight decay 0.05. Everything else matches the
# reference run submit_mesh_impact_history_1704_design_change_nodes.sh.
# See the "small" and "regularized" launchers for the reasoning behind each.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_small_regularized.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_LAYERS="0"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="2"
# The 2026-10-25 sweep ran on the whole deck with impact-relative change tokens.
export HOOD_MESH_MESH_NODES="all"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="impact_relative"
export HOOD_MESH_DROPOUT="0.2"
export HOOD_MESH_WEIGHT_DECAY="0.05"
# Pin the rest of the reference settings so stale exports cannot leak in.
export HOOD_MESH_EPOCHS="100" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_RUN_NAME="mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_small_regularized"

printf 'Sweep variant: small change path (no change self-attention, 2 octaves) | dropout 0.2 | weight decay 0.05\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_small_reg \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_small_regularized_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_small_regularized_%j.err" \
    "$@"
