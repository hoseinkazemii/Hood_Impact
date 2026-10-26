#!/usr/bin/env bash
# Design-change sweep, variant "regularized": the reference change path
# (2 self-attention layers, 5 octaves) with dropout 0.2 instead of 0.1 and
# AdamW weight decay 0.05 instead of 1e-5. Everything else matches the
# reference run submit_mesh_impact_history_1704_design_change_nodes.sh.
#
# Why: by epoch 52 the reference's training MSE (0.051) is a third of its
# validation MSE (0.173) and still falling while validation has flattened.
# Weight decay 1e-5 does almost nothing with AdamW at lr 3e-4: over ~16k steps
# it shrinks weights by about 0.002%. At 0.05 the shrink is about 11%, a
# standard transformer setting. Dropout and weight decay act on the backbone
# too, so check that absolute accuracy (metrics.json, hic_metrics.json) holds.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_regularized.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_LAYERS="2"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="5"
# The 2026-10-25 sweep ran on the whole deck with impact-relative change tokens.
export HOOD_MESH_MESH_NODES="all"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="impact_relative"
export HOOD_MESH_DROPOUT="0.2"
export HOOD_MESH_WEIGHT_DECAY="0.05"
# Pin the rest of the reference settings so stale exports cannot leak in.
export HOOD_MESH_EPOCHS="100" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_RUN_NAME="mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_regularized"

printf 'Sweep variant: reference change path (2 layers, 5 octaves) | dropout 0.2 | weight decay 0.05\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_reg \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_regularized_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_regularized_%j.err" \
    "$@"
