#!/usr/bin/env bash
# Design-change sweep, variant "small": the change path without its two
# self-attention layers and with coarse impact-relative position features
# (sin/cos wavelengths 800 and 400 mm instead of 800 down to 50 mm). The path
# drops from 0.55 M to 0.15 M parameters. Everything else, including dropout
# 0.1 and weight decay 1e-5, matches the reference run
# submit_mesh_impact_history_1704_design_change_nodes.sh.
#
# Why: by epoch 52 the reference fits the training designs better than the
# baseline 20261016_151923_3394500_1704 (train MSE 0.051 vs 0.066) but
# validates worse on design 11 (0.173 vs 0.133). The change path is learning
# something specific to the six training edits. A path that cannot resolve
# 50 mm detail or patch layouts has less to memorize and should carry the edit
# effect over to 10->11 and 4->5. The response analysis
# (analyze_change_node_relevance.py) found the effect fades over ~250 mm, which
# the 400 mm wavelength and the plain distance feature still resolve.
#
# Result: best of the sweep (run 3447060), so ..._nodes.sh now uses its 0
# change layers. This wrapper repeats that run exactly (whole deck,
# impact-relative tokens).
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_small.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_LAYERS="0"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="2"
# The 2026-10-25 sweep ran on the whole deck with impact-relative change tokens
# and displacement-gated change readouts.
export HOOD_MESH_MESH_NODES="all"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="impact_relative"
export HOOD_MESH_DESIGN_CHANGE_DISPLACEMENT_GATE="true"
export HOOD_MESH_DROPOUT="0.1"
export HOOD_MESH_WEIGHT_DECAY="0.00001"
# Pin the rest of the reference settings so stale exports cannot leak in.
export HOOD_MESH_EPOCHS="100" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_RUN_NAME="mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_small"

printf 'Sweep variant: small change path (no change self-attention, 2 octaves) | dropout 0.1 | weight decay 1e-5\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_small \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_small_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_small_%j.err" \
    "$@"
