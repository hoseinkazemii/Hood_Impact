#!/usr/bin/env bash
# Long run of the best design-change model so far, run 20261027_055436_3458374_1704
# (inner-panel mesh, XYZ change tokens, no change self-attention layers; best
# test acceleration of all runs, RMSE 15.43 g vs 15.54 g for the neighborhood
# baseline 20261016_151923_3394500_1704), with two changes:
#   --design-change-displacement-gate false   the change readout is plain
#        cross-attention over the change tokens. 3458374 scaled each token's
#        value by its displacement from the cluster base / 10 mm, so its base
#        designs (0, 4, 6, 10) always got the backbone prediction. The tokens are
#        already only the nodes that differ between near-clones, so here the
#        attention alone decides which matter.
#   800 epochs instead of 100   the one-location test (loc 137) separated the
#        near-clones only after ~1500 full-batch epochs. 3458374 trained in 5.1 h
#        per 100 epochs (185 s/epoch on one GH200), so 800 epochs take ~41 h of
#        the 48 h DeltaAI limit; the cosine schedule spans all 800.
# Everything else matches 3458374: k256 FiLM neighborhoods, Fourier time, all 142
# locations, stride 16, test 4,5 | validation 11, batch 8, lr 3e-4, dropout 0.1,
# weight decay 1e-5, seed 42, plain acceleration MSE.
#
# Goal: beat the baseline on test accuracy (accel RMSE/R2, HIC) and on the 4-5
# difference curves (design_difference_r2.json test R2 > 0, i.e. better than
# predicting the same history for both designs).
#
# The best-validation checkpoint (design 11) is evaluated as usual. Its epoch
# can come early (3458374: epoch 60 of 100), so training also scores the final
# epoch's checkpoint into <run>/last_checkpoint/ (metrics.json, train/test
# acceleration histories, design_difference_r2.json).
#
# A timeout keeps hood_impact_last_model.pt from the last completed epoch;
# continue it (same 800-epoch schedule) with
#   bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh <run dir> --time=24:00:00
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_long.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_LAYERS="0"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="2"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="xyz"
export HOOD_MESH_DESIGN_CHANGE_DISPLACEMENT_GATE="false"
export HOOD_MESH_MESH_NODES="inner_panel"
export HOOD_MESH_DROPOUT="0.1"
export HOOD_MESH_WEIGHT_DECAY="0.00001"
# Pin the rest of the 3458374 settings so stale exports cannot leak in.
export HOOD_MESH_EPOCHS="800" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_RUN_NAME="mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_no_gate_800ep"

printf 'Long run: 800 epochs | change readout without displacement gate | inner panel | XYZ tokens\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_long --time=48:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_long_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_long_%j.err" \
    "$@"
