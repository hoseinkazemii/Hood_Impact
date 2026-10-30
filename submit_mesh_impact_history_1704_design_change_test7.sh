#!/usr/bin/env bash
# The long no-gate design-change run (submit_mesh_impact_history_1704_design_change_long.sh)
# with a different holdout:
#   test        design 7 (cluster C); its near-clones 6, 8, 9 stay in training
#   validation  design 5 (cluster B); its near-clone 4 stays in training
#   training    designs 0, 1, 2, 3, 4, 6, 8, 9, 10, 11 (1420 runs instead of 1278)
#
# Why design 7: its edit changes the response more than any other design's.
# Mean |HIC15 difference| to 6, 8 and 9 is 7.7-8.1% over all 142 locations and
# 22-24% at the top-10 locations (up to 53%); for 4 vs 5 it is 3.7% and 8%.
# Validation 5 keeps the only other large within-cluster effect (10 vs 11) in
# training and gives up the small 4-5 pair, the one most affected by the solver
# time-step artifact.
#
# With siblings in training, copying them is a strong baseline (preflight needs
# --allow-clone-leak for this split). design_difference_r2.json therefore has a
# "test_vs_train" section: the difference R2 of 7 against 6, 8 and 9, and design
# 7's test RMSE for the model and for the mean true curve of 6, 8 and 9 at the
# same location (12.7 g over all 142 locations). Beating that mean is the goal.
#
# Everything else matches the long run: no displacement gate, inner panel, XYZ
# change tokens, 63 time points, batch 8, lr 3e-4, dropout 0.1, seed 42. With
# 11% more training runs an epoch takes ~205 s instead of 185 s, so 700 epochs
# (~40 h) keep the 48 h margin; the cosine schedule spans all 700. The final
# epoch's checkpoint is scored in <run>/last_checkpoint/ as well.
#
# A timeout keeps hood_impact_last_model.pt from the last completed epoch;
# continue it (same schedule and split) with
#   bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh <run dir> --time=24:00:00
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_test7.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_HOLDOUT_TEST_DESIGNS="7"
export HOOD_MESH_HOLDOUT_VAL_DESIGNS="5"
export HOOD_MESH_HOLDOUT_ALLOW_CLONE_LEAK="1"
export HOOD_MESH_DESIGN_CHANGE_LAYERS="0"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="2"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="xyz"
export HOOD_MESH_DESIGN_CHANGE_DISPLACEMENT_GATE="false"
export HOOD_MESH_MESH_NODES="inner_panel"
export HOOD_MESH_DROPOUT="0.1"
export HOOD_MESH_WEIGHT_DECAY="0.00001"
export HOOD_MESH_EPOCHS="700" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_DESIGN_CHANGE_TIME_STRIDE="16"
export HOOD_MESH_FOURIER_NUM_FREQUENCIES="6" HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ="20"
export HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ="640"
export HOOD_MESH_RUN_NAME="mesh_history_test7_val5_k256_fourier_no_shortcuts_design_change_no_gate_700ep"

printf 'Long run, design 7 held out: test 7 | validation 5 | 700 epochs | no displacement gate\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_test7 --time=48:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_test7_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_test7_%j.err" \
    "$@"
