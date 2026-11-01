#!/usr/bin/env bash
# Chunked teacher forcing on the design-change model. The 63-point history is
# predicted as 5 chunks in time order (13, 13, 13, 13 and 11 points: 0-4.8,
# 5.2-10.0, 10.4-15.2, 15.6-20.4 and 20.8-24.8 ms). Every time query also reads
# the acceleration one chunk earlier, and the temporal attention is
# block-causal, so a chunk sees only the accelerations of earlier chunks:
#   training     the true earlier chunks (teacher forcing), all in one pass
#   validation   the model's own predictions of the earlier chunks (rollout),
#                as the model is used; the best checkpoint is chosen on this.
#                W&B also logs val/teacher_forced_loss, the same model fed the
#                true earlier chunks; the gap between the two is the error
#                that rollout feeds back.
#   test         rollout as usual (metrics.json, test_acceleration_histories.csv),
#                and again fed the true earlier chunks in teacher_forced/: the
#                best case, as if the first chunks were measured.
# time_windows.json gives the test RMSE in each chunk for both, and for the mean
# training curve at the same impact location (a predictor that ignores the
# geometry). Run 3458374 (same split and grid, no history) on these chunks:
#   model           22.4  13.7  13.3  13.1  11.8 g
#   location mean   22.9  14.5  13.3  13.7  12.9 g
# Teacher forcing helps only if chunks 2-5 fall below the model row, first fed
# the true earlier chunks, and then in rollout. See README_teacher_forcing.md.
#
# Everything else is the no-gate design-change model of the long runs: inner
# panel, XYZ change tokens, no change self-attention layers, no displacement
# gate, k256 FiLM neighborhoods, Fourier time 20-640 Hz, all 142 locations,
# stride 16, test 4,5 | validation 11, batch 8, lr 3e-4, dropout 0.1, weight
# decay 1e-5, seed 42, plain acceleration MSE. 100 epochs, as a first test;
# 3458374 took 5.2 h per 100 epochs, and the extra validation pass and decoder
# rollouts add a few percent.
#
# A timeout keeps hood_impact_last_model.pt from the last completed epoch;
# continue it (same schedule and split) with
#   bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh <run dir> --time=08:00:00
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_teacher_forcing.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_TEACHER_FORCING_CHUNKS="5"
export HOOD_MESH_HOLDOUT_TEST_DESIGNS="4 5"
export HOOD_MESH_HOLDOUT_VAL_DESIGNS="11"
export HOOD_MESH_HOLDOUT_ALLOW_CLONE_LEAK="0"
export HOOD_MESH_DESIGN_CHANGE_LAYERS="0"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="2"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="xyz"
export HOOD_MESH_DESIGN_CHANGE_DISPLACEMENT_GATE="false"
export HOOD_MESH_MESH_NODES="inner_panel"
export HOOD_MESH_DROPOUT="0.1"
export HOOD_MESH_WEIGHT_DECAY="0.00001"
export HOOD_MESH_EPOCHS="100" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_DESIGN_CHANGE_TIME_STRIDE="16"
export HOOD_MESH_FOURIER_NUM_FREQUENCIES="6" HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ="20"
export HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ="640"
export HOOD_MESH_RUN_NAME="mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_no_gate_teacher_forced_5chunks"

printf 'Teacher forcing: 5 time chunks | true earlier chunks in training, rollout in validation and test | 100 epochs\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_teacher --time=10:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_teacher_forcing_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_teacher_forcing_%j.err" \
    "$@"
