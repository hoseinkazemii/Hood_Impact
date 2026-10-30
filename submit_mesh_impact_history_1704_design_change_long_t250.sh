#!/usr/bin/env bash
# The long no-gate design-change run (submit_mesh_impact_history_1704_design_change_long.sh)
# on a 4x finer time grid. Everything else is identical, 800 epochs included, so
# the two runs form an A/B on time resolution alone:
#   time grid   stride 4: 250 of the 1000 source samples (0.1 ms apart) instead of
#               stride 16: 63 (0.4 ms apart).
#   Fourier time features   8 frequencies, 20-2560 Hz (doublings) instead of 6,
#               20-640 Hz. With 640 Hz as the top feature the decoder has no input
#               at the detail the extra points add. 10.5% of the near-clone
#               difference power lies above 640 Hz (time_resolution_check.json in
#               runs/mesh_impact_history/nodes_difference_attention/sweep_comparison);
#               2560 Hz stays below this grid's ~5 kHz Nyquist.
# Cost: a full-size training step (17,102 inner-panel nodes, k256) measured
# 1.00-1.02x the 63-point step, because the neighborhood encoder dominates. So
# 800 epochs should take ~41-42 h of the 48 h limit, like the 63-point run.
#
# Comparing with 63-point runs: RMSE and difference R2 on 250 points include
# detail the 63-point runs are never scored on. Every 4th point of this run's
# histories (stride 4 x 4 = stride 16, both from sample 0) is exactly the
# 63-point grid; compare there too.
#
# A timeout keeps hood_impact_last_model.pt from the last completed epoch;
# continue it (same schedule and grid) with
#   bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh <run dir> --time=24:00:00
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_long_t250.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_DESIGN_CHANGE_LAYERS="0"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="2"
export HOOD_MESH_DESIGN_CHANGE_FEATURES="xyz"
export HOOD_MESH_DESIGN_CHANGE_DISPLACEMENT_GATE="false"
export HOOD_MESH_MESH_NODES="inner_panel"
export HOOD_MESH_DROPOUT="0.1"
export HOOD_MESH_WEIGHT_DECAY="0.00001"
export HOOD_MESH_EPOCHS="800" HOOD_MESH_BATCH_SIZE="8" HOOD_MESH_LR="0.0003" HOOD_MESH_SEED="42"
export HOOD_MESH_DESIGN_CHANGE_TIME_STRIDE="4"
export HOOD_MESH_FOURIER_NUM_FREQUENCIES="8" HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ="20"
export HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ="2560"
export HOOD_MESH_RUN_NAME="mesh_history_clusterB_k256_fourier_no_shortcuts_design_change_no_gate_800ep_t250"

printf 'Long run, fine time grid: 800 epochs | 250 time points | Fourier 20-2560 Hz | no displacement gate\n'
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --job-name=mesh_hist_dc_long_t250 --time=48:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_long_t250_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_long_t250_%j.err" \
    "$@"
