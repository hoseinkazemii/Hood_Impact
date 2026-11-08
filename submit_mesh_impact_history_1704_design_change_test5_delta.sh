#!/usr/bin/env bash
# Design 5 held out with its near-clone 4 in training, on NCSA Delta (not DeltaAI).
#   test        design 5 (cluster B); its near-clone 4 is in training
#   validation  design 3 (cluster A); its near-clones 0, 1, 2 stay in training
#   training    designs 0, 1, 2, 4, 6, 7, 8, 9, 10, 11 (1420 runs)
#
# Question: do predictions for near-clones collapse because cluster B's shape is
# in no training design (out of distribution), or because no training pair of
# near-clones differs enough? This split gives the model both: design 4's shape,
# and every within-cluster pair whose true curves differ more than 4 and 5 do.
# Mean |difference| over all 142 locations, 63 points: 6-7, 7-8, 7-9 9.1-9.3 g;
# 10-11 8.5 g; 4-5 7.3 g; the other pairs 4.1-7.3 g. Validation 3 gives up only
# small pairs (0-3 4.1 g, 1-3 6.6 g, 2-3 6.4 g) and, like the test design, is a
# held-out near-clone whose siblings are in training.
#
# Preview from run 20261103_010645_3501305_1704 (4 in training, 5 = validation,
# so its checkpoint was chosen on 5): design 5 RMSE 11.1 g, against 15.5 g in
# 3458374 (4 and 5 held out) and 11.7 g for copying design 4's true curve; but
# the 4-5 difference stays collapsed (R2 0.03, predicted 2.8 g vs true 7.3 g).
# 4-5 also has the largest solver time-step mismatch of all near-clone pairs
# (median ~27k vs ~40k increments), so much of its far-field difference is that
# artifact rather than geometry.
#
# Model and training match run 3501305 (test 7): no displacement gate, inner
# panel, XYZ change tokens, 63 time points, 100 epochs, batch 8, lr 3e-4, weight
# decay 1e-5, dropout 0.1, seed 42. Preflight needs --allow-clone-leak for this
# split. design_difference_r2.json "test_vs_train" scores 5 against 4: the
# difference R2, and design 5's RMSE against copying design 4's true curve.
#
# Delta: x86_64 (AMD Milan), its own home directory, no access to the DeltaAI
# venv. Shared with DeltaAI: /work/hdd only. One-time setup on a Delta login node:
#   1. git clone -b feature/design-change-node-injection \
#          https://github.com/hoseinkazemii/Hood_Impact.git ~/projects/Hood_Impact
#   2. the 1704 dataset (7.4 GB, with the neighbor-graph and design-difference
#      caches) in Data/HoodImpact_1704_EuroNCAP, or anywhere with
#      HOOD_MESH_DATA_ROOT=<dir> exported before submitting (a /work path adds
#      --constraint=work). From a DeltaAI login node:
#        rsync -a ~/projects/Hood_Impact/Data/HoodImpact_1704_EuroNCAP/ \
#              /work/hdd/bbqg/$USER/HoodImpact_1704_EuroNCAP/
#   3. bash -l setup_mesh_impact_history_env_delta.sh, then wandb login
#
# GPU: GPU memory is no constraint (~3 GiB at batch 8). Locally, TF32 matmuls
# change the step time by <1% and the laptop-to-GH200 speedup (4.6x) tracks FP32
# throughput (5.2x) rather than memory bandwidth (10.4x), so the step looks
# FP32-bound. Delta's FP32 rates: A40 37 TFLOPS, A100 19.5, H200 67 (GH200 67).
# Estimated from 3501305's 6.4 h on GH200 (same 1420 training runs), not
# measured on Delta: A40 ~11 h, A100 ~20 h, H200 ~6.5 h. Charge factors: A40
# 0.5, A100 1.0, H200 3.0, so A40 is the default (100 nodes). The limits below
# leave about 2x margin; jobs are charged for time used, not the limit. Lower a
# pending or running job's limit with: scontrol update jobid=<id> TimeLimit=...
#   HOOD_DELTA_PARTITION=gpuH200x8 bash ...   fastest; 8 nodes, 3x charge
#   HOOD_DELTA_PARTITION=gpuA100x4 bash ...
# Account: bbqg-delta-gpu (Delta's <allocation>-delta-gpu form; the CPU work
# used bbqg-delta-cpu). Check yours with `accounts` and set HOOD_DELTA_ACCOUNT.
#
# A timeout keeps hood_impact_last_model.pt from the last completed epoch;
# continue it (same schedule and split) with the same Delta environment:
#   HOOD_MESH_PYTORCH_MODULE=python HOOD_MESH_VENV=$PWD/.venv-mesh-history-delta \
#   bash submit_mesh_impact_history_1704_local_sensitivity_resume.sh <run dir> \
#       --account=bbqg-delta-gpu --partition=gpuA40x4 --cpus-per-task=16 --time=12:00:00
#
# Submit from any directory on a Delta login node; extra arguments go to sbatch.
#   bash submit_mesh_impact_history_1704_design_change_test5_delta.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: This launcher is for Delta (x86_64); this machine is $(uname -m) (DeltaAI?)." >&2
    exit 2
fi

export HOOD_MESH_HOLDOUT_TEST_DESIGNS="5"
export HOOD_MESH_HOLDOUT_VAL_DESIGNS="3"
export HOOD_MESH_HOLDOUT_ALLOW_CLONE_LEAK="1"
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
export HOOD_MESH_RUN_NAME="mesh_history_test5_val3_k256_fourier_no_shortcuts_design_change_no_gate_delta"

# Delta environment for run_mesh_impact_history_1704.sbatch.
export HOOD_MESH_PYTORCH_MODULE="${HOOD_MESH_PYTORCH_MODULE:-python}"
export HOOD_MESH_VENV="${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history-delta}"
if [[ ! -r "${HOOD_MESH_VENV}/bin/activate" ]]; then
    echo "ERROR: Delta environment not found: ${HOOD_MESH_VENV}" >&2
    echo "Run bash -l setup_mesh_impact_history_env_delta.sh first, or set HOOD_MESH_VENV." >&2
    exit 2
fi
DATA_ROOT="${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}"
if [[ ! -d "${DATA_ROOT}/inp_files" || ! -d "${DATA_ROOT}/output_history_acc" ]]; then
    echo "ERROR: 1704 dataset not found at ${DATA_ROOT} (expects inp_files/ and output_history_acc/)." >&2
    echo "Copy it there or export HOOD_MESH_DATA_ROOT=<dataset dir>." >&2
    exit 2
fi

ACCOUNT="${HOOD_DELTA_ACCOUNT:-bbqg-delta-gpu}"
PARTITION="${HOOD_DELTA_PARTITION:-gpuA40x4}"
case "${PARTITION}" in
    gpuA40x4)  TIME_LIMIT="24:00:00"; CPUS=16 ;;  # 64 cores / 4 GPUs
    gpuA100x4) TIME_LIMIT="40:00:00"; CPUS=16 ;;  # 64 cores / 4 GPUs
    gpuH200x8) TIME_LIMIT="13:00:00"; CPUS=12 ;;  # 96 cores / 8 GPUs
    *)
        echo "ERROR: HOOD_DELTA_PARTITION must be gpuA40x4, gpuA100x4 or gpuH200x8; got ${PARTITION}" >&2
        exit 2 ;;
esac
EXTRA=()
if [[ "${DATA_ROOT}" == /work/* ]]; then
    # Delta schedules around file-system outages only for declared dependencies.
    EXTRA+=(--constraint=work)
fi

printf 'Design 5 held out: test 5 | validation 3 | design 4 in training | 100 epochs | batch 8 | no gate\n'
printf 'Delta: account %s | partition %s | %s CPUs | limit %s | venv %s\n' \
    "${ACCOUNT}" "${PARTITION}" "${CPUS}" "${TIME_LIMIT}" "${HOOD_MESH_VENV}"
printf 'Dataset: %s\n' "${DATA_ROOT}"
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_design_change_nodes.sh" \
    --account="${ACCOUNT}" --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
    --job-name=mesh_hist_dc_test5 --time="${TIME_LIMIT}" ${EXTRA[@]+"${EXTRA[@]}"} \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_test5_delta_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_design_change_test5_delta_%j.err" \
    "$@"
