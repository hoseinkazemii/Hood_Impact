#!/usr/bin/env bash
# Head-centred, location-free network (README_head_centred.md), as an A/B
# against the design-change-nodes run 20261023_154041_3440305_1704. Everything
# matches that run (DesignChangeHistoryNet, k256 neighborhoods x 2, Fourier
# time, all 142 locations, stride 16, query residual off, headform out of
# pooling, cluster-B holdout: test 4,5 | validation 11, 100 epochs, batch 8,
# lr 3e-4, seed 42, plain acceleration MSE) except:
#   --impact-conditioning head_relative  no impact location is read. The origin
#                                moves to the headform centroid (the first 286
#                                mesh rows), every node enters as its offset from
#                                it, and the local pooling tokens' radial prior is
#                                centred there. No impact embedding or FiLM.
#   --node-features structure    each node also gets its panel (outer / inner /
#                                hinge), unit surface normal and log outer-inner gap.
#   --mirror-augmentation true   half of each training batch is mirrored left-right
#                                (the impact is purely vertical, so that is exact).
# Compare design_difference_r2.json, test_design_sensitivity.json, metrics.json
# and hic_metrics.json with 20261023_154041 and the no-change-node baseline
# 20261016_151923_3394500_1704.
#
# This skips submit_mesh_impact_history_1704_fourier_k256.sh, which pins
# impact conditioning to film, and pins its Fourier settings here instead.
# Preflight loads the first run with its structure channels and runs a CUDA
# forward/backward through the head-relative change model before the dataset loads.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_head_centred.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_IMPACT_CONDITIONING="head_relative"
export HOOD_MESH_NODE_FEATURES="${HOOD_MESH_NODE_FEATURES:-structure}"
export HOOD_MESH_MIRROR_AUGMENTATION="${HOOD_MESH_MIRROR_AUGMENTATION:-true}"
export HOOD_MESH_DESIGN_CHANGE_NODES="${HOOD_MESH_DESIGN_CHANGE_NODES:-true}"
export HOOD_MESH_DESIGN_CHANGE_LAYERS="${HOOD_MESH_DESIGN_CHANGE_LAYERS:-2}"
export HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES="${HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES:-5}"
export HOOD_MESH_TIME_ENCODING="fourier"
export HOOD_MESH_FOURIER_NUM_FREQUENCIES="6"
export HOOD_MESH_FOURIER_MIN_FREQUENCY_HZ="20"
export HOOD_MESH_FOURIER_MAX_FREQUENCY_HZ="640"
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
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_head_centred_structure_mirror_design_change}"
# Online by default so epochs and plots reach the W&B website.
export WANDB_MODE="${WANDB_MODE:-online}"

printf 'Head-centred input: ON | no impact location read | node features: %s | mirror augmentation: %s\n' \
    "${HOOD_MESH_NODE_FEATURES}" "${HOOD_MESH_MIRROR_AUGMENTATION}"
printf 'Design-change nodes: %s | %s change layers | %s octaves of head-relative XY\n' \
    "${HOOD_MESH_DESIGN_CHANGE_NODES}" "${HOOD_MESH_DESIGN_CHANGE_LAYERS}" "${HOOD_MESH_DESIGN_CHANGE_FOURIER_OCTAVES}"
printf 'Baselines: 20261023_154041_3440305_1704 (design-change nodes, FiLM) | 20261016_151923_3394500_1704\n'
printf 'Training: %s epochs | batch %s | lr %s | weight decay %s | seed %s | plain acceleration MSE\n' \
    "${HOOD_MESH_EPOCHS}" "${HOOD_MESH_BATCH_SIZE}" "${HOOD_MESH_LR}" \
    "${HOOD_MESH_WEIGHT_DECAY}" "${HOOD_MESH_SEED}"
printf 'W&B tracking: %s\n' "${WANDB_MODE}"
mkdir -p "${PROJECT_DIR}/runs/slurm"
# Same 14 h as the design-change launcher (its baseline needed 11.5 h on one
# GH200); the 7 structure channels only widen the first node-embedding layer.
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_hic_filtered_k256.sh" \
    --job-name=mesh_hist_head_centred --time=14:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_head_centred_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_head_centred_%j.err" \
    "$@"
