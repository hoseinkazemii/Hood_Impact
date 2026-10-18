#!/usr/bin/env bash
# Geometry memory injection (Xia & Aviles-Rivero, "Do Neural Operators Forget
# Geometry?", ICML 2026), as an A/B against the no-location-shortcut run
# 20261016_151923_3394500_1704. Everything matches that run (k256 FiLM
# neighborhood network, Fourier time, all 142 locations, stride 16, query
# residual off, headform out of pooling, cluster-B holdout: test 4,5 |
# validation 11, 100 epochs, batch 8, seed 42) except:
#   --geometry-film all   a separate XYZ memory encoder (sin/cos octaves, as in
#                         the paper) FiLM-modulates the output of every
#                         neighborhood block (per node) and every latent mixing
#                         block (per token, pooled with the token's own
#                         attention weights). The decoder is not injected.
# The FiLM heads start at zero, so with seed 42 the network starts exactly as
# the baseline did. Compare test_design_sensitivity.json and hic_metrics.json
# with the baseline run.
#
# HOOD_MESH_GEOMETRY_FILM=latent runs the ablation without the node sites.
# Measured on 38,977-node meshes: about 0.9 GiB per mesh at batch 8 (baseline
# 0.7 GiB), roughly 7.6 GiB in total, and 2-4% more time per step.
#
# Submit from any directory; extra arguments are forwarded to sbatch.
#   bash submit_mesh_impact_history_1704_geometry_film.sh
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HOOD_MESH_GEOMETRY_FILM="${HOOD_MESH_GEOMETRY_FILM:-all}"
if [[ "${HOOD_MESH_GEOMETRY_FILM}" != "all" && "${HOOD_MESH_GEOMETRY_FILM}" != "latent" ]]; then
    echo "ERROR: HOOD_MESH_GEOMETRY_FILM must be all or latent for this launcher; got '${HOOD_MESH_GEOMETRY_FILM}'." >&2
    echo "The uninjected baseline is submit_mesh_impact_history_1704_no_location_shortcuts.sh." >&2
    exit 2
fi
export HOOD_MESH_GEOMETRY_FOURIER_OCTAVES="5"
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
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_clusterB_k256_fourier_no_shortcuts_geometry_film_${HOOD_MESH_GEOMETRY_FILM}}"
# Online by default so epochs and plots reach the W&B website; training retries
# a slow W&B start (job 3317059 timed out) before failing ahead of data loading.
export WANDB_MODE="${WANDB_MODE:-online}"

printf 'Geometry memory FiLM: %s | %s sin/cos octaves of normalized XYZ\n' \
    "${HOOD_MESH_GEOMETRY_FILM}" "${HOOD_MESH_GEOMETRY_FOURIER_OCTAVES}"
printf 'Location shortcuts: query residual OFF | headform rows excluded from pooling\n'
printf 'Baseline for comparison: 20261016_151923_3394500_1704 (same settings, no injection)\n'
printf 'Training: %s epochs | batch %s | lr %s | weight decay %s | seed %s\n' \
    "${HOOD_MESH_EPOCHS}" "${HOOD_MESH_BATCH_SIZE}" "${HOOD_MESH_LR}" \
    "${HOOD_MESH_WEIGHT_DECAY}" "${HOOD_MESH_SEED}"
printf 'W&B tracking: %s (startup retries online; WANDB_MODE=offline records to disk only)\n' "${WANDB_MODE}"
mkdir -p "${PROJECT_DIR}/runs/slurm"
# The baseline needed 11.5 h on one GH200; injection adds a few percent.
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_fourier_k256.sh" \
    --job-name=mesh_hist_geometry_film --time=24:00:00 \
    --output="${PROJECT_DIR}/runs/slurm/mesh_history_1704_geometry_film_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_history_1704_geometry_film_%j.err" \
    "$@"
