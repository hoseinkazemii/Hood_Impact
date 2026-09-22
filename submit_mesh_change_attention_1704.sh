#!/usr/bin/env bash
# Change-region attention: anchors sit on the geometry that differs between
# near-clone TRAINING siblings, not on the nodes nearest the impact point.
# Objective is plain normalized acceleration MSE; HIC15 stays postprocessing.
# Cluster-B holdout: test 4,5; validation 11; training uses the other nine.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
if [[ -n "${HOOD_MESH_RESUME_FROM:-}" ]]; then
    echo "ERROR: This launcher starts a fresh experiment and does not support resume. Unset HOOD_MESH_RESUME_FROM." >&2
    exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi

# Pin this experiment even if older exports are still in the environment.
export HOOD_MESH_IMPACTOR_NODES="286"
export HOOD_MESH_TEST_DESIGNS="4 5"
export HOOD_MESH_VAL_DESIGNS="11"
export HOOD_CHANGE_LAYERS="2"
export HOOD_CHANGE_K="${HOOD_CHANGE_K:-256}"
export HOOD_CHANGE_ANCHORS="${HOOD_CHANGE_ANCHORS:-32}"
export HOOD_CHANGE_SCALE_MM="20"
export HOOD_CHANGE_CHUNK_SIZE="256"
# Rank anchors by variation within a near-clone family. Pooled ranking puts
# every anchor on the base-shape difference between families instead; keep
# HOOD_CHANGE_SCOPE=all_designs for that ablation only.
export HOOD_CHANGE_SCOPE="${HOOD_CHANGE_SCOPE:-within_family}"
export HOOD_CHANGE_MIN_CHANGE_MM="${HOOD_CHANGE_MIN_CHANGE_MM:-0.5}"
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_change_attention_1704_k${HOOD_CHANGE_K}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-mesh-attention}"

printf 'Experiment: %s change anchors, %s nodes per change region, 2 patch attention layers\n' \
    "${HOOD_CHANGE_ANCHORS}" "${HOOD_CHANGE_K}"
printf 'Anchor scope: %s | minimum change %s mm | headform (286 nodes) excluded\n' \
    "${HOOD_CHANGE_SCOPE}" "${HOOD_CHANGE_MIN_CHANGE_MM}"
printf 'Loss: plain normalized acceleration MSE\n'
printf 'Holdout: train 0,1,2,3,6,7,8,9,10 | validation 11 | test 4,5 (whole cluster B, unopened)\n'
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name="mesh_change_k${HOOD_CHANGE_K}" \
    --time=48:00:00 \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/${HOOD_MESH_RUN_NAME}_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/${HOOD_MESH_RUN_NAME}_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_change_attention_1704.sbatch"
