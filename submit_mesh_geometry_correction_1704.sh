#!/usr/bin/env bash
# Freeze a saved mesh model and fit a coefficient-based geometry correction.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"
export HOOD_CORRECTION_BASELINE_RUN="${HOOD_CORRECTION_BASELINE_RUN:-${PROJECT_DIR}/runs/mesh_impact_history/20260930_204155_3283115_1704}"
for artifact in hood_impact_best_model.pt config.json scalers.joblib splits.json prediction_times.npy; do
    if [[ ! -r "${HOOD_CORRECTION_BASELINE_RUN}/${artifact}" ]]; then
        printf 'ERROR: Required baseline artifact is missing or unreadable: %s/%s\n' \
            "${HOOD_CORRECTION_BASELINE_RUN}" "${artifact}" >&2
        printf 'Copy the complete baseline run to this machine, or set HOOD_CORRECTION_BASELINE_RUN to its run directory.\n' >&2
        exit 2
    fi
done
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is unavailable. Run this on the DeltaAI login node." >&2
    exit 127
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hood-impact-geometry-correction}"
printf 'Baseline (frozen): %s\n' "${HOOD_CORRECTION_BASELINE_RUN}"
printf 'Experiment: fixed geometry coefficients multiplied by learned impact/time response functions\n'
printf 'Training: ordinary acceleration MSE; split, scalers, locations and times restored from baseline\n'
mkdir -p "${PROJECT_DIR}/runs/slurm"
exec sbatch \
    --chdir="${PROJECT_DIR}" \
    --job-name=mesh_geom_correction \
    --export=ALL \
    --output="${PROJECT_DIR}/runs/slurm/mesh_geometry_correction_1704_%j.out" \
    --error="${PROJECT_DIR}/runs/slurm/mesh_geometry_correction_1704_%j.err" \
    "$@" "${PROJECT_DIR}/run_mesh_geometry_correction_1704.sbatch"
