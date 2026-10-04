#!/bin/bash -l
#SBATCH --job-name=mesh_geom_correction
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=mesh_geometry_correction_1704_%j.out
#SBATCH --error=mesh_geometry_correction_1704_%j.err

# Module activation can reference unset variables, so do not enable nounset.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root or use the submission wrapper}"
PROJECT_DIR=$PWD
BASELINE_RUN=${HOOD_CORRECTION_BASELINE_RUN:-${PROJECT_DIR}/runs/mesh_impact_history/20260930_204155_3283115_1704}
for artifact in hood_impact_best_model.pt config.json scalers.joblib splits.json prediction_times.npy; do
    if [[ ! -r "${BASELINE_RUN}/${artifact}" ]]; then
        printf 'ERROR: Required baseline artifact is missing or unreadable: %s/%s\n' \
            "${BASELINE_RUN}" "${artifact}" >&2
        exit 2
    fi
done

PYTORCH_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}
MESH_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history}
if ! type module >/dev/null 2>&1; then
    echo "ERROR: DeltaAI's module command is unavailable in the job environment." >&2
    exit 2
fi
module load "${PYTORCH_MODULE}"
if [[ ! -r "${MESH_VENV}/bin/activate" ]]; then
    printf 'ERROR: Environment not found: %s\n' "${MESH_VENV}" >&2
    echo "Run bash -l setup_mesh_impact_history_env.sh first, or set HOOD_MESH_VENV." >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${MESH_VENV}/bin/activate"
PYTHON_BIN=$(command -v python)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_MESH_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MESH_MKL_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-geometry-correction}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}
STAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR=${HOOD_CORRECTION_OUTPUT_DIR:-${PROJECT_DIR}/runs/mesh_geometry_correction/${STAMP}_${JOB_ID}_1704}
RUN_NAME=${HOOD_CORRECTION_RUN_NAME:-mesh_geometry_correction_1704_${JOB_ID}}
printf 'Node: %s\nPython: %s\nModule: %s\n' "$(hostname)" "${PYTHON_BIN}" "${PYTORCH_MODULE}"
printf 'Commit: %s\nDataset: %s\nBaseline: %s\nOutput: %s\n' \
    "$(git rev-parse HEAD)" "${DATA_ROOT}" "${BASELINE_RUN}" "${OUTPUT_DIR}"
printf 'W&B: %s | project: %s | entity: %s\n' \
    "${WANDB_MODE}" "${WANDB_PROJECT}" "${WANDB_ENTITY:-default team}"
module list 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

TRAIN_ARGS=(
    --baseline-run "${BASELINE_RUN}"
    --data-root "${DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}" --device cuda
    --epochs "${HOOD_CORRECTION_EPOCHS:-300}"
    --batch-size "${HOOD_CORRECTION_BATCH_SIZE:-64}"
    --lr "${HOOD_CORRECTION_LR:-1e-3}"
    --weight-decay "${HOOD_CORRECTION_WEIGHT_DECAY:-0}"
    --seed "${HOOD_CORRECTION_SEED:-42}"
    --run-name "${RUN_NAME}"
    --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT}"
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
OPTIONS=(
    ANCHOR_SPACING_MM:anchor-spacing-mm PROFILE_K:profile-k
    GEOMETRY_NOISE_FLOOR_MM:geometry-noise-floor-mm
    WIDTH:correction-width LAYERS:correction-layers
    BASELINE_CHUNK_SIZE:baseline-neighborhood-chunk-size
    FOURIER_NUM_FREQUENCIES:fourier-num-frequencies
    FOURIER_MIN_FREQUENCY_HZ:fourier-min-frequency-hz
    FOURIER_MAX_FREQUENCY_HZ:fourier-max-frequency-hz
)
for option in "${OPTIONS[@]}"; do
    variable="HOOD_CORRECTION_${option%%:*}"
    if [[ -n "${!variable:-}" ]]; then
        TRAIN_ARGS+=("--${option#*:}" "${!variable}")
    fi
done
if [[ -n "${HOOD_MESH_NEIGHBOR_CACHE_DIR:-}" ]]; then
    TRAIN_ARGS+=(--neighbor-cache-dir "${HOOD_MESH_NEIGHBOR_CACHE_DIR}")
fi
if [[ "${HOOD_CORRECTION_NO_PLOTS:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--no-plots)
fi

# Use Slurm's allocated GPU directly; no extra srun interconnect step is needed.
printf 'Training command:'
printf ' %q' "${PYTHON_BIN}" -u train_mesh_geometry_correction.py "${TRAIN_ARGS[@]}"
printf '\n'
"${PYTHON_BIN}" -u train_mesh_geometry_correction.py "${TRAIN_ARGS[@]}"
printf 'Finished. Baseline/corrected metrics and acceleration/HIC plots: %s\n' "${OUTPUT_DIR}"
