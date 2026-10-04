#!/bin/bash -l
#SBATCH --job-name=temporal_dpn_1704
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=temporal_deeponet_1704_%j.out
#SBATCH --error=temporal_deeponet_1704_%j.err

# Reuse the working attention-job environment; no extra dependencies.
# Avoid nounset because module/conda activation may read unset variables.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root or use the submission wrapper}"
PROJECT_DIR=$PWD
PYTORCH_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}
DEEPONET_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history}
if ! type module >/dev/null 2>&1; then
    echo "ERROR: DeltaAI's module command is unavailable." >&2
    exit 2
fi
module load "${PYTORCH_MODULE}"
if [[ ! -r "${DEEPONET_VENV}/bin/activate" ]]; then
    echo "ERROR: Run bash -l setup_mesh_impact_history_env.sh first, or set HOOD_MESH_VENV." >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${DEEPONET_VENV}/bin/activate"
PYTHON_BIN=$(command -v python)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_MESH_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MESH_MKL_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-mesh-attention}

# Only infrastructure paths are shared with HOOD_MESH_*; stale attention-model,
# resume, split, and loss exports cannot change this fresh baseline experiment.
DATA_ROOT=${HOOD_TD_DATA_ROOT:-${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}}
STAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR=${HOOD_TD_OUTPUT_DIR:-${PROJECT_DIR}/runs/temporal_deeponet_1704/${STAMP}_${JOB_ID}_clusterB}
RUN_NAME=${HOOD_TD_RUN_NAME:-temporal_deeponet_1704_clusterB_${JOB_ID}}
TRAIN_ARGS=(
    --data-root "${DATA_ROOT}" --test-designs 4 5 --val-designs 11
    --device cuda --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}"
    --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT}"
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
for option in EPOCHS:epochs BATCH_SIZE:batch-size LR:lr WEIGHT_DECAY:weight-decay \
    SEED:seed TIME_SUBSAMPLE_STRIDE:time-subsample-stride MAX_TRAIN_TIME:max-train-time; do
    variable="HOOD_TD_${option%%:*}"
    if [[ -n "${!variable}" ]]; then
        TRAIN_ARGS+=("--${option#*:}" "${!variable}")
    fi
done
printf 'Node: %s\nPython: %s\nCommit: %s\nDataset: %s\nOutput: %s\n' \
    "$(hostname)" "${PYTHON_BIN}" "$(git rev-parse HEAD)" "${DATA_ROOT}" "${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Run in Slurm's existing batch allocation, preserving CUDA_VISIBLE_DEVICES.
# Python validates the dataset/split and probes this model before loading all data.
printf 'Training command:'
printf ' %q' "${PYTHON_BIN}" -u train_temporal_deeponet_1704.py "${TRAIN_ARGS[@]}"
printf '\n'
"${PYTHON_BIN}" -u train_temporal_deeponet_1704.py "${TRAIN_ARGS[@]}"
printf 'Finished. Checkpoints, metrics and acceleration histories: %s\n' "${OUTPUT_DIR}"
