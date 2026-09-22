#!/bin/bash -l
#SBATCH --job-name=multiscale_1704
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=gpu_multiscale_1704_%j.out
#SBATCH --error=gpu_multiscale_1704_%j.err

# Reuse the working attention-job environment; no extra dependencies.
# Avoid nounset because module/conda activation may read unset variables.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root or use the submission wrapper}"
PROJECT_DIR=$PWD
PYTORCH_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}
CNN_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history}
if ! type module >/dev/null 2>&1; then
    echo "ERROR: DeltaAI's module command is unavailable." >&2
    exit 2
fi
module load "${PYTORCH_MODULE}"
if [[ ! -r "${CNN_VENV}/bin/activate" ]]; then
    echo "ERROR: Run bash -l setup_mesh_impact_history_env.sh first, or set HOOD_MESH_VENV." >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${CNN_VENV}/bin/activate"
PYTHON_BIN=$(command -v python)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_MESH_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MESH_MKL_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact}

# DATA_ROOT is the parent containing HoodImpact_1704_EuroNCAP.
DATA_ROOT=${HOOD_V6_DATA_ROOT:-${PROJECT_DIR}/Data}
OUTPUT_DIR=${HOOD_V6_OUTPUT_DIR:-${PROJECT_DIR}/runs/gpu_multiscale_1704/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}
TRAIN_ARGS=(
    --dataset 1704 --data-root "${DATA_ROOT}" --require-cuda
    --validation-design "${HOOD_VAL_DESIGN:-10}" --test-design "${HOOD_TEST_DESIGN:-11}"
    --epochs "${HOOD_V6_EPOCHS:-200}" --batch-size "${HOOD_V6_BATCH_SIZE:-16}"
    --workers "${HOOD_V6_WORKERS:-8}" --evaluate-test
    --output-dir "${OUTPUT_DIR}" --wandb-project "${WANDB_PROJECT}"
)
printf 'Node: %s\nPython: %s\nCommit: %s\nOutput: %s\n' \
    "$(hostname)" "${PYTHON_BIN}" "$(git rev-parse HEAD)" "${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# The training entry point performs dataset and CUDA preflight before training.
"${PYTHON_BIN}" -u gpu_multiscale_hic_v6.py "${TRAIN_ARGS[@]}"
printf 'Finished: %s\n' "${OUTPUT_DIR}"
