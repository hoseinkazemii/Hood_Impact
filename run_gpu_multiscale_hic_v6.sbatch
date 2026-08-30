#!/bin/bash
#SBATCH --job-name=hood_gpu_v6_1704
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=hood_gpu_v6_1704_%j.out
#SBATCH --error=hood_gpu_v6_1704_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Hood_Impact repository root}"

# DeltaAI's default module set does not provide Python. Load its aarch64
# PyTorch stack explicitly, then restore an inherited/explicit overlay venv.
INHERITED_VIRTUAL_ENV=${VIRTUAL_ENV:-}
PYTORCH_MODULE=${HOOD_V6_PYTORCH_MODULE:-python/miniforge3_pytorch}
module load "${PYTORCH_MODULE}"
V6_VIRTUAL_ENV=${HOOD_V6_VENV:-${INHERITED_VIRTUAL_ENV}}
if [[ -n "${V6_VIRTUAL_ENV}" ]]; then
    if [[ ! -r "${V6_VIRTUAL_ENV}/bin/activate" ]]; then
        echo "FATAL: V6 environment is missing: ${V6_VIRTUAL_ENV}" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${V6_VIRTUAL_ENV}/bin/activate"
fi
PYTHON_COMMAND=${HOOD_V6_PYTHON:-python3}
if ! PYTHON_BIN=$(command -v "${PYTHON_COMMAND}"); then
    echo "FATAL: ${PYTHON_COMMAND} was not found after loading the DeltaAI PyTorch module" >&2
    exit 127
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MKL_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact}

EPOCHS=${HOOD_V6_EPOCHS:-200}
BATCH_SIZE=${HOOD_V6_BATCH_SIZE:-16}
WORKERS=${HOOD_V6_WORKERS:-8}
VAL_DESIGN=${HOOD_VAL_DESIGN:-10}
TEST_DESIGN=${HOOD_TEST_DESIGN:-11}
DATA_ROOT=${HOOD_V6_DATA_ROOT:-Data}
CACHE_DIR=${HOOD_V6_CACHE_DIR:-runs/_gpu_multiscale_hic_v6_cache}
DATASET_DIR="${DATA_ROOT}/HoodImpact_1704_EuroNCAP"
MANIFEST_PATH="${DATASET_DIR}/manifest_1704.csv"

if [[ ! -r "${MANIFEST_PATH}" ]]; then
    echo "FATAL: missing merged 1,704-sample manifest: ${MANIFEST_PATH}" >&2
    echo "Set HOOD_V6_DATA_ROOT if the Data directory is outside the repository." >&2
    exit 2
fi

echo "Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"
echo "Python: ${PYTHON_BIN}"
echo "W&B: project=${WANDB_PROJECT}; mode=${WANDB_MODE}"
echo "Split: validation design ${VAL_DESIGN}; historical reference design ${TEST_DESIGN}"
echo "Training: epochs=${EPOCHS}; batch=${BATCH_SIZE}; workers=${WORKERS}"
echo "Dataset: ${DATASET_DIR} (12 designs x 142 locations)"
echo "Cache: ${CACHE_DIR}"
module list 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON_BIN}" - <<'PY'
import numpy
import pandas
import matplotlib
import torch
import wandb

assert torch.cuda.is_available(), "CUDA is unavailable in the inherited Python environment"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("numpy:", numpy.__version__, "pandas:", pandas.__version__)
print("wandb:", wandb.__version__)
PY

# Validate all 1,704 files, the 12 x 142 design/location mapping, headform
# removal, merged-source geometry agreement, CUDA forward/backward, and the
# 1,000-point decoder before training.
srun "${PYTHON_BIN}" -u gpu_multiscale_hic_v6.py \
    --preflight-only \
    --require-cuda \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --validation-design "${VAL_DESIGN}" \
    --test-design "${TEST_DESIGN}"

srun "${PYTHON_BIN}" -u gpu_multiscale_hic_v6.py \
    --require-cuda \
    --evaluate-test \
    --data-root "${DATA_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --validation-design "${VAL_DESIGN}" \
    --test-design "${TEST_DESIGN}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --wandb-project "${WANDB_PROJECT}"

echo "Latest V6 run directory:"
ls -dt runs/*_gpu_multiscale_hic_v6/ | sed -n '1p'
