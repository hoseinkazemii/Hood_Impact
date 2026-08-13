#!/bin/bash
#SBATCH --job-name=hood_gpu_v6
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=hood_gpu_v6_%j.out
#SBATCH --error=hood_gpu_v6_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the Hood_Impact repository root}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MKL_THREADS:-8}

EPOCHS=${HOOD_V6_EPOCHS:-200}
BATCH_SIZE=${HOOD_V6_BATCH_SIZE:-16}
WORKERS=${HOOD_V6_WORKERS:-8}
VAL_DESIGN=${HOOD_VAL_DESIGN:-10}
TEST_DESIGN=${HOOD_TEST_DESIGN:-11}

echo "Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"
echo "Python: $(command -v python)"
echo "Split: validation design ${VAL_DESIGN}; historical reference design ${TEST_DESIGN}"
echo "Training: epochs=${EPOCHS}; batch=${BATCH_SIZE}; workers=${WORKERS}"
module list 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import numpy
import pandas
import matplotlib
import torch

assert torch.cuda.is_available(), "CUDA is unavailable in the inherited Python environment"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("numpy:", numpy.__version__, "pandas:", pandas.__version__)
PY

# Validate all 660 files, the design mapping, headform removal, source geometry
# agreement, CUDA forward/backward, and the 1,000-point decoder before training.
srun python -u gpu_multiscale_hic_v6.py \
    --preflight-only \
    --require-cuda \
    --validation-design "${VAL_DESIGN}" \
    --test-design "${TEST_DESIGN}"

srun python -u gpu_multiscale_hic_v6.py \
    --require-cuda \
    --evaluate-test \
    --validation-design "${VAL_DESIGN}" \
    --test-design "${TEST_DESIGN}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}"

echo "Latest V6 run directory:"
ls -dt runs/*_gpu_multiscale_hic_v6/ | sed -n '1p'
