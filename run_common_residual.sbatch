#!/bin/bash -l
#SBATCH --job-name=common_residual
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/common_residual_%j.out
#SBATCH --error=runs/slurm/common_residual_%j.err

# Common-mode + design-residual surrogate (README_common_residual.md):
#   a_hat_{d,p}(t) = common_p(t) + residual(G_d - G_ref, p)(t)
# trained with absolute + common + residual + same-location pair (+ optional HIC)
# losses. Split: test = cluster B (4, 5), validation = design 11.
#
# Submit from the repository root; arguments after the script go to the trainer:
#     mkdir -p runs/slurm
#     sbatch run_common_residual.sbatch                          # main run
#     sbatch run_common_residual.sbatch --lambda-pair 0          # pair-loss ablation
#     sbatch run_common_residual.sbatch --lambda-hic 0.1         # + HIC term
#     sbatch run_common_residual.sbatch --resume runs/common_residual/<run>   # after a walltime kill
# ~0.3 s per mesh on a GH200: ~7.5 min per epoch, ~12.5 h for the default 100 epochs.
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PROJECT_DIR=$PWD
PYTORCH_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}
MESH_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history}
module load "${PYTORCH_MODULE}"
if [[ ! -r "${MESH_VENV}/bin/activate" ]]; then
    echo "ERROR: Environment not found: ${MESH_VENV}; run bash -l setup_mesh_impact_history_env.sh" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${MESH_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_MESH_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MESH_MKL_THREADS:-8}
# Online by default: per-epoch curves on the W&B website (slow starts are retried).
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-mesh-attention}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}
printf 'Node: %s\nCommit: %s\nData: %s\nArguments: %s\nW&B: %s | project: %s\n' \
    "$(hostname)" "$(git rev-parse HEAD)" "${DATA_ROOT}" "$*" "${WANDB_MODE}" "${WANDB_PROJECT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_common_residual.py --data-dir "${DATA_ROOT}" --device cuda "$@"
