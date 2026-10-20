#!/bin/bash -l
#SBATCH --job-name=dd_ladder
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/dd_ladder_%j.out
#SBATCH --error=runs/slurm/dd_ladder_%j.err

# Design-difference oracle ladder (approach 2 of README_design_difference_root_cause.md):
# where does the near-clone difference signal die?
#   1. oracle_interpolation_ceiling.py  model-free ceiling at unseen locations
#   2. oracle_design_code.py            the surrogate's decoder with a design code
#   3. oracle_delta_features.py         explicit edit features + ridge / trees
#   4. probe_frozen_latents.py          linear probe of the frozen no-shortcut run
# Submit from the repository root:  mkdir -p runs/slurm && sbatch run_design_difference_ladder.sbatch
# Select steps with DD_LADDER_STEPS="ceiling design_code delta_features latent_probe".
# Do not enable nounset: module/conda activation may access unset variables.
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
export OMP_NUM_THREADS=${HOOD_MESH_OMP_THREADS:-16}
# Online by default so the design-code oracle's curves reach the W&B website.
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-design-difference}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}
PROBE_RUN=${DD_PROBE_RUN:-${PROJECT_DIR}/runs/mesh_impact_history/20261016_151923_3394500_1704}
STEPS=${DD_LADDER_STEPS:-ceiling design_code delta_features latent_probe}
OUT=${DD_LADDER_OUTPUT_DIR:-${PROJECT_DIR}/runs/design_difference/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_ladder}
mkdir -p "${OUT}"
printf 'Node: %s\nCommit: %s\nData: %s\nOutput: %s\nSteps: %s\nW&B: %s\n' \
    "$(hostname)" "$(git rev-parse HEAD)" "${DATA_ROOT}" "${OUT}" "${STEPS}" "${WANDB_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for step in ${STEPS}; do
    echo "=== ${step} ($(date)) ==="
    case "${step}" in
        ceiling)
            python oracle_interpolation_ceiling.py --data-dir "${DATA_ROOT}" --out-dir "${OUT}/interpolation_ceiling" ;;
        design_code)
            python oracle_design_code.py --data-dir "${DATA_ROOT}" --out-dir "${OUT}/design_code_oracle" --device cuda ;;
        delta_features)
            python oracle_delta_features.py --data-dir "${DATA_ROOT}" --out-dir "${OUT}/delta_features" ;;
        latent_probe)
            if [[ ! -d "${PROBE_RUN}" ]]; then
                echo "ERROR: probe run not found: ${PROBE_RUN} (set DD_PROBE_RUN)" >&2
                exit 2
            fi
            python probe_frozen_latents.py --run-dir "${PROBE_RUN}" --data-dir "${DATA_ROOT}" \
                --out-dir "${OUT}/latent_probe" --device cuda ;;
        *)
            echo "ERROR: unknown step ${step}" >&2
            exit 2 ;;
    esac
done
echo "=== done ($(date)): ${OUT} ==="
