#!/bin/bash -l
#SBATCH --job-name=dd_edit_op
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/dd_edit_operator_%j.out
#SBATCH --error=runs/slurm/dd_edit_operator_%j.err

# Difference-first edit operator (approach 3 of README_design_difference_root_cause.md).
# Trains every variant below for each seed, one after the other, each into
# runs/design_difference/edit_operator/<stamp>_<job>_<name>/, logging to W&B online.
#   linear_dt          first-order kernel operator + time-step nuisance (main model)
#   linear             same without the nuisance term
#   attention_dt       antisymmetrized cross-attention operator + nuisance
#   dt_only            no geometry: what the time-step ratio alone explains
#   linear_dt_baseline main model + the baseline design's own curve as context
#   linear_dt_near     main model trained only within 250 mm of the edit
# Submit from the repository root:  mkdir -p runs/slurm && sbatch run_edit_operator.sbatch
# Choose with DD_EDIT_VARIANTS="linear_dt attention_dt" and DD_EDIT_SEEDS="0 1 2".
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
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-design-difference}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}
VARIANTS=${DD_EDIT_VARIANTS:-linear_dt linear attention_dt dt_only linear_dt_baseline linear_dt_near}
SEEDS=${DD_EDIT_SEEDS:-0 1 2}
EPOCHS=${DD_EDIT_EPOCHS:-400}
printf 'Node: %s\nCommit: %s\nData: %s\nVariants: %s\nSeeds: %s\nEpochs: %s\nW&B: %s\n' \
    "$(hostname)" "$(git rev-parse HEAD)" "${DATA_ROOT}" "${VARIANTS}" "${SEEDS}" "${EPOCHS}" "${WANDB_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for seed in ${SEEDS}; do
    for variant in ${VARIANTS}; do
        case "${variant}" in
            linear_dt)          options=(--operator linear --nuisance dt) ;;
            linear)             options=(--operator linear --nuisance none) ;;
            attention_dt)       options=(--operator attention --nuisance dt) ;;
            dt_only)            options=(--operator none --nuisance dt) ;;
            linear_dt_baseline) options=(--operator linear --nuisance dt --context baseline) ;;
            linear_dt_near)     options=(--operator linear --nuisance dt --max-train-distance-mm 250) ;;
            *) echo "ERROR: unknown variant ${variant}" >&2; exit 2 ;;
        esac
        echo "=== ${variant} seed ${seed} ($(date)) ==="
        python train_edit_operator.py --data-dir "${DATA_ROOT}" --device cuda --epochs "${EPOCHS}" \
            --seed "${seed}" --run-name "edit_${variant}_seed${seed}" "${options[@]}"
    done
done
echo "=== done ($(date)) ==="
