#!/bin/bash -l
#SBATCH --job-name=mesh_history_1704
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=mesh_history_1704_%j.out
#SBATCH --error=mesh_history_1704_%j.err

# Train MeshImpactHistoryNet from scratch on the 1704 acceleration histories.
# Preferred submission: bash submit_mesh_impact_history_1704.sh
# Direct sbatch also works when submitted from the repository root.
# Do not enable nounset: module/conda activation may access unset variables.
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root or use the submission wrapper}"
PROJECT_DIR=$PWD
PYTORCH_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}
MESH_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history}
if ! type module >/dev/null 2>&1; then
    echo "ERROR: DeltaAI's module command is unavailable in the job environment." >&2
    exit 2
fi
module load "${PYTORCH_MODULE}"
if [[ ! -r "${MESH_VENV}/bin/activate" ]]; then
    echo "ERROR: Environment not found: ${MESH_VENV}" >&2
    echo "Run bash -l setup_mesh_impact_history_env.sh first, or set HOOD_MESH_VENV." >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${MESH_VENV}/bin/activate"
PYTHON_BIN=$(command -v python)

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${HOOD_MESH_OMP_THREADS:-8}
export MKL_NUM_THREADS=${HOOD_MESH_MKL_THREADS:-8}
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-mesh-attention}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}
read -r -a TEST_DESIGNS <<< "${HOOD_MESH_TEST_DESIGNS:-2}"
read -r -a VAL_DESIGNS <<< "${HOOD_MESH_VAL_DESIGNS:-10}"
STAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR=${HOOD_MESH_OUTPUT_DIR:-${PROJECT_DIR}/runs/mesh_impact_history/${STAMP}_${JOB_ID}_1704}
RUN_NAME=${HOOD_MESH_RUN_NAME:-mesh_history_1704_${JOB_ID}}

printf 'Node: %s\nPython: %s\nModule: %s\n' "$(hostname)" "${PYTHON_BIN}" "${PYTORCH_MODULE}"
printf 'Commit: %s\nDataset: %s\nOutput: %s\n' "$(git rev-parse HEAD)" "${DATA_ROOT}" "${OUTPUT_DIR}"
printf 'Test designs: %s | Validation designs: %s\n' "${TEST_DESIGNS[*]}" "${VAL_DESIGNS[*]}"
printf 'W&B: %s | project: %s\n' "${WANDB_MODE}" "${WANDB_PROJECT}"
module list 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# This validates the exact files used by acceleration training and performs a
# new-network CUDA forward/backward check before loading all 1704 meshes.
srun "${PYTHON_BIN}" -u preflight_mesh_impact_history_1704.py \
    --data-root "${DATA_ROOT}" --require-cuda \
    --test-designs "${TEST_DESIGNS[@]}" --val-designs "${VAL_DESIGNS[@]}"

# Dataset selection is explicit; time subsampling remains in utils.utils.Config.
# Optional settings override the Python defaults only when explicitly provided.
TRAIN_ARGS=(
    --data-format euroncap1704 --num-samples 1704 --samples-per-design 142
    --inp-dir "${DATA_ROOT}/inp_files"
    --impact-coords-path "${DATA_ROOT}/ImpactCoords_1704.csv"
    --acceleration-dir "${DATA_ROOT}/output_history_acc"
    --test-designs "${TEST_DESIGNS[@]}" --val-designs "${VAL_DESIGNS[@]}"
    --device cuda --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}"
    --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT}"
)
OPTIONS=(
    EPOCHS:epochs BATCH_SIZE:batch-size LR:lr WEIGHT_DECAY:weight-decay SEED:seed
    WIDTH:width NUM_HEADS:num-heads NUM_LATENTS:num-latents
    LATENT_LAYERS:latent-layers TEMPORAL_LAYERS:temporal-layers DROPOUT:dropout
)
for option in "${OPTIONS[@]}"; do
    variable="HOOD_MESH_${option%%:*}"
    if [[ -n "${!variable}" ]]; then
        TRAIN_ARGS+=("--${option#*:}" "${!variable}")
    fi
done

printf 'Training command:'
printf ' %q' "${PYTHON_BIN}" -u train_mesh_impact_history.py "${TRAIN_ARGS[@]}"
printf '\n'
srun "${PYTHON_BIN}" -u train_mesh_impact_history.py "${TRAIN_ARGS[@]}"
printf 'Finished. Checkpoints, metrics and acceleration histories: %s\n' "${OUTPUT_DIR}"
