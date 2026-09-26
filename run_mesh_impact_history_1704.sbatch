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
# Preferred baseline submission: bash submit_mesh_impact_history_1704_clusterB.sh
# Neighborhood experiment: bash submit_mesh_impact_history_1704_local_sensitivity.sh
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
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-mesh-attention}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_1704_EuroNCAP}
# Fresh-run default: whole cluster B in test, design 11 in validation.
# Preflight rejects test clones in training; saved splits govern resumes.
read -r -a TEST_DESIGNS <<< "${HOOD_MESH_TEST_DESIGNS:-4 5}"
read -r -a VAL_DESIGNS <<< "${HOOD_MESH_VAL_DESIGNS:-11}"
DECODER=${HOOD_MESH_DECODER:-temporal}
STAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR=${HOOD_MESH_OUTPUT_DIR:-${PROJECT_DIR}/runs/mesh_impact_history/${STAMP}_${JOB_ID}_1704}
RUN_NAME=${HOOD_MESH_RUN_NAME:-mesh_history_1704_${JOB_ID}}

printf 'Node: %s\nPython: %s\nModule: %s\n' "$(hostname)" "${PYTHON_BIN}" "${PYTORCH_MODULE}"
printf 'Commit: %s\nDataset: %s\nOutput: %s\n' "$(git rev-parse HEAD)" "${DATA_ROOT}" "${OUTPUT_DIR}"
if [[ -n "${HOOD_MESH_RESUME_FROM:-}" ]]; then
    printf 'Resume: split and architecture restored from %s\n' "${HOOD_MESH_RESUME_FROM}"
else
    printf 'Test designs: %s | Validation designs: %s\n' "${TEST_DESIGNS[*]}" "${VAL_DESIGNS[*]}"
    printf 'Decoder arm: %s\n' "${DECODER}"
fi
printf 'W&B: %s | project: %s | entity: %s\n' "${WANDB_MODE}" "${WANDB_PROJECT}" "${WANDB_ENTITY:-default team}"
module list 2>&1 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# This validates the exact files used by acceleration training and performs a
# new-network CUDA forward/backward check before loading all 1704 meshes.
PREFLIGHT_ARGS=(--data-root "${DATA_ROOT}" --require-cuda
    --test-designs "${TEST_DESIGNS[@]}" --val-designs "${VAL_DESIGNS[@]}"
    --decoder "${DECODER}")
EXPERIMENT_ARGS=()
for option in NEIGHBORHOOD_LAYERS:neighborhood-layers NEIGHBORHOOD_K:neighborhood-k \
    NEIGHBORHOOD_SCALE_MM:neighborhood-scale-mm NEIGHBORHOOD_CHUNK_SIZE:neighborhood-chunk-size \
    IMPACTOR_NODES:impactor-nodes \
    BATCH_SIZE:batch-size; do
    variable="HOOD_MESH_${option%%:*}"
    if [[ -n "${!variable}" ]]; then
        EXPERIMENT_ARGS+=("--${option#*:}" "${!variable}")
    fi
done
PREFLIGHT_ARGS+=("${EXPERIMENT_ARGS[@]}")
if [[ "${HOOD_MESH_ALLOW_CLONE_LEAK:-0}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--allow-clone-leak)
fi
if [[ -n "${HOOD_MESH_RESUME_FROM:-}" ]]; then
    PREFLIGHT_ARGS=(--resume-from "${HOOD_MESH_RESUME_FROM}" --require-cuda)
    if [[ -n "${HOOD_MESH_DATA_ROOT:-}" ]]; then
        PREFLIGHT_ARGS+=(--data-root "${DATA_ROOT}")
    fi
fi
# This is one Python process on one allocated GPU. Run in Slurm's existing
# batch step: an extra srun step can fail with "Error configuring interconnect"
# before Python starts (job 3139891). Inherit Slurm's CUDA_VISIBLE_DEVICES.
printf 'Launch: direct Python in Slurm batch allocation | CUDA_VISIBLE_DEVICES=%s\n' \
    "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'Starting dataset and CUDA preflight.\n'
"${PYTHON_BIN}" -u preflight_mesh_impact_history_1704.py "${PREFLIGHT_ARGS[@]}"

# Dataset selection is explicit; time subsampling remains in utils.utils.Config.
# Optional settings override the Python defaults only when explicitly provided.
TRAIN_ARGS=(
    --data-format euroncap1704 --num-samples 1704 --samples-per-design 142
    --inp-dir "${DATA_ROOT}/inp_files"
    --impact-coords-path "${DATA_ROOT}/ImpactCoords_1704.csv"
    --acceleration-dir "${DATA_ROOT}/output_history_acc"
    --test-designs "${TEST_DESIGNS[@]}" --val-designs "${VAL_DESIGNS[@]}"
    --decoder "${DECODER}"
    --device cuda --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}"
    --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT}"
    "${EXPERIMENT_ARGS[@]}"
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
OPTIONS=(
    HIC_RANGE_THRESHOLD_PERCENT:hic-range-threshold-percent
    EPOCHS:epochs LR:lr WEIGHT_DECAY:weight-decay SEED:seed
    WIDTH:width NUM_HEADS:num-heads NUM_LATENTS:num-latents
    LATENT_LAYERS:latent-layers TEMPORAL_LAYERS:temporal-layers DROPOUT:dropout
)
for option in "${OPTIONS[@]}"; do
    variable="HOOD_MESH_${option%%:*}"
    if [[ -n "${!variable}" ]]; then
        TRAIN_ARGS+=("--${option#*:}" "${!variable}")
    fi
done

if [[ -n "${HOOD_MESH_RESUME_FROM:-}" ]]; then
    # Architecture, split, loss, optimizer settings and total epochs come from
    # the source config, so stale experiment exports cannot change a resume.
    TRAIN_ARGS=(--resume-from "${HOOD_MESH_RESUME_FROM}"
        --device cuda --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}"
        --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    if [[ -n "${HOOD_MESH_DATA_ROOT:-}" ]]; then
        TRAIN_ARGS+=(--inp-dir "${DATA_ROOT}/inp_files"
            --impact-coords-path "${DATA_ROOT}/ImpactCoords_1704.csv"
            --acceleration-dir "${DATA_ROOT}/output_history_acc")
    fi
fi

printf 'Training command:'
printf ' %q' "${PYTHON_BIN}" -u train_mesh_impact_history.py "${TRAIN_ARGS[@]}"
printf '\n'
"${PYTHON_BIN}" -u train_mesh_impact_history.py "${TRAIN_ARGS[@]}"
if [[ "${HOOD_MESH_EXPORT_ATTENTION:-0}" == "1" ]]; then
    ATTENTION_ARGS=("${OUTPUT_DIR}" --device cuda
        --split "${HOOD_MESH_ATTENTION_SPLIT:-test}")
    if [[ -n "${HOOD_MESH_DATA_ROOT:-}" ]]; then
        ATTENTION_ARGS+=(--data-root "${DATA_ROOT}")
    fi
    if [[ "${HOOD_MESH_ATTENTION_SUMMARY_ONLY:-0}" == "1" ]]; then
        ATTENTION_ARGS+=(--summary-only)
    fi
    if [[ -n "${HOOD_MESH_ATTENTION_RUNS:-}" ]]; then
        read -r -a ATTENTION_RUNS <<< "${HOOD_MESH_ATTENTION_RUNS}"
        ATTENTION_ARGS+=(--runs "${ATTENTION_RUNS[@]}")
    fi
    "${PYTHON_BIN}" -u export_mesh_pooling_attention.py "${ATTENTION_ARGS[@]}"
fi
printf 'Finished. Checkpoints, metrics and acceleration histories: %s\n' "${OUTPUT_DIR}"
