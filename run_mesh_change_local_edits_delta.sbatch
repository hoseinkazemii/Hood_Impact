#!/bin/bash -l
#SBATCH --job-name=mesh_change_local_edits
#SBATCH --account=bbqg-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=24:00:00
#SBATCH --output=mesh_change_local_edits_%j.out
#SBATCH --error=mesh_change_local_edits_%j.err

# Mesh-change attention model (DesignChangeHistoryNet) on the local-edit dataset,
# on NCSA Delta. Submit from the repository root on a Delta login node:
#   cd ~/projects/Hood_Impact && git pull && sbatch run_mesh_change_local_edits_delta.sbatch
#
# Data: Data/HoodImpact_LocalEdits, where the Abaqus campaign left it (manifest,
# output_history_acc/, design_changes/, and one design-6 deck in inp_files/).
# Elsewhere: sbatch --export=ALL,HOOD_MESH_DATA_ROOT=<dir> run_mesh_change_local_edits_delta.sbatch
#
# Split (design N54 is design ID 54; every design keeps all 20 of its locations):
#   test        N54  crease cut line at x=30 (diagonal + centre beam 20 mm, +-30 mm)
#   validation  N55  crease cut line at x=100
#   training    the other 70 edits (1400 runs)
#   design 6    not trained on, not in validation/test, not in the scalers. Its true
#               curves are read only after training, as the reference the edit
#               effects are measured from.
# N54 and N55 have the strongest, most repeatable edit effect in the data (mirror
# twins agree with R2 0.94 and 0.90; HIC changes up to 36% and 43%) and no
# location 114. Their two single-beam creases (N69/N70, N71/N72), half-depth
# creases (N59, N60) and standard-depth cut lines (N4, N50) stay in training.
#
# Model: the earlier design-change default (run 3501305) unchanged: inner panel
# (18,182 nodes), 2 neighborhood layers (k=256, 20 mm), FiLM impact conditioning,
# Fourier time (6 pairs, 20-640 Hz), temporal decoder, change tokens = XYZ of
# each design's moved nodes (50-377 nodes, exact from design_changes/), no
# change self-attention, no displacement gate, 63 time points, 100 epochs,
# batch 8, lr 3e-4, weight decay 1e-5, dropout 0.1, seed 42, plain acceleration
# MSE, best checkpoint by validation loss.
#
# Results in runs/mesh_impact_history/<stamp>_<job>_localedits/:
#   local_edit_metrics.json         edit-effect difference R2 (design - design 6),
#                                   train / validation / test, windows 0-5, 5-12,
#                                   12+ ms, by class, HIC changes, and baselines
#   local_edit_effects_N54.png      test: true vs predicted effect at each location
#   local_edit_effects_N55.png      validation, same plot
#   local_edit_runs.csv             every run's scores; local_edit_samples.csv maps
#                                   sample numbers to manifest runs
#   last_checkpoint/                the same for the final epoch
# Bars on the 63-point grid: copy design 6 = 0; sibling copy (training designs
# at the same beams, same sign, same location) N54 0.42, N55 0.52; perfect model
# about 0.8 (mirror-twin repeatability). Plain accel R2 near 0.9 comes from the
# location map alone and is not evidence that the model learned the edit.
#
# Time: 3501305 took 6.4 h on a GH200 for the same model and 1420 training runs;
# Delta's A40 is estimated at ~11 h (not measured). Jobs are charged for time
# used, not the limit. The first job builds 72 kNN graphs (~1 s each, 1.3 GB in
# Data/HoodImpact_LocalEdits/neighbor_graphs); later jobs load them.
# Other GPUs: sbatch --partition=gpuA100x4 --time=40:00:00 ... (1.0x charge, slower
# FP32), or --partition=gpuH200x8 --cpus-per-task=12 --time=13:00:00 (3x charge).
#
# A timeout keeps hood_impact_last_model.pt from the last finished epoch.
# Continue it with the saved split and settings:
#   sbatch --export=ALL,HOOD_MESH_RESUME_FROM=runs/mesh_impact_history/<run> run_mesh_change_local_edits_delta.sbatch
#
# One-time setup (already done for the earlier Delta runs):
#   bash -l setup_mesh_impact_history_env_delta.sh, then wandb login in that venv.
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
PROJECT_DIR=$PWD
if [[ ! -f "${PROJECT_DIR}/train_mesh_impact_history.py" ]]; then
    echo "ERROR: submit from the repository root (no train_mesh_impact_history.py in ${PROJECT_DIR})" >&2
    exit 2
fi
module load "${HOOD_MESH_PYTORCH_MODULE:-python}"
MESH_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history-delta}
if [[ ! -r "${MESH_VENV}/bin/activate" ]]; then
    echo "ERROR: Delta environment not found: ${MESH_VENV}; run bash -l setup_mesh_impact_history_env_delta.sh" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${MESH_VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact-mesh-attention}

DATA_ROOT=${HOOD_MESH_DATA_ROOT:-${PROJECT_DIR}/Data/HoodImpact_LocalEdits}
TEST_DESIGN=${HOOD_LOCAL_EDITS_TEST_DESIGN:-54}
VAL_DESIGN=${HOOD_LOCAL_EDITS_VAL_DESIGN:-55}
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR=${PROJECT_DIR}/runs/mesh_impact_history/$(date +%Y%m%d_%H%M%S)_${JOB_ID}_localedits
RUN_NAME=mesh_change_localedits_test_N${TEST_DESIGN}_val_N${VAL_DESIGN}_${JOB_ID}

COMMON=(--device cuda --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}"
        --wandb-mode "${WANDB_MODE}" --wandb-project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    COMMON+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${HOOD_MESH_RESUME_FROM:-}" ]]; then
    # Split, model, optimizer settings and total epochs come from the saved run.
    ARGS=(--resume-from "${HOOD_MESH_RESUME_FROM}" "${COMMON[@]}")
else
    ARGS=(
        --data-format localedits
        --inp-dir "${DATA_ROOT}/inp_files"
        --impact-coords-path "${DATA_ROOT}/manifest_local_edits.csv"
        --acceleration-dir "${DATA_ROOT}/output_history_acc"
        --test-designs "${TEST_DESIGN}" --val-designs "${VAL_DESIGN}"
        --mesh-nodes inner_panel
        --design-change-nodes true --design-change-layers 0 --design-change-fourier-octaves 2
        --design-change-features xyz --design-change-displacement-gate false
        --decoder temporal --impact-conditioning film
        --time-encoding fourier --fourier-num-frequencies 6
        --fourier-min-frequency-hz 20 --fourier-max-frequency-hz 640
        --neighborhood-layers 2 --neighborhood-k 256 --neighborhood-scale-mm 20 --neighborhood-chunk-size 1024
        --pooling-query-residual false --pool-impactor-nodes false --geometry-film none
        --teacher-forcing-chunks 0 --time-subsample-stride 16
        --width 128 --num-heads 4 --num-latents 256 --latent-layers 3 --temporal-layers 2 --dropout 0.1
        --epochs 100 --batch-size 8 --lr 0.0003 --weight-decay 0.00001 --seed 42
        "${COMMON[@]}"
    )
fi

printf 'Node: %s | Python: %s\nCommit: %s\nDataset: %s\nOutput: %s\n' "$(hostname)" "$(command -v python)" \
    "$(git rev-parse HEAD)" "${DATA_ROOT}" "${OUTPUT_DIR}"
printf 'Resume from: %s\n' "${HOOD_MESH_RESUME_FROM:-none (fresh run: test N${TEST_DESIGN}, validation N${VAL_DESIGN})}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# One Python process in Slurm's batch step (an extra srun step failed in job 3139891).
python -u preflight_local_edits.py "${ARGS[@]}"
printf 'Training command:'
printf ' %q' python -u train_mesh_impact_history.py "${ARGS[@]}"
printf '\n'
python -u train_mesh_impact_history.py "${ARGS[@]}"
printf 'Finished. Metrics, plots and acceleration histories: %s\n' "${OUTPUT_DIR}"
