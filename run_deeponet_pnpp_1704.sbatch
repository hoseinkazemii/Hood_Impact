#!/bin/bash -l
#SBATCH --job-name=hood_dpn_1704
#SBATCH --account=bbqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192g
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#
# Direct-HIC DeepONet + PointNet++ on HoodImpact_1704_EuroNCAP
# ============================================================================
# Same network as runs/hic_target_value/20260123_113525_best (the best direct-HIC
# model, trained on the legacy 1000-sample set): temporal_deeponet_pointnetpp.py
# with the HIC_DIRECT_BEST_ARCH preset -- 929,281 parameters, verified to load
# that checkpoint with strict=True.
#
# Data:  12 designs x 142 Euro-NCAP impact locations = 1,704 runs
#        run = 142 * design + loc,  design 0..11,  loc 1..142
# Split: design-wise -- 1 test design (all 142 locations), 1 validation design,
#        the remaining 10 designs for training (1,420 / 142 / 142).
#
# Submit from the repository root:
#     sbatch run_deeponet_pnpp_1704.sbatch
#
# Override anything without editing this file:
#     HOOD_TEST_DESIGNS=7 HOOD_VAL_DESIGNS=3 sbatch run_deeponet_pnpp_1704.sbatch
#     HOOD_EPOCHS=300 HOOD_BATCH_SIZE=8 sbatch run_deeponet_pnpp_1704.sbatch
#     WANDB_MODE=offline sbatch run_deeponet_pnpp_1704.sbatch
#
# NOTE ON THE SPLIT. The 12 designs are 4 clusters of near-clone geometries:
#     A = 0,1,2,3   B = 4,5   C = 6,7,8,9   D = 10,11
# (mean surface distance within a cluster <= 0.05 mm, between clusters 1.3-3.6 mm).
# Holding out a single design leaves its clones in the training set, and a
# zero-learning baseline that just copies a clone's HIC at the same location
# already scores R2 = 0.87-0.99. To measure real generalisation to a new
# geometry, hold out a whole cluster instead, e.g.
#     HOOD_TEST_DESIGNS="0 1 2 3" HOOD_VAL_DESIGNS="4 5" sbatch run_deeponet_pnpp_1704.sbatch
# ============================================================================

# No `set -u`: conda's shell function trips over unset PS1 in a batch shell.
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- environment -----------------------------------------------------------
# `bash -l` sources the profile, so conda is normally already initialised (same
# as the interactive `srun --pty bash -l` session that produced the January run).
# Fall back to sourcing conda.sh when the shell function is missing.
HOOD_CONDA_ENV=${HOOD_CONDA_ENV:-hood_impact}
if ! type conda >/dev/null 2>&1; then
    CONDA_SH=${HOOD_CONDA_SH:-$HOME/.conda/etc/profile.d/conda.sh}
    if [[ -r "${CONDA_SH}" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
    else
        echo "FATAL: conda is not initialised and ${CONDA_SH} is not readable." >&2
        echo "       Set HOOD_CONDA_SH to your conda.sh, or module load your python stack." >&2
        exit 2
    fi
fi
conda activate "${HOOD_CONDA_ENV}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
# Run `wandb login` once on a login node, or export WANDB_MODE=offline.
export WANDB_PROJECT=${WANDB_PROJECT:-hood-impact}

# --- run configuration -----------------------------------------------------
TEST_DESIGNS=${HOOD_TEST_DESIGNS:-2}      # space-separated 0-indexed design ids
VAL_DESIGNS=${HOOD_VAL_DESIGNS:-10}
EPOCHS=${HOOD_EPOCHS:-200}
BATCH_SIZE=${HOOD_BATCH_SIZE:-16}
LR=${HOOD_LR:-3e-4}
WEIGHT_DECAY=${HOOD_WEIGHT_DECAY:-1e-5}
LOCAL_RADIUS=${HOOD_LOCAL_RADIUS:-0.8}
SEED=${HOOD_SEED:-42}

JOB_ID=${SLURM_JOB_ID:-local}
STAMP=$(date +%Y%m%d_%H%M%S)
TAG="test$(echo "${TEST_DESIGNS}" | tr ' ' '-')_val$(echo "${VAL_DESIGNS}" | tr ' ' '-')"
OUTPUT_DIR=${HOOD_OUTPUT_DIR:-./runs/${STAMP}_${JOB_ID}_dpn_pnpp_1704_${TAG}/}
RUN_NAME=${HOOD_RUN_NAME:-dpn_pnpp_1704_${TAG}_${JOB_ID}}

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Node        : $(hostname)"
echo "Job         : ${JOB_ID}"
echo "Commit      : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "Conda env   : ${HOOD_CONDA_ENV} -> $(command -v python)"
echo "Dataset     : Data/HoodImpact_1704_EuroNCAP (12 designs x 142 locations)"
echo "Test design : ${TEST_DESIGNS}"
echo "Val  design : ${VAL_DESIGNS}"
echo "Training    : epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LR} wd=${WEIGHT_DECAY} seed=${SEED}"
echo "Local radius: ${LOCAL_RADIUS}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "W&B         : project=${WANDB_PROJECT} mode=${WANDB_MODE:-online} name=${RUN_NAME}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# --- preflight -------------------------------------------------------------
# Fail in seconds on a bad environment or an incomplete dataset rather than
# after several minutes of mesh parsing.
python - <<'PREFLIGHT'
import os
import sys

import pandas as pd
import torch

assert torch.cuda.is_available(), "CUDA is unavailable in this environment"
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda,
      "| GPU:", torch.cuda.get_device_name(0))

root = "./Data/HoodImpact_1704_EuroNCAP"
inp = os.path.join(root, "inp_files")
acc = os.path.join(root, "output_history_acc")
metrics_path = os.path.join(root, "generation_metrics.csv")
coords_path = os.path.join(root, "ImpactCoords_1704.csv")

# The dataset is deliberately gitignored, so a fresh checkout will not have it.
missing = [p for p in (root, inp, acc, metrics_path, coords_path) if not os.path.exists(p)]
if missing:
    print("PREFLIGHT FAILED: the dataset is not present in this checkout.", file=sys.stderr)
    for p in missing:
        print("  missing:", p, file=sys.stderr)
    print("  Data/HoodImpact_1704_EuroNCAP/ is gitignored on purpose -- copy or", file=sys.stderr)
    print("  symlink the generated dataset into place before submitting.", file=sys.stderr)
    raise SystemExit(1)

n_inp = len([f for f in os.listdir(inp) if f.endswith(".inp")])
n_acc = len([f for f in os.listdir(acc) if f.endswith(".csv")])
metrics = pd.read_csv(metrics_path)
coords = pd.read_csv(coords_path)

problems = []
if n_inp != 1704:
    problems.append(f"{inp}: expected 1704 decks, found {n_inp}")
if n_acc != 1704:
    problems.append(f"{acc}: expected 1704 histories, found {n_acc}")
if len(coords) != 1704:
    problems.append(f"ImpactCoords_1704.csv: expected 1704 rows, found {len(coords)}")
if sorted(metrics["run"]) != list(range(1, 1705)):
    problems.append("generation_metrics.csv: run column is not exactly 1..1704")
bad = metrics.loc[metrics["status"] != "ok", "run"].tolist()
if bad:
    problems.append(f"generation_metrics.csv: {len(bad)} runs are not status=ok: {bad[:10]}")
if metrics["hic15"].isna().any():
    problems.append("generation_metrics.csv: hic15 has missing values")
if not (metrics["run"] == 142 * metrics["design"] + metrics["loc"]).all():
    problems.append("generation_metrics.csv: run != 142 * design + loc")

if problems:
    print("PREFLIGHT FAILED:", file=sys.stderr)
    for p in problems:
        print("  -", p, file=sys.stderr)
    raise SystemExit(1)

print(f"dataset OK: {n_inp} decks, {n_acc} histories, "
      f"{metrics['design'].nunique()} designs x {metrics['loc'].nunique()} locations, "
      f"HIC15 {metrics['hic15'].min():.1f}..{metrics['hic15'].max():.1f}")
PREFLIGHT

# --- train -----------------------------------------------------------------
SRUN=$(command -v srun || true)
${SRUN:+"${SRUN}"} python -u temporal_deeponet_pointnetpp.py \
    --data-format euroncap1704 \
    --arch hic_direct_best \
    --prediction-target hic \
    --test-designs ${TEST_DESIGNS} \
    --val-designs ${VAL_DESIGNS} \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --local-radius "${LOCAL_RADIUS}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --wandb-project "${WANDB_PROJECT}"

echo "============================================================"
echo "Done. Artefacts in ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"
