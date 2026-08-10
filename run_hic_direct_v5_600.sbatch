#!/bin/bash
#SBATCH --job-name=hood_hic_v5
#SBATCH --output=hood_hic_v5_%j.out
#SBATCH --error=hood_hic_v5_%j.err
#SBATCH --cpus-per-task=24
#SBATCH --mem=32g
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --account=bbqg-delta-cpu

# Direct full-resolution HIC15 surrogate for the 600-run EuroNCAP corpus.
# The estimator is CPU-bound and uses all allocated cores for grouped CV and
# the final random forest. No GPU is requested.

set -euo pipefail

source /sw/external/python/anaconda3_cpu/etc/profile.d/conda.sh
conda activate hood_impact
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$0")}
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VAL_DESIGN=${HOOD_VAL_DESIGN:-10}
TEST_DESIGN=${HOOD_TEST_DESIGN:-11}
FINAL_TREES=${HOOD_HIC_TREES:-768}
CV_TREES=${HOOD_HIC_CV_TREES:-192}
SELECTION_MODE=${HOOD_HIC_SELECTION_MODE:-group-cv}

echo "Node: $(hostname)"
echo "Split: validation design ${VAL_DESIGN}; held-out reference design ${TEST_DESIGN}"
echo "Trees: final=${FINAL_TREES}; grouped-CV=${CV_TREES}"

# Fail before the expensive work if any deck/history is absent, the manifest
# mapping is wrong, or the headform cannot be stripped from a design deck.
python hic_direct_surrogate_v5.py --preflight-only

python hic_direct_surrogate_v5.py \
    --validation-design "${VAL_DESIGN}" \
    --test-design "${TEST_DESIGN}" \
    --selection-mode "${SELECTION_MODE}" \
    --evaluate-test \
    --n-estimators "${FINAL_TREES}" \
    --cv-estimators "${CV_TREES}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-24}"

echo "Latest direct-HIC run directory:"
ls -dt runs/*_hic_direct_v5/ | sed -n '1p'
