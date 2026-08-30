#!/usr/bin/env bash

# Submit the unchanged GPU multi-scale HIC V6 architecture on the canonical
# 1,704-sample corpus. Environment overrides such as HOOD_V6_VENV,
# HOOD_V6_EPOCHS, and HOOD_V6_BATCH_SIZE are inherited by the Slurm job.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"

SBATCH_SCRIPT="${REPO_ROOT}/run_gpu_multiscale_hic_v6.sbatch"
DATA_ROOT=${HOOD_V6_DATA_ROOT:-Data}
DATASET_DIR="${DATA_ROOT}/HoodImpact_1704_EuroNCAP"
MANIFEST_PATH="${DATASET_DIR}/manifest_1704.csv"
INP_DIR="${DATASET_DIR}/inp_files"
HISTORY_DIR="${DATASET_DIR}/output_history_acc"
EXPECTED_SAMPLES=1704

if ! command -v sbatch >/dev/null 2>&1; then
    echo "FATAL: sbatch is unavailable; run this script on a Slurm login node." >&2
    exit 127
fi

for required in "${SBATCH_SCRIPT}" "${MANIFEST_PATH}" "${INP_DIR}" "${HISTORY_DIR}"; do
    if [[ ! -e "${required}" ]]; then
        echo "FATAL: required path is missing: ${required}" >&2
        echo "Raw .inp and history files are not stored in Git; sync the merged dataset separately." >&2
        exit 2
    fi
done

manifest_rows=$(awk 'END { print NR - 1 }' "${MANIFEST_PATH}")
inp_count=$(find "${INP_DIR}" -maxdepth 1 -type f -name 'HoodImpact_*.inp' | wc -l)
history_count=$(find "${HISTORY_DIR}" -maxdepth 1 -type f \
    -name 'HoodImpact_*_SAE1000_interp1000.csv' | wc -l)

if [[ "${manifest_rows}" -ne "${EXPECTED_SAMPLES}" \
      || "${inp_count}" -ne "${EXPECTED_SAMPLES}" \
      || "${history_count}" -ne "${EXPECTED_SAMPLES}" ]]; then
    echo "FATAL: incomplete 1,704-sample corpus under ${DATASET_DIR}" >&2
    echo "  manifest rows: ${manifest_rows}; input decks: ${inp_count}; histories: ${history_count}" >&2
    exit 2
fi

echo "Submitting GPU V6 on ${EXPECTED_SAMPLES} samples from ${DATASET_DIR}"
job_id=$(sbatch --parsable "$@" "${SBATCH_SCRIPT}")
echo "Submitted Slurm job ${job_id}"
