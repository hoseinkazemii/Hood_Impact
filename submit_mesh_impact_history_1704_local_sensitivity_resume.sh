#!/usr/bin/env bash
# Continue a neighborhood run in a new output directory with a 48-hour limit.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 RUN_DIRECTORY_OR_CHECKPOINT [sbatch options]" >&2
    exit 2
fi
SOURCE=$1
shift
if [[ -d "${SOURCE}" ]]; then
    export HOOD_MESH_RESUME_FROM=$(cd -- "${SOURCE}" && pwd)
elif [[ -f "${SOURCE}" ]]; then
    export HOOD_MESH_RESUME_FROM="$(cd -- "$(dirname -- "${SOURCE}")" && pwd)/$(basename -- "${SOURCE}")"
else
    echo "ERROR: Resume source does not exist: ${SOURCE}" >&2
    exit 2
fi
export HOOD_MESH_RUN_NAME="${HOOD_MESH_RUN_NAME:-mesh_history_1704_clusterD_local_sensitivity_resume}"
printf 'Resume source: %s\n' "${HOOD_MESH_RESUME_FROM}"
exec bash "${PROJECT_DIR}/submit_mesh_impact_history_1704_local_sensitivity.sh" \
    --job-name=mesh_hist_local_resume "$@"
