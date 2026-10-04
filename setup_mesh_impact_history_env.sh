#!/bin/bash -l
# Run once on DeltaAI: bash -l setup_mesh_impact_history_env.sh
set -eo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTORCH_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}
MESH_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history}

if ! type module >/dev/null 2>&1; then
    echo "ERROR: Run this setup in a DeltaAI login shell with the module command available." >&2
    exit 2
fi
module load "${PYTORCH_MODULE}"

# Build against the cluster's native CUDA PyTorch, including on aarch64.
# The module is read-only; all additional dependencies go into this overlay.
python3 - <<'PY'
import torch
from packaging.version import Version

if Version(torch.__version__.split('+')[0]) < Version('2.8'):
    raise SystemExit('The selected system module must provide PyTorch >= 2.8')
if torch.version.cuda is None:
    raise SystemExit('The selected system module must provide a CUDA PyTorch build')
print('System PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)
PY

python3 -m venv --system-site-packages "${MESH_VENV}"
# shellcheck disable=SC1090
source "${MESH_VENV}/bin/activate"
python -m pip install -r "${PROJECT_DIR}/requirements_mesh_impact_history.txt"

cd "${PROJECT_DIR}"
python - <<'PY'
import platform
import sys
import torch
import wandb
import google.protobuf
import train_mesh_impact_history

print('Environment:', sys.executable)
print('Architecture:', platform.machine())
print('PyTorch:', torch.__version__, '| W&B:', wandb.__version__,
      '| Protobuf:', google.protobuf.__version__)
PY

printf 'Environment ready: %s\n' "${MESH_VENV}"
printf 'Submit with: HOOD_MESH_VENV=%q bash %q\n' \
    "${MESH_VENV}" "${PROJECT_DIR}/submit_mesh_impact_history_1704.sh"
