#!/bin/bash -l
# Run once on an NCSA Delta login node (x86_64, AMD Milan):
#   bash -l setup_mesh_impact_history_env_delta.sh
#
# Delta cannot use the DeltaAI environment: DeltaAI is aarch64 (Grace) with its
# own home directory, and its PyTorch comes from a DeltaAI-only module. This
# builds a separate venv from Delta's python module and installs the CUDA
# PyTorch wheel, pinned to the 2.10.0 that the DeltaAI runs used
# (python/miniforge3_pytorch/2.10.0). Override with HOOD_MESH_TORCH_SPEC and
# HOOD_MESH_TORCH_INDEX if that wheel is unavailable. The venv needs ~6 GB in
# home (check with: quota).
#
# Afterwards, once per Delta account: activate the venv and run `wandb login`
# (Delta's home has no W&B credentials from DeltaAI; online runs fail without them).
set -eo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_MODULE=${HOOD_MESH_PYTORCH_MODULE:-python}
MESH_VENV=${HOOD_MESH_VENV:-${PROJECT_DIR}/.venv-mesh-history-delta}
TORCH_SPEC=${HOOD_MESH_TORCH_SPEC:-torch==2.10.0}
TORCH_INDEX=${HOOD_MESH_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "ERROR: This is the Delta (x86_64) setup; on DeltaAI use setup_mesh_impact_history_env.sh." >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    echo "ERROR: Run this setup in a Delta login shell (bash -l) with the module command available." >&2
    exit 2
fi
module load "${PYTHON_MODULE}"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required; the module gives {sys.version.split()[0]}. "
                     "Pick another with HOOD_MESH_PYTORCH_MODULE (see: module avail python).")
print("Python:", sys.version.split()[0], "|", sys.executable)
PY

python3 -m venv "${MESH_VENV}"
# shellcheck disable=SC1090
source "${MESH_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX}"
# torch is already satisfied here, so the requirements keep the CUDA wheel.
python -m pip install -r "${PROJECT_DIR}/requirements_mesh_impact_history.txt"

cd "${PROJECT_DIR}"
python - <<'PY'
import platform
import sys
import torch
import wandb
import google.protobuf
import train_mesh_impact_history

if torch.version.cuda is None:
    raise SystemExit("The installed PyTorch is a CPU build; set HOOD_MESH_TORCH_INDEX to a CUDA wheel index")
print("Environment:", sys.executable)
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda,
      "| W&B:", wandb.__version__, "| Protobuf:", google.protobuf.__version__)
PY

printf 'Environment ready: %s\n' "${MESH_VENV}"
printf 'Next (once): source %q && wandb login\n' "${MESH_VENV}/bin/activate"
