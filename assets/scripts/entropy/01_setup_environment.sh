#!/usr/bin/env bash

# Create the project environment required by Entropy GPU jobs.
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ENVIRONMENT_PATH=${ENVIRONMENT_PATH:-"${REPOSITORY_ROOT}/.venv"}

if [[ ! -x "${ENVIRONMENT_PATH}/bin/python" ]]; then
    python3 -m venv "${ENVIRONMENT_PATH}"
fi

"${ENVIRONMENT_PATH}/bin/python" -m pip install --upgrade pip
"${ENVIRONMENT_PATH}/bin/python" -m pip install \
    "torch==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu118
"${ENVIRONMENT_PATH}/bin/python" -m pip install -e \
    "${REPOSITORY_ROOT}/packages/msi_dataset_manager"
"${ENVIRONMENT_PATH}/bin/python" -m pip install -e \
    "${REPOSITORY_ROOT}"
"${ENVIRONMENT_PATH}/bin/python" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"Bundled CUDA runtime: {torch.version.cuda}")
print("CUDA hardware will be validated by the staging job on asusgpu6.")
PY
